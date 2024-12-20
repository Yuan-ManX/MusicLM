import math
from functools import wraps, partial

import torch
import torch.nn.functional as F
from torch import nn, einsum
from torchaudio.transforms import Spectrogram, TimeStretch, FrequencyMasking, TimeMasking
import torch.distributed as dist
from distributed import AllGather

from audiolm_pytorch import AudioLM
from audiolm_pytorch.utils import AudioConditionerBase
from x_clip.tokenizer import tokenizer
from vector_quantize_pytorch import ResidualVQ

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange
from beartype.typing import List, Optional, Tuple
from beartype import beartype


def exists(val):
    """
    检查值是否存在（即不为 None）。

    Args:
        val (Any): 要检查的值。

    Returns:
        bool: 如果值存在（即不为 None）则返回 True，否则返回 False。
    """
    return val is not None


def first(it):
    """
    返回可迭代对象中的第一个元素。

    Args:
        it (Iterable[Any]): 可迭代对象。

    Returns:
        Any: 可迭代对象中的第一个元素。
    """
    return it[0]


def default(val, d):
    """
    如果值存在（即不为 None），则返回该值；否则，返回默认值。

    Args:
        val (Any): 要检查的值。
        d (Any): 默认值。

    Returns:
        Any: 如果 val 存在则返回 val，否则返回 d。
    """
    return val if exists(val) else d


def round_down_nearest_multiple(n, divisor):
    """
    将一个数向下取整到最接近的指定除数的倍数。

    例如:
        round_down_nearest_multiple(15, 4) = 12
        round_down_nearest_multiple(17, 5) = 15

    Args:
        n (int): 要取整的数。
        divisor (int): 除数。

    Returns:
        int: 向下取整到最接近的指定除数的倍数。
    """
    return n // divisor * divisor


def Sequential(*modules):
    """
    创建一个顺序的神经网络模块序列，自动过滤掉任何为 None 的模块。

    Args:
        *modules (Sequence[nn.Module]): 任意数量的 nn.Module 实例或 None。

    Returns:
        nn.Sequential: 包含所有非 None 模块的顺序模块序列。
    """
    return nn.Sequential(*filter(exists, modules))


def once(fn):
    """
    创建一个装饰器，确保被装饰的函数只被调用一次。

    Args:
        fn (callable): 要装饰的函数。

    Returns:
        callable: 装饰后的函数。
    """
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)


# tensor functions

def log(t, eps = 1e-20):
    """
    计算张量的对数，并避免出现 log(0) 的情况。

    Args:
        t (torch.Tensor): 输入张量。
        eps (float, optional): 一个非常小的数，用于避免 log(0)。默认为 1e-20。

    Returns:
        torch.Tensor: 对数变换后的张量。
    """
    return torch.log(t.clamp(min = eps))


def l2norm(t):
    """
    对张量进行 L2 归一化。

    Args:
        t (torch.Tensor): 输入张量。

    Returns:
        torch.Tensor: L2 归一化后的张量。
    """
    # 对输入张量进行 L2 归一化，维度为最后一个维度
    return F.normalize(t, p = 2, dim = -1)


def matrix_diag(t):
    """
    从输入矩阵中提取对角线元素。

    Args:
        t (torch.Tensor): 输入矩阵，张量形状为 (..., i, j)。

    Returns:
        torch.Tensor: 对角线元素组成的张量，张量形状为 (..., min(i, j))。
    """
    device = t.device
    # 获取输入张量的最后两个维度大小
    i, j = t.shape[-2:]
    # 计算对角线元素的数量
    num_diag_el = min(i, j)
    # 创建行索引范围
    i_range = torch.arange(i, device = device)
    # 创建列索引范围
    j_range = torch.arange(j, device = device)

    # 创建对角线掩码，标记对角线元素的位置
    diag_mask = rearrange(i_range, 'i -> i 1') == rearrange(j_range, 'j -> 1 j')
    # 使用掩码提取对角线元素
    diag_el = t.masked_select(diag_mask)

    # 重塑对角线元素张量的形状为 (..., min(i, j))
    return rearrange(diag_el, '(b d) -> b d', d = num_diag_el)


# 2d sinusoidal positional embedding
# simple vit paper shows it is good enough compared to learned
# 2D 正弦余弦位置编码
# 简单的 ViT 论文表明，与学习得到的位置编码相比，这种方法已经足够好

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    """
    生成 2D 正弦余弦位置编码。

    Args:
        patches (torch.Tensor): 输入的 patches 张量，形状为 (batch_size, height, width, dim)。
        temperature (int, optional): 温度参数，用于调整频率。默认为 10000。
        dtype (torch.dtype, optional): 返回的张量数据类型。默认为 torch.float32。

    Returns:
        torch.Tensor: 位置编码张量，形状为 (height, width, dim)。
    """
    # 获取 patches 的形状、设备和类型
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    # 创建网格坐标 y 和 x，形状分别为 (h, w)
    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    # 确保维度 dim 是 4 的倍数，因为位置编码需要分别对 y 和 x 生成正弦和余弦
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'

    # 生成 omega 序列，从 0 到 1
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    # 应用温度参数调整 omega
    omega = 1. / (temperature ** omega)

    # 对 y 和 x 进行缩放
    y = y.flatten()[:, None] * omega[None, :] # 形状 (h*w, dim//4)
    x = x.flatten()[:, None] * omega[None, :] # 形状 (h*w, dim//4)

    # 生成正弦和余弦位置编码
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1) # 形状 (h*w, dim)
    pe = pe.type(dtype) # 转换数据类型

    # 重塑为 (h, w, dim)
    return rearrange(pe, '(h w) d -> h w d', h = h, w = w)


# biasless layernorm

class LayerNorm(nn.Module):
    """
    自定义的 LayerNorm 层，不使用偏置项。

    Args:
        dim (int): 输入的维度。
        scale (bool, optional): 是否使用可学习的缩放因子。默认为 True。
    """
    def __init__(self, dim, scale = True):
        super().__init__()
        # 如果 scale 为 True，则创建一个可学习的缩放因子 gamma；否则，使用固定为 1 的 gamma
        self.learned_gamma = nn.Parameter(torch.ones(dim)) if scale else None

        # 注册固定的 gamma 和 beta 参数，不作为可训练参数
        self.register_buffer('gamma', torch.ones(dim), persistent = False)
        self.register_buffer('beta', torch.zeros(dim), persistent = False)

    def forward(self, x):
        """
        前向传播方法。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 经过 LayerNorm 处理后的张量。
        """
        # 使用 F.layer_norm 进行归一化
        # 使用 default 函数选择可学习的 gamma 或固定的 gamma
        return F.layer_norm(x, x.shape[-1:], default(self.learned_gamma, self.gamma), self.beta)


# feedforward

class GEGLU(nn.Module):
    """
    GEGLU 激活函数模块。
    将输入张量沿最后一个维度分成两部分，一部分用于门控（Gating），另一部分用于值（Values）。
    然后应用 GeLU 激活函数到门控部分，并将结果与值部分相乘。

    Args:
        None

    Forward Args:
        x (torch.Tensor): 输入张量，形状为 (..., dim)。

    Returns:
        torch.Tensor: 经过 GEGLU 激活后的张量，形状为 (..., dim/2)。
    """
    def forward(self, x):
        # 将输入张量沿最后一个维度分成两部分
        x, gate = x.chunk(2, dim = -1)
        # 对门控部分应用 GeLU 激活函数，并将其与值部分相乘
        return F.gelu(gate) * x


def FeedForward(dim, mult = 4, dropout = 0.):
    """
    前馈网络模块。
    由 LayerNorm、线性变换、GEGLU 激活函数、Dropout 和另一个线性变换组成。

    Args:
        dim (int): 输入和输出的维度。
        mult (int, optional): 隐藏层维度的乘数因子。默认为 4。
        dropout (float, optional): Dropout 概率。默认为 0。

    Returns:
        nn.Sequential: 包含前馈网络各层的有序序列。
    """
    # 计算隐藏层的维度
    dim_hidden = int(dim * mult * 2 / 3)

    return nn.Sequential(
        LayerNorm(dim), # 第一个 LayerNorm 层
        nn.Linear(dim, dim_hidden * 2, bias = False), # 第一个线性变换层，不使用偏置
        GEGLU(), # GEGLU 激活函数
        nn.Dropout(dropout), # Dropout 层
        nn.Linear(dim_hidden, dim, bias = False) # 第二个线性变换层，不使用偏置
    )


# attention

class Attention(nn.Module):
    """
    自注意力机制模块。

    Args:
        dim (int): 输入和输出的维度。
        causal (bool, optional): 是否使用因果掩码。默认为 False。
        dim_head (int, optional): 每个注意力头的维度。默认为 64。
        heads (int, optional): 注意力头的数量。默认为 8。
        dropout (float, optional): Dropout 概率。默认为 0。
        scale (int, optional): 缩放因子，用于调整相似度得分。默认为 8。
    """
    def __init__(
        self,
        dim,
        causal = False,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        scale = 8
    ):
        super().__init__()
        # 注意力头的数量
        self.heads = heads
        # 缩放因子
        self.scale = scale
        # 是否使用因果掩码
        self.causal = causal
        # 内部维度
        inner_dim = dim_head * heads

        # LayerNorm 层
        self.norm = LayerNorm(dim)

        # Dropout 层，用于注意力得分
        self.attn_dropout = nn.Dropout(dropout)

        # 线性变换，用于生成查询（q）
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        # 线性变换，用于生成键（k）和值（v）
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        # 查询的缩放因子参数
        self.q_scale = nn.Parameter(torch.ones(dim_head))
        # 键的缩放因子参数
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        # 输出线性变换和 Dropout 层
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward( 
        self,
        x, # 输入张量，形状为 (batch_size, sequence_length, dim)
        rel_pos_bias = None, # 相对位置偏置，可选
        mask = None  # 注意力掩码，可选
    ):
        # 获取输入张量的批大小、序列长度和设备信息
        b, n, _, device = *x.shape, x.device

        # prenorm
        # 前置 LayerNorm
        x = self.norm(x)

        # project for queries, keys, values
        # 线性变换生成查询（q）、键（k）和值（v）
        q, k, v = self.to_q(x), *self.to_kv(x).chunk(2, dim = -1)

        # split for multi-headed attention
        # 多头注意力：将查询、键和值重塑为 (batch_size, heads, sequence_length, dim_head)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        # qk rmsnorm, technique circulating within brain used to stabilize a 22B parameter vision model training
        # qk RMSNorm，稳定训练大模型的技术
        # 对查询和键进行 L2 归一化
        q, k = map(l2norm, (q, k))
        # 对查询进行缩放
        q = q * self.q_scale
        # 对键进行缩放
        k = k * self.k_scale

        # similarities
        # 计算相似度得分
        # 计算 q 和 k 的点积，并乘以缩放因子
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(rel_pos_bias):
            # 如果存在相对位置偏置，则将其加到相似度得分上
            sim = sim + rel_pos_bias

        if exists(mask):
            # 如果提供了掩码，则使用掩码填充相似度得分
            mask = rearrange(mask, 'b j -> b 1 1 j') # 重塑掩码形状
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max) # 使用掩码填充

        # 如果使用因果掩码，则应用因果掩码
        if self.causal:
            # 获取相似度得分的最后两个维度
            i, j = sim.shape[-2:]
            # 创建上三角掩码
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = x.device).triu(j - i + 1)
            # 使用因果掩码填充
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention
        # 计算注意力权重
        # 对相似度得分进行 softmax 归一化
        attn = sim.softmax(dim = -1)
        # 应用 Dropout
        attn = self.attn_dropout(attn)

        # aggregate
        # 计算注意力权重与值的乘积
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads
        # 合并多头
        # 将多头张量重塑为 (batch_size, sequence_length, heads * dim_head)
        out = rearrange(out, 'b h n d -> b n (h d)')
        # 通过输出线性变换层
        return self.to_out(out)


# transformer

class Transformer(nn.Module):
    """
    Transformer 编码器模块，由多个自注意力层和前馈层组成。

    Args:
        dim (int): 输入和输出的维度。
        depth (int): Transformer 层的数量。
        dim_head (int, optional): 每个注意力头的维度。默认为 64。
        heads (int, optional): 注意力头的数量。默认为 8。
        attn_dropout (float, optional): 注意力层的 Dropout 概率。默认为 0。
        ff_mult (int, optional): 前馈网络中隐藏层维度的乘数因子。默认为 4。
        ff_dropout (float, optional): 前馈层的 Dropout 概率。默认为 0。
    """
    def __init__(
        self,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.
    ):
        super().__init__()
        # 创建一个空的有序模块列表，用于存储 Transformer 层的子模块
        self.layers = nn.ModuleList([])
        # 根据指定的深度，添加 Transformer 层
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout), # 自注意力层
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout), # 前馈层
            ]))

    def forward(
        self,
        x, # 输入张量，形状为 (batch_size, sequence_length, dim)
        rel_pos_bias = None, # 相对位置偏置，可选
        mask = None, # 注意力掩码，可选
        return_all_layers = False # 是否返回所有层的输出，默认为 False
    ):
        # 初始化一个空列表，用于存储每一层的输出
        layers = []

        for attn, ff in self.layers:
            # 自注意力层
            # 自注意力层的输出与输入相加
            x = attn(x, rel_pos_bias = rel_pos_bias, mask = mask) + x
            # 前馈层
            # 前馈层的输出与输入相加
            x = ff(x) + x
            # 将当前层的输出添加到列表中
            layers.append(x)

        if not return_all_layers:
            # 如果不需要返回所有层的输出，则返回最后一层的输出
            return x
        # 如果需要返回所有层的输出，则返回最后一层的输出和所有层的输出张量
        return x, torch.stack(layers[:-1])


# contrastive losses

class SoftmaxContrastiveLearning(nn.Module):
    """
    Softmax 对比学习模块。

    Args:
        layers (int, optional): 对比学习层的数量。默认为 1。
        decoupled_contrastive_learning (bool, optional): 是否使用解耦对比学习。默认为 False。
        init_temp (float, optional): 初始温度参数。默认为 10。
    """
    def __init__(
        self,
        *,
        layers = 1,
        decoupled_contrastive_learning = False,
        init_temp = 10
    ):
        super().__init__()
        # 初始化温度参数，使用 log 缩放
        self.temperatures = nn.Parameter(torch.ones(layers, 1, 1) * math.log(init_temp))
        # 是否使用解耦对比学习
        self.decoupled_contrastive_learning = decoupled_contrastive_learning

        # 初始化 AllGather 模块，用于分布式训练
        self.all_gather = AllGather(dim = 2)

    @property
    def device(self):
        """
        获取模型所在的设备。

        Returns:
            torch.device: 模型所在的设备。
        """
        return next(self.parameters()).device

    def forward(self, audio_latents, text_latents):
        """
        前向传播方法，计算对比学习损失。

        Args:
            audio_latents (torch.Tensor): 音频的潜在表示，形状为 (batch_size, latent_dim)。
            text_latents (torch.Tensor): 文本的潜在表示，形状为 (batch_size, latent_dim)。

        Returns:
            torch.Tensor: 对比学习损失。
        """
        if audio_latents.ndim == 2:
            # 如果音频潜在表示的维度为 2，则添加一个维度，使其形状为 (1, batch_size, latent_dim)
            audio_latents = rearrange(audio_latents, '... -> 1 ...')

        if text_latents.ndim == 2:
            # 如果文本潜在表示的维度为 2，则添加一个维度，使其形状为 (1, batch_size, latent_dim)
            text_latents = rearrange(text_latents, '... -> 1 ...')

        # 获取批大小
        batch = audio_latents.shape[1]

        if self.all_gather.is_distributed:
            # 如果在分布式环境中，则将音频和文本潜在表示堆叠起来
            latents = torch.stack((audio_latents, text_latents))
            # 使用 AllGather 模块收集所有进程的潜在表示
            latents, _ = self.all_gather(latents)
            audio_latents, text_latents = latents

        # 计算音频和文本潜在表示之间的相似度得分
        sims = einsum('l i d, l j d -> l i j', audio_latents, text_latents)

        # 将相似度得分乘以温度参数的指数
        sims = sims * self.temperatures.exp()

        # 计算指数相似度得分
        cosine_sims_exp = sims.exp()

        # 计算对角线元素，即正样本的指数相似度得分
        numerator = matrix_diag(cosine_sims_exp)

        if self.decoupled_contrastive_learning:
            # 如果使用解耦对比学习，则将自身相似度得分掩码掉
            eye = torch.eye(batch, device = self.device, dtype = torch.bool)
            cosine_sims_exp = cosine_sims_exp.masked_fill(eye, 0.)

        # 计算分母中的 i 和 j 维度上的和
        denominator_i = reduce(cosine_sims_exp, 'l i j -> l i', 'sum')
        denominator_j = reduce(cosine_sims_exp, 'l i j -> l j', 'sum')

        # 计算对比学习损失
        contrastive_loss = -log(numerator) + 0.5 * (log(denominator_i) + log(denominator_j))

        # 对每个样本的损失进行平均
        contrastive_loss = reduce(contrastive_loss, 'l n -> l', 'mean')
        # 返回所有层的对比学习损失之和
        return contrastive_loss.sum()


class SigmoidContrastiveLearning(nn.Module):
    """ https://arxiv.org/abs/2303.15343 """
    """
    Sigmoid 对比学习模块。

    Args:
        layers (int, optional): 对比学习层的数量。默认为 1。
        init_temp (float, optional): 初始温度参数。默认为 10。
        init_bias (float, optional): 初始偏置参数。默认为 -10。
    """

    def __init__(
        self,
        *,
        layers = 1,
        init_temp = 10,
        init_bias = -10
    ):
        super().__init__()
        # 初始化温度参数，使用 log 缩放
        self.temperatures = nn.Parameter(torch.ones(layers, 1, 1) * math.log(init_temp))
        # 初始化偏置参数
        self.bias = nn.Parameter(torch.ones(layers, 1, 1) * init_bias)

        # 初始化 AllGather 模块，用于分布式训练
        self.all_gather = AllGather(dim = 1, all_reduce_grads = True)

    @property
    def device(self):
        """
        获取模型所在的设备。

        Returns:
            torch.device: 模型所在的设备。
        """
        return next(self.parameters()).device

    def forward(self, audio_latents, text_latents):
        """
        前向传播方法，计算 Sigmoid 对比学习损失。

        Args:
            audio_latents (torch.Tensor): 音频的潜在表示，形状为 (batch_size, latent_dim)。
            text_latents (torch.Tensor): 文本的潜在表示，形状为 (batch_size, latent_dim)。

        Returns:
            torch.Tensor: Sigmoid 对比学习损失。
        """
        device = self.device

        if audio_latents.ndim == 2:
            # 如果音频潜在表示的维度为 2，则添加一个维度，使其形状为 (1, batch_size, latent_dim)
            audio_latents = rearrange(audio_latents, '... -> 1 ...')

        if text_latents.ndim == 2:
            # 如果文本潜在表示的维度为 2，则添加一个维度，使其形状为 (1, batch_size, latent_dim)
            text_latents = rearrange(text_latents, '... -> 1 ...')

        # 使用 AllGather 模块收集所有进程的文本潜在表示
        text_latents, rank_sizes = self.all_gather(text_latents)

        # 获取文本潜在表示的批大小
        n = text_latents.shape[1]

        # 计算音频和文本潜在表示之间的相似度得分
        sims = einsum('l i d, l j d -> l i j', audio_latents, text_latents)

        # 将相似度得分乘以温度参数的指数，并加上偏置
        sims = sims * self.temperatures.exp() + self.bias

        # 创建标签矩阵，对角线为 1，其余为 0
        labels = torch.eye(n, device = device)

        if exists(rank_sizes):
            # 如果存在 rank_sizes，则根据进程数量分割标签矩阵
            labels_by_ranks = labels.split(rank_sizes.tolist(), dim = 0)
            # 获取当前进程的标签矩阵
            labels = labels_by_ranks[dist.get_rank()]

        # 将标签矩阵转换为与相似度得分相同的形状，并进行缩放
        labels = 2 * rearrange(labels, 'i j -> 1 i j') - torch.ones_like(sims)

        # 计算 Sigmoid 对比学习损失
        return -F.logsigmoid(labels * sims).sum() / n


# Audio Spectrogram Transformer - https://arxiv.org/abs/2104.01778

def pair(t):
    """
    如果输入 t 不是元组，则返回 (t, t)；否则返回 t。

    Args:
        t (Any): 输入值。

    Returns:
        Tuple[Any, Any]: 返回的元组。
    """
    return (t, t) if not isinstance(t, tuple) else t


class AudioSpectrogramTransformer(nn.Module):
    """
    AST-音频频谱Transformer模型。

    Args:
        dim (int): 输入和输出的维度。
        depth (int): Transformer 层的数量。
        patch_size (int or tuple, optional): 图像块的大小。默认为 16。
        dim_head (int, optional): 每个注意力头的维度。默认为 64。
        heads (int, optional): 注意力头的数量。默认为 8。
        attn_dropout (float, optional): 注意力层的 Dropout 概率。默认为 0。
        ff_mult (int, optional): 前馈网络中隐藏层维度的乘数因子。默认为 4。
        ff_dropout (float, optional): 前馈层的 Dropout 概率。默认为 0。
        accept_spec (bool, optional): 是否接受频谱作为输入。默认为 False。
        accept_spec_time_first (bool, optional): 频谱的时间维度是否在第一维。默认为 True。
        spec_n_fft (int, optional): 快速傅里叶变换的窗口大小。默认为 128。
        spec_power (float, optional): 频谱的幂。默认为 2。
        spec_win_length (int, optional): 窗口长度。默认为 24。
        spec_hop_length (int, optional): 跳跃长度。默认为 None。
        spec_pad (int, optional): 填充长度。默认为 0。
        spec_center (bool, optional): 是否以中心为中心进行填充。默认为 True。
        spec_pad_mode (str, optional): 填充模式。默认为 'reflect'。
        spec_aug_stretch_factor (float, optional): 时拉伸因子。默认为 0.8。
        spec_aug_freq_mask (int, optional): 频率掩码参数。默认为 80。
        spec_aug_time_mask (int, optional): 时间掩码参数。默认为 80。
        patch_dropout_prob (float, optional): 图像块 Dropout 概率。默认为 0.25。
    """
    def __init__(
        self,
        dim,
        depth,
        patch_size = 16,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        accept_spec = False,
        accept_spec_time_first = True,
        spec_n_fft = 128,
        spec_power = 2,
        spec_win_length = 24,
        spec_hop_length = None,
        spec_pad = 0,
        spec_center = True,
        spec_pad_mode = 'reflect',
        spec_aug_stretch_factor = 0.8,
        spec_aug_freq_mask = 80,
        spec_aug_time_mask = 80,
        patch_dropout_prob = 0.25
    ):
        super().__init__()
        # 输入和输出的维度
        self.dim = dim
        # Transformer 层的数量
        self.depth = depth

        # 图像块的大小
        self.patch_size = pair(patch_size)
        # 图像块的输入维度
        patch_input_dim = self.patch_size[0] * self.patch_size[1]

        # 将输入图像转换为图像块
        self.to_patch_tokens = Sequential(
            Rearrange('b (h p1) (w p2) -> b h w (p1 p2)', p1 = self.patch_size[0], p2 = self.patch_size[1]), # 重塑张量形状
            nn.LayerNorm(patch_input_dim), # LayerNorm 层
            nn.Linear(patch_input_dim, dim), # 线性变换层
            nn.LayerNorm(dim) # LayerNorm 层
        )

        # 是否接受频谱作为输入
        self.accept_spec = accept_spec
        # 频谱的时间维度是否在第一维
        self.accept_spec_time_first = accept_spec_time_first

        # 定义频谱转换器
        self.spec = Spectrogram(
            n_fft = spec_n_fft, # 快速傅里叶变换的窗口大小
            power = spec_power, # 频谱的幂
            win_length = spec_win_length, # 窗口长度
            hop_length = spec_hop_length, # 跳跃长度
            pad = spec_pad, # 填充长度
            center = spec_center, # 是否以中心为中心进行填充
            pad_mode = spec_pad_mode # 填充模式
        )

        # SpecAugment - seems to be widely used in audio field https://arxiv.org/abs/1904.08779
        # SpecAugment 增强方法
        self.aug = torch.nn.Sequential(
            TimeStretch(spec_aug_stretch_factor, fixed_rate = True), # 时拉伸增强
            FrequencyMasking(freq_mask_param = spec_aug_freq_mask), # 频率掩码增强
            TimeMasking(time_mask_param = spec_aug_time_mask), # 时间掩码增强
        )
        
        # Transformer 编码器
        self.transformer = Transformer(
            dim = dim, # 输入和输出的维度
            depth = depth, # Transformer 层的数量
            dim_head = dim_head, # 每个注意力头的维度
            heads = heads, # 注意力头的数量
            attn_dropout = attn_dropout, # 注意力层的 Dropout 概率
            ff_mult = ff_mult, # 前馈网络中隐藏层维度的乘数因子
            ff_dropout = ff_dropout # 前馈层的 Dropout 概率
        )

        # LayerNorm 层
        self.norm = LayerNorm(dim)

        # patch dropout
        # 图像块 Dropout 概率
        self.patch_dropout_prob = patch_dropout_prob

        # 2d dynamic positional bias
        # 2D 动态位置偏置
        # MLP 隐藏层的维度
        mlp_hidden_dim = dim // 4

        self.dynamic_pos_bias_mlp = nn.Sequential(
            nn.Linear(2, mlp_hidden_dim), # 线性变换层
            nn.SiLU(), # SiLU 激活函数
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim), # 线性变换层
            nn.SiLU(), # SiLU 激活函数
            nn.Linear(mlp_hidden_dim, heads), # 线性变换层
            Rearrange('... i j h -> ... h i j') # 重塑张量形状
        )

    def forward(
        self,
        x, # 输入张量
        force_no_patch_dropout = False, # 是否强制不进行图像块 Dropout
        return_all_layers = False # 是否返回所有 Transformer 层的输出
    ):
        # 获取批大小和设备信息
        batch, device = x.shape[0], x.device
        # 确保输入张量的维度是否符合要求
        # 如果接受频谱输入，则维度应为 3；否则，维度应为 2
        assert (self.accept_spec and x.ndim == 3) or (not self.accept_spec and x.ndim == 2)

        if self.accept_spec and self.accept_spec_time_first:
            # 如果接受频谱输入且时间维度在第一维，则重塑张量形状
            x = rearrange(x, 'b t f -> b f t')

        if not self.accept_spec:
            # 如果不接受频谱输入，则应用频谱转换
            x = self.spec(x)

        if self.training:
            # 如果在训练模式下，则应用数据增强
            x = self.aug(x)

        # automatically crop if audio does not yield a 2d spectrogram that is divisible by patch sizes
        # 如果音频频谱不能被图像块大小整除，则自动裁剪

        # 获取频谱的高度和宽度
        height, width = x.shape[-2:]
        # 获取图像块的高度和宽度
        patch_height, patch_width = self.patch_size

        # 计算频谱的高度和宽度向下取整到最接近的图像块大小的倍数
        rounded_height, rounded_width = map(lambda args: round_down_nearest_multiple(*args), ((height, patch_height), (width, patch_width)))

        if (height, width) != (rounded_height, rounded_width): # just keep printing to be annoying until it is fixed
            # 如果裁剪后的尺寸与原始尺寸不同，则打印警告信息
            # 这里使用 print_once 确保只打印一次
            print_once(f'spectrogram yielded shape of {(height, width)}, but had to be cropped to {(rounded_height, rounded_width)} to be patchified for transformer')

        # 裁剪频谱到计算后的尺寸
        x = x[..., :rounded_height, :rounded_width]

        # to patches
        # 将输入转换为图像块
        x = self.to_patch_tokens(x)

        # get number of patches along height and width
        # 获取沿高度和宽度方向的图像块数量
        _, num_patch_height, num_patch_width, _ = x.shape

        # get 2d relative positions
        # 生成 2D 相对位置网格
        grid = torch.stack(torch.meshgrid(
            torch.arange(num_patch_height, device = device),
            torch.arange(num_patch_width, device = device)
        , indexing = 'ij'), dim = -1)

        # 重塑网格形状
        grid = rearrange(grid, '... c -> (...) c')

        # 2d sinusoidal positional embedding
        # 生成 2D 正弦余弦位置编码并添加到输入张量中
        x = x + posemb_sincos_2d(x)

        # 重塑张量形状
        x = rearrange(x, 'b ... c -> b (...) c')

        # patch dropout
        # 图像块 Dropout
        if self.training and self.patch_dropout_prob > 0. and not force_no_patch_dropout:
            n, device = x.shape[1], x.device

            # 生成批索引
            batch_indices = torch.arange(batch, device = device)
            # 重塑批索引形状
            batch_indices = rearrange(batch_indices, '... -> ... 1')
            # 计算要保留的图像块数量
            num_patches_keep = max(1, int(n * (1 - self.patch_dropout_prob)))
            # 随机选择要保留的图像块索引
            patch_indices_keep = torch.randn(batch, n, device = device).topk(num_patches_keep, dim = -1).indices

            # 根据索引选择图像块
            x = x[batch_indices, patch_indices_keep]

            # 重复网格以匹配批大小
            grid = repeat(grid, '... -> b ...', b = batch)
            # 根据索引选择网格
            grid = grid[batch_indices, patch_indices_keep]

        # 2d relative positional bias
        # 计算 2D 相对位置偏置
        # 计算相对距离
        rel_dist = rearrange(grid, '... i c -> ... i 1 c') - rearrange(grid, '... j c -> ... 1 j c')
        # 计算相对位置偏置
        rel_pos_bias = self.dynamic_pos_bias_mlp(rel_dist.float())

        # attention, what else
        # 应用自注意力机制
        x, all_layers = self.transformer(x, rel_pos_bias = rel_pos_bias, return_all_layers = True)

        # final global average and norm (most recent papers show this is superior to CLS token)
        # 对输出进行全局平均池化和归一化
        # 最近的研究表明，这种方法比使用 CLS 标记更优
        x = reduce(x, 'b n d -> b d', 'mean')

        # 应用归一化
        out = self.norm(x)

        if not return_all_layers:
            # 如果不需要返回所有层的输出，则返回最终输出
            return out

        # 如果需要返回所有层的输出，则返回最终输出和所有层的输出列表
        return out, all_layers


# text transformer

class TextTransformer(nn.Module):
    """
    基于 Transformer 的文本编码器模型。

    Args:
        dim (int): 输入和输出的维度。
        depth (int): Transformer 层的数量。
        num_tokens (int, optional): 词汇表的大小。默认为 tokenizer.vocab_size。
        max_seq_len (int, optional): 最大序列长度。默认为 256。
        dim_head (int, optional): 每个注意力头的维度。默认为 64。
        heads (int, optional): 注意力头的数量。默认为 8。
        attn_dropout (float, optional): 注意力层的 Dropout 概率。默认为 0。
        ff_dropout (float, optional): 前馈层的 Dropout 概率。默认为 0。
        ff_mult (int, optional): 前馈网络中隐藏层维度的乘数因子。默认为 4。
        pad_id (int, optional): 填充标记的 ID。默认为 0。
    """
    @beartype
    def __init__(
        self,
        dim,
        depth,
        num_tokens = tokenizer.vocab_size,
        max_seq_len = 256,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_mult = 4,
        pad_id = 0
    ):
        super().__init__()
        # 输入和输出的维度
        self.dim = dim

        # 词嵌入层
        self.token_emb = nn.Embedding(num_tokens, dim)
        # 位置嵌入层
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        # Transformer 层的数量
        self.depth = depth
        # 最大序列长度
        self.max_seq_len = max_seq_len

        # CLS 标记参数
        self.cls_token = nn.Parameter(torch.randn(dim))

        # Transformer 编码器
        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            ff_mult = ff_mult
        )

        # 填充标记的 ID
        self.pad_id = pad_id
        # LayerNorm 层
        self.norm = LayerNorm(dim)

    @property
    def device(self):
        """
        获取模型所在的设备。

        Returns:
            torch.device: 模型所在的设备。
        """
        return next(self.parameters()).device

    @beartype
    def forward(
        self,
        x = None,
        raw_texts: Optional[List[str]] = None,
        mask = None,
        return_all_layers = False
    ):
        """
        前向传播方法。

        Args:
            x (Optional[torch.Tensor]): 输入张量。
            raw_texts (Optional[List[str]]): 原始文本列表。
            mask (Optional[torch.Tensor]): 注意力掩码。
            return_all_layers (bool): 是否返回所有 Transformer 层的输出。

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]: 模型的输出或包含最终输出和所有层输出的元组。
        """
        assert exists(x) ^ exists(raw_texts)

        if exists(raw_texts):
            # 如果提供了原始文本，则对其进行分词并移动到设备上
            x = tokenizer.tokenize(raw_texts).to(self.device)

        if not exists(mask):
            # 如果没有提供掩码，则生成掩码
            mask = x != self.pad_id

        # 获取批大小、序列长度和设备信息
        b, n, device = *x.shape, x.device

        # token embedding + positional embedding
        # 词嵌入 + 位置嵌入
        x = self.token_emb(x)

        # 序列长度是否超过最大长度
        assert n <= self.max_seq_len, f'text sequence length {n} must be less than {self.max_seq_len}'

        # 添加位置嵌入
        x = x + self.pos_emb(torch.arange(n, device = device))

        # cls tokens, as in bert
        # CLS 标记，如在 BERT 中
        # 重复 CLS 标记
        cls_tokens = repeat(self.cls_token, 'd -> b d', b = b)
        # 打包 CLS 标记和输入
        x, ps = pack([cls_tokens, x], 'b * d')

        # account for attending to cls token with self attention mask
        # 考虑到自注意力掩码中包含 CLS 标记
        # 在掩码前填充一个 True 值
        mask = F.pad(mask, (1, 0), value = True)

        # attention
        # 自注意力
        # 应用 Transformer 编码器
        x, all_layers = self.transformer(x, mask = mask, return_all_layers = True)

        # unpack the cls tokens
        # 解包 CLS 标记
        cls_tokens, _ = unpack(x, ps, 'b * d')

        # 应用 LayerNorm
        out = self.norm(cls_tokens)

        if not return_all_layers:
            return out

        # 如果需要返回所有层的输出，则返回最终输出和所有层的输出列表
        return out, all_layers


# hierarchical cl loss

def interspersed_indices(layers, total_layers):
    """
    计算在多个层中均匀分布的层索引。

    Args:
        layers (int): 需要选择的层数。
        total_layers (int): 总层数。

    Returns:
        torch.Tensor: 均匀分布的层索引。
    """
    assert total_layers >= layers
    # 计算步长
    step = total_layers / layers
    return (torch.arange(0, layers) * step).floor().long()


class MultiLayerContrastiveLoss(nn.Module):
    """
    多层对比学习损失模块。

    Args:
        audio_dim (int): 音频特征的维度。
        text_dim (int): 文本特征的维度。
        dim_latent (int): 潜在空间的维度。
        layers (int): 要选择的层数。
        decoupled_contrastive_learning (bool, optional): 是否使用解耦对比学习。默认为 False。
        sigmoid_contrastive_loss (bool, optional): 是否使用 Sigmoid 对比学习损失。默认为 False。
    """
    def __init__(
        self,
        *,
        audio_dim,
        text_dim,
        dim_latent,
        layers,
        decoupled_contrastive_learning = False,
        sigmoid_contrastive_loss = False
    ):
        super().__init__()
        # 要选择的层数
        self.layers = layers

        # 初始化音频特征相关的参数
        # LayerNorm 层，不进行缩放
        self.audio_norm = LayerNorm(audio_dim, scale = False) 
        # 可学习的缩放因子
        self.audio_gamma = nn.Parameter(torch.ones(layers, 1, audio_dim))
        # 权重矩阵
        self.audio_latent_weight = nn.Parameter(torch.randn(layers, audio_dim, dim_latent))
        # 偏置向量
        self.audio_latent_bias = nn.Parameter(torch.randn(layers, 1, dim_latent))

        # 初始化文本特征相关的参数
        # LayerNorm 层，不进行缩放
        self.text_norm = LayerNorm(text_dim, scale = False)
        # 可学习的缩放因子
        self.text_gamma = nn.Parameter(torch.ones(layers, 1, text_dim))
        # 权重矩阵
        self.text_latent_weight = nn.Parameter(torch.randn(layers, text_dim, dim_latent))
        # 偏置向量
        self.text_latent_bias = nn.Parameter(torch.randn(layers, 1, dim_latent))

        # 根据是否使用 Sigmoid 对比学习损失，选择相应的类
        klass = SigmoidContrastiveLearning if sigmoid_contrastive_loss else partial(SoftmaxContrastiveLearning, decoupled_contrastive_learning = decoupled_contrastive_learning)
        # 实例化对比学习损失类
        self.contrast = klass(layers = layers)

    def forward(self, *, audio_layers, text_layers):
        """
        前向传播方法，计算多层对比学习损失。

        Args:
            audio_layers (torch.Tensor): 音频特征，形状为 (layers, batch_size, audio_dim)。
            text_layers (torch.Tensor): 文本特征，形状为 (layers, batch_size, text_dim)。

        Returns:
            torch.Tensor: 对比学习损失。
        """
        device, batch = audio_layers.device, audio_layers.shape[1]

        # 计算音频特征的全局平均池化
        audio_gap = reduce(audio_layers, 'l b n d -> l b d', 'mean')
        # 对音频特征进行归一化，并应用缩放因子
        audio_embeds = self.audio_norm(audio_gap) * self.audio_gamma
        # 计算音频潜在表示
        audio_latents = einsum('l b d, l d e -> l b e', audio_embeds, self.audio_latent_weight) + self.audio_latent_bias
        # 对音频潜在表示进行 L2 归一化
        audio_latents = l2norm(audio_latents)

        # 提取文本特征的 CLS 标记
        text_cls_tokens = text_layers[:, :, 0]
        # 对文本特征进行归一化，并应用缩放因子
        text_embeds = self.text_norm(text_cls_tokens) * self.text_gamma
        # 计算文本潜在表示
        text_latents = einsum('l b d, l d e -> l b e', text_embeds, self.text_latent_weight) + self.text_latent_bias
        # 对文本潜在表示进行 L2 归一化
        text_latents = l2norm(text_latents)

        # 计算对比学习损失
        return self.contrast(audio_latents, text_latents)


# main classes

class MuLaN(nn.Module):
    """
    多层对齐网络（MuLaN）模型，用于音频和文本的多模态对齐。

    Args:
        audio_transformer (AudioSpectrogramTransformer): 音频的 Transformer 编码器。
        text_transformer (TextTransformer): 文本的 Transformer 编码器。
        dim_latent (int, optional): 潜在空间的维度。默认为 128。
        decoupled_contrastive_learning (bool, optional): 是否使用解耦对比学习。默认为 True。
        hierarchical_contrastive_loss (bool, optional): 是否使用分层对比学习损失。默认为 False。
        hierarchical_contrastive_loss_layers (int, optional): 分层对比学习损失的层数。如果为 None，则使用音频和文本编码器深度的最小值减一。默认为 None。
        sigmoid_contrastive_loss (bool, optional): 是否使用 Sigmoid 对比学习损失。默认为 False。
    """
    @beartype
    def __init__(
        self,
        audio_transformer: AudioSpectrogramTransformer,
        text_transformer: TextTransformer,
        dim_latent = 128,                       # they use 128
        decoupled_contrastive_learning = True,  # think this was used, make it optional
        hierarchical_contrastive_loss = False,
        hierarchical_contrastive_loss_layers = None,
        sigmoid_contrastive_loss = False
    ):
        super().__init__()
        # 潜在空间的维度
        self.dim_latent = dim_latent

        # 音频的 Transformer 编码器
        self.audio = audio_transformer
        # 文本的 Transformer 编码器
        self.text = text_transformer

        # 线性变换，将文本嵌入映射到潜在空间
        self.text_to_latents = nn.Linear(self.text.dim, dim_latent)
        # 线性变换，将音频嵌入映射到潜在空间
        self.audio_to_latents = nn.Linear(self.audio.dim, dim_latent)

        # 根据是否使用 Sigmoid 对比学习损失，选择相应的类
        klass = SigmoidContrastiveLearning if sigmoid_contrastive_loss else partial(SoftmaxContrastiveLearning, decoupled_contrastive_learning = decoupled_contrastive_learning)
        # 实例化对比学习损失类
        self.contrast = klass()

        # 初始化多层对比学习损失为 None
        self.multi_layer_contrastive_learning = None

        if hierarchical_contrastive_loss:
            # 如果使用分层对比学习损失，则计算要选择的层数
            num_layers = default(hierarchical_contrastive_loss_layers, min(audio_transformer.depth, text_transformer.depth) - 1)
            assert num_layers > 0

            # 计算音频和文本编码器中要选择的层索引
            self.register_buffer('text_layers_indices', interspersed_indices(num_layers, text_transformer.depth))
            self.register_buffer('audio_layers_indices', interspersed_indices(num_layers, audio_transformer.depth))

            # 实例化多层对比学习损失类
            self.multi_layer_contrastive_learning = MultiLayerContrastiveLoss(
                audio_dim = self.audio.dim,
                text_dim = self.text.dim,
                dim_latent = dim_latent,
                layers = num_layers,
                decoupled_contrastive_learning = decoupled_contrastive_learning,
                sigmoid_contrastive_loss = sigmoid_contrastive_loss
            )

    def get_audio_latents(
        self,
        wavs,
        return_all_layers = False
    ):
        """
        获取音频的潜在表示。

        Args:
            wavs (torch.Tensor): 输入音频张量。
            return_all_layers (bool, optional): 是否返回所有层的输出。默认为 False。

        Returns:
            torch.Tensor 或 Tuple[torch.Tensor, List[torch.Tensor]]: 音频的潜在表示或包含潜在表示和所有层输出的元组。
        """
        # 获取音频嵌入和所有层的输出
        audio_embeds, audio_layers = self.audio(wavs, return_all_layers = True)
        # 将音频嵌入映射到潜在空间
        audio_latents = self.audio_to_latents(audio_embeds)
        # 对潜在表示进行 L2 归一化
        out = l2norm(audio_latents)

        if not return_all_layers:
            # 如果不需要返回所有层的输出，则返回最终输出
            return out
        # 如果需要返回所有层的输出，则返回最终输出和所有层的输出列表
        return out, audio_layers

    @beartype
    def get_text_latents(
        self,
        texts = None,
        raw_texts: Optional[List[str]] = None,
        return_all_layers = False
    ):
        """
        获取文本的潜在表示。

        Args:
            texts (torch.Tensor, optional): 输入文本张量。
            raw_texts (List[str], optional): 原始文本列表。
            return_all_layers (bool, optional): 是否返回所有层的输出。默认为 False。

        Returns:
            torch.Tensor 或 Tuple[torch.Tensor, List[torch.Tensor]]: 文本的潜在表示或包含潜在表示和所有层输出的元组。
        """
        # 获取文本嵌入和所有层的输出
        text_embeds, text_layers = self.text(texts, raw_texts = raw_texts, return_all_layers = True)
        # 将文本嵌入映射到潜在空间
        text_latents = self.text_to_latents(text_embeds)
        # 对潜在表示进行 L2 归一化
        out = l2norm(text_latents)

        if not return_all_layers:
            # 如果不需要返回所有层的输出，则返回最终输出
            return out
        # 如果需要返回所有层的输出，则返回最终输出和所有层的输出列表
        return out, text_layers

    @beartype
    def forward(
        self,
        wavs,
        texts = None,
        raw_texts: Optional[List[str]] = None,
        return_latents = False,
        return_similarities = False,
        return_pairwise_similarities = False
    ):
        """
        前向传播方法，计算对比学习损失。

        Args:
            wavs (torch.Tensor): 输入音频张量。
            texts (torch.Tensor, optional): 输入文本张量。
            raw_texts (List[str], optional): 原始文本列表。
            return_latents (bool, optional): 是否返回潜在表示。默认为 False。
            return_similarities (bool, optional): 是否返回相似度。默认为 False。
            return_pairwise_similarities (bool, optional): 是否返回成对相似度。默认为 False。

        Returns:
            torch.Tensor 或 Tuple[torch.Tensor, torch.Tensor] 或 torch.Tensor: 对比学习损失或包含音频和文本的潜在表示、相似度或成对相似度的元组。
        """
        # 获取批大小和设备信息
        batch, device = wavs.shape[0], wavs.device

        # 获取音频和文本的潜在表示和所有层的输出
        audio_latents, audio_layers = self.get_audio_latents(wavs, return_all_layers = True)
        text_latents, text_layers = self.get_text_latents(texts, raw_texts = raw_texts, return_all_layers = True)

        if return_latents:
            # 如果需要返回潜在表示，则返回音频和文本的潜在表示
            return audio_latents, text_latents

        if return_similarities:
            # 如果需要返回相似度，则计算并返回音频和文本的相似度
            return einsum('i d, i d -> i', audio_latents, text_latents)

        if return_pairwise_similarities:
            # 如果需要返回成对相似度，则计算并返回成对相似度
            cosine_sim = einsum('i d, j d -> i j', audio_latents, text_latents)
            return cosine_sim
        
        # 计算对比学习损失
        cl_loss = self.contrast(audio_latents, text_latents)

        if not exists(self.multi_layer_contrastive_learning):
            # 如果没有多层对比学习损失，则返回对比学习损失
            return cl_loss

        # 获取要选择的音频和文本层的索引
        audio_layers = audio_layers[self.audio_layers_indices]
        text_layers = text_layers[self.text_layers_indices]

        # whether to do cl loss across all layers, from ViCHA paper https://arxiv.org/abs/2208.13628
        # 计算分层对比学习损失
        hierarchical_cl_loss = self.multi_layer_contrastive_learning(
            audio_layers = audio_layers,
            text_layers = text_layers
        )

        # 返回对比学习损失和分层对比学习损失的总和
        return cl_loss + hierarchical_cl_loss


# Music LM

class MuLaNEmbedQuantizer(AudioConditionerBase):
    """
    MuLaN 嵌入量化器，用于将音频或文本嵌入量化为离散代码，并生成相应的条件嵌入。

    Args:
        mulan (MuLaN): 多层对齐网络（MuLaN）模型。
        conditioning_dims (Tuple[int, ...]): 每个命名空间的条件嵌入维度。
        rq_num_quantizers (int, optional): 残差量化器的量化器数量。默认为 8。
        rq_ema_decay (float, optional): 指数移动平均的衰减率，用于更新码本。默认为 0.9。
        codebook_size (int, optional): 码本大小。默认为 1024。
        namespaces (Tuple[str, ...], optional): 命名空间，用于区分不同类型的数据（如语义、粗粒度、细粒度）。默认为 ('semantic', 'coarse', 'fine')。
    """
    @beartype
    def __init__(
        self,
        mulan: MuLaN,
        conditioning_dims: Tuple[int, ...],
        rq_num_quantizers = 8,
        rq_ema_decay = 0.9,
        codebook_size = 1024,
        namespaces: Tuple[str, ...] = ('semantic', 'coarse', 'fine'),

    ):
        super().__init__()
        # 多层对齐网络（MuLaN）模型
        self.mulan = mulan

        assert len(namespaces) > 0
        # 命名空间列表
        self.namespaces = namespaces
        # 每个命名空间的条件嵌入维度
        self.conditioning_dims = conditioning_dims

        assert len(conditioning_dims) == len(namespaces), 'number of conditioning dimensions must be equal to number of namespaces'

        # 潜在空间的维度
        dim = mulan.dim_latent

        # 初始化残差量化器（ResidualVQ）
        self.rq = ResidualVQ(
            dim = dim, # 潜在空间的维度
            num_quantizers = rq_num_quantizers, # 量化器数量
            codebook_size = codebook_size, # 码本大小
            decay = rq_ema_decay, # 指数移动平均的衰减率
            commitment_weight = 0,    # 不使用承诺损失，仅使用 EMA 更新码本
            kmeans_init = True, # 使用 K-means 初始化码本
            threshold_ema_dead_code = 2, # EMA 死码的阈值
            quantize_dropout = False  # 不使用量化 dropout
        )

        # 潜在空间的维度
        self.dim = dim
        # 量化器数量
        self.num_codebooks = rq_num_quantizers

        # 初始化条件嵌入的参数字典
        self.cond_embeddings = nn.ParameterDict({})

        # 为每个命名空间初始化条件嵌入
        for namespace, conditioning_dim in zip(namespaces, conditioning_dims):
            cond_embeddings = nn.Parameter(torch.randn(rq_num_quantizers, codebook_size, conditioning_dim))
            # 使用正态分布初始化条件嵌入
            nn.init.normal_(cond_embeddings, std = 0.02)

            self.cond_embeddings[namespace] = cond_embeddings
        # 设置默认命名空间为第一个命名空间
        self.set_default_namespace(namespaces[0])

    def parameters(self):
        """
        获取条件嵌入的参数。

        Returns:
            Iterator: 条件嵌入参数的迭代器。
        """
        return self.cond_embeddings.parameters()

    def set_default_namespace(self, namespace):
        """
        设置默认的命名空间。

        Args:
            namespace (str): 要设置的默认命名空间。
        """
        self._default_namespace = namespace

    def forward(
        self,
        wavs = None,
        texts = None,
        namespace = None
    ):
        """
        前向传播方法，将音频或文本嵌入量化并生成条件嵌入。

        Args:
            wavs (torch.Tensor, optional): 输入音频张量。
            texts (torch.Tensor, optional): 输入文本张量。
            namespace (str, optional): 命名空间。默认为 None。

        Returns:
            torch.Tensor: 条件嵌入。
        """
        assert exists(wavs) ^ exists(texts)

        # 设置命名空间为默认命名空间
        namespace = default(namespace, self._default_namespace)
        assert namespace in self.namespaces, f'namespace {namespace} not found'
        # 获取对应命名空间的条件嵌入
        cond_embeddings = self.cond_embeddings[namespace]

        with torch.no_grad():
            self.mulan.eval()

            # sound and language live in joint embedding space because of contrastive learning
            # 由于对比学习，音频和语言共享联合嵌入空间

            if exists(wavs):
                # 获取音频嵌入
                latents = self.mulan.get_audio_latents(wavs)
            elif exists(texts):
                # 获取文本嵌入
                latents = self.mulan.get_text_latents(texts)

        # 对嵌入进行量化
        _, indices, _ = self.rq(latents)

        batch, num_codebooks, dim = indices.shape[0], self.num_codebooks, cond_embeddings.shape[-1]

        # 对条件嵌入和索引进行重复，以便进行聚集操作
        cond_embeddings = repeat(cond_embeddings, 'q c d -> b q c d', b = batch)
        indices = repeat(indices, 'b q -> b q 1 d', q = num_codebooks, d = dim)

        # 根据索引从条件嵌入中选择相应的嵌入
        cond_embeddings = cond_embeddings.gather(2, indices)

        # 重塑张量形状
        return rearrange(cond_embeddings, 'b q 1 d -> b q d')


class MusicLM(nn.Module):
    """
    MusicLM 模型，用于文本到音频的生成。

    Args:
        audio_lm (AudioLM): 音频语言模型。
        mulan_embed_quantizer (MuLaNEmbedQuantizer): MuLaN 嵌入量化器，用于将文本嵌入到联合嵌入空间中。
    """
    @beartype
    def __init__(
        self,
        audio_lm: AudioLM,
        mulan_embed_quantizer: MuLaNEmbedQuantizer
    ):
        super().__init__()
        # 确保 audio_lm 的 audio_conditioner 不存在，因为 mulan 将被外部管理，用于将文本嵌入到联合嵌入空间中以进行文本到音频的合成
        assert not exists(audio_lm.audio_conditioner), 'mulan must not have been passed into AudioLM. it will be managed externally now, embedding the text into the joint embedding space for text-to-audio synthesis'

        # MuLaN 嵌入量化器
        self.mulan_embed_quantizer = mulan_embed_quantizer
        # 音频语言模型
        self.audio_lm = audio_lm

    @property
    def device(self):
        """
        获取模型所在的设备。

        Returns:
            torch.device: 模型所在的设备。
        """
        return next(self.parameters()).device

    @torch.no_grad()
    def forward(
        self,
        text: str,
        num_samples = 1,
        **audio_lm_kwargs
    ):
        """
        前向传播方法，用于生成音频。

        Args:
            text (str): 输入的文本描述。
            num_samples (int, optional): 要生成的样本数量。默认为 1。
            **audio_lm_kwargs (Any): 传递给音频语言模型的额外关键字参数。

        Returns:
            torch.Tensor: 生成的音乐样本。
        """
        self.eval()

        # 对输入文本进行分词并移动到模型所在的设备
        texts = tokenizer.tokenize([text]).to(self.device)

        # 使用 MuLaN 嵌入量化器将文本嵌入到联合嵌入空间中
        text_embeds = self.mulan_embed_quantizer(texts = texts)

        # unable to deal with variable lengthed audio for now
        # 目前无法处理可变长度的音频

        # 初始化样本列表
        samples = []

        # 根据指定的样本数量生成音频
        for _ in range(num_samples):
            music = self.audio_lm(text_embeds = text_embeds, **audio_lm_kwargs)
            samples.append(music)

        # if one sample, just return it
        # 如果只生成一个样本，则直接返回
        if num_samples == 1:
            return first(samples)

        # 获取 MuLaN 模型
        mulan = self.mulan_embed_quantizer.mulan

        # get the one with the highest similarity score, of all the samples
        # 将所有生成的样本与输入文本进行比较，计算相似度
        sims = torch.cat([mulan(texts = texts, wavs = music, return_similarities = True) for music in samples], dim = 0)
        # 找到相似度最高的样本的索引
        top_matching_index = sims.topk(1, dim = 0).indices.item()

        # 返回相似度最高的样本
        return samples[top_matching_index]
