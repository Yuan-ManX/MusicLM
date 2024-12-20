import torch
from torch import nn
from torch.autograd import Function
import torch.distributed as dist

from einops import rearrange


def all_gather_same_dim(t):
    """
    在所有进程上收集张量，并确保所有张量具有相同的维度。

    Args:
        t (torch.Tensor): 要收集的张量。

    Returns:
        List[torch.Tensor]: 所有进程上的张量列表。
    """
    # 获取进程总数
    world_size = dist.get_world_size()
    # 为每个进程创建一个空张量
    gathered_tensors = [torch.empty_like(t, device = t.device, dtype = t.dtype) for i in range(world_size)]
    # 在所有进程上收集张量
    dist.all_gather(gathered_tensors, t)
    # 返回收集到的张量列表
    return gathered_tensors


def all_gather_variable_dim(t, dim = 0, sizes = None):
    """
    在所有进程上收集具有可变维度大小的张量。

    Args:
        t (torch.Tensor): 要收集的张量。
        dim (int, optional): 要收集的维度。默认为0。
        sizes (List[int], optional): 每个进程上张量在该维度的大小列表。如果未提供，将自动收集。

    Returns:
        (torch.Tensor, List[int]): 收集到的张量以及每个进程上张量在该维度的大小列表。
    """
    # 获取设备、当前进程排名和进程总数
    device, rank, world_size = t.device, dist.get_rank(), dist.get_world_size()

    if not exists(sizes):
        # 获取当前张量在该维度的大小
        size = torch.tensor(t.shape[dim], device = device, dtype = torch.long)
        # 在所有进程上收集张量大小
        sizes = all_gather_same_dim(size)
        # 将收集到的大小堆叠成一个张量
        sizes = torch.stack(sizes)

    if torch.unique(sizes).numel() == 1:
        # 如果所有进程上的张量在该维度的大小相同，则执行简单的收集
        gathered_tensors = all_gather_same_dim(t)
        # 连接连接后的张量和大小列表
        return torch.cat(gathered_tensors, dim = dim), sizes

    # 获取所有进程中张量在该维度上的最大大小
    max_size = sizes.amax().item()

    # 将张量在该维度上填充到最大大小
    padded_t = pad_dim_to(t, max_size, dim = dim)
    # 在所有进程上收集填充后的张量
    gathered_tensors = all_gather_same_dim(padded_t)

    # 连接收集到的张量
    gathered_tensor = torch.cat(gathered_tensors, dim = dim)
    # 创建一个序列张量 [0, 1, ..., max_size-1]
    seq = torch.arange(max_size, device = device)

    # 创建掩码，标记有效元素的位置
    mask = rearrange(seq, 'j -> 1 j') < rearrange(sizes, 'i -> i 1')
    # 重塑掩码形状
    mask = rearrange(mask, 'i j -> (i j)')
    # 创建一个序列张量 [0, 1, ..., mask.shape[-1]-1]
    seq = torch.arange(mask.shape[-1], device = device)
    # 获取有效元素的索引
    indices = seq[mask]

    # 根据索引选择有效元素
    gathered_tensor = gathered_tensor.index_select(dim, indices)

    # 返回最终的收集张量和大小列表
    return gathered_tensor, sizes


class AllGatherFunction(Function):
    """
    自定义的 AllGatherFunction，用于在前向传播中收集张量，并在反向传播中分配梯度。
    """
    @staticmethod
    def forward(ctx, x, dim, sizes, all_reduce_grads):
        """
        前向传播方法，用于在所有进程上收集张量。

        Args:
            ctx (torch.autograd.function._ContextMethodMixin): 上下文对象，用于保存反向传播所需的信息。
            x (torch.Tensor): 要收集的张量。
            dim (int): 要收集的维度。
            sizes (List[int]): 每个进程上张量在该维度的大小列表。
            all_reduce_grads (bool): 是否在反向传播时对梯度进行全局规约。

        Returns:
            (torch.Tensor, List[int]): 收集到的张量以及每个进程上张量在该维度的大小列表。
        """
        # 在所有进程上收集张量
        x, batch_sizes = all_gather_variable_dim(x, dim = dim, sizes = sizes)
        # 保存反向传播所需的信息到上下文对象中
        ctx.dim = dim
        ctx.all_reduce_grads = all_reduce_grads
        # 将大小列表转换为列表类型
        ctx.batch_sizes = batch_sizes.tolist()
        return x, batch_sizes

    @staticmethod
    def backward(ctx, grads, _):
        """
        反向传播方法，用于分配梯度到各个进程。

        Args:
            ctx (torch.autograd.function._ContextMethodMixin): 上下文对象，包含前向传播保存的信息。
            grads (torch.Tensor): 从后续层反向传播回来的梯度。
            _ (torch.Tensor): 保留参数，未使用。

        Returns:
            (torch.Tensor, None, None, None): 当前进程对应的梯度，以及三个 None 值。
        """
        # 获取每个进程上张量的大小列表和当前进程的排名
        batch_sizes, rank = ctx.batch_sizes, dist.get_rank()
        if ctx.all_reduce_grads:
            # 如果需要全局规约梯度，则对梯度进行全局规约
            dist.all_reduce(grads)

        # 根据每个进程的大小分割梯度
        grads_by_rank = grads.split(batch_sizes, dim = ctx.dim)
        # 返回当前进程对应的梯度
        return grads_by_rank[rank], None, None, None


class AllGather(nn.Module):
    """
    AllGather 模块，用于在分布式训练中收集张量。
    """
    def __init__(
        self,
        dim,
        *,
        all_reduce_grads = False
    ):
        """
        初始化 AllGather 模块。

        Args:
            dim (int): 要收集的维度。
            all_reduce_grads (bool, optional): 是否在反向传播时对梯度进行全局规约。默认为 False。
        """
        super().__init__()
        # 保存要收集的维度
        self.dim = dim
        # 保存是否需要全局规约梯度的标志
        self.all_reduce_grads = all_reduce_grads
        # 判断是否处于分布式训练环境
        self.is_distributed = dist.is_initialized() and dist.get_world_size() > 1

    def forward(
        self,
        x,
        sizes = None
    ):
        """
        前向传播方法，用于在分布式训练中收集张量。

        Args:
            x (torch.Tensor): 输入张量。
            sizes (List[int], optional): 每个进程上张量在该维度的大小列表。默认为 None。

        Returns:
            (torch.Tensor, Optional[List[int]]): 收集到的张量以及每个进程上张量在该维度的大小列表。
        """
        if not self.is_distributed:
            # 如果不是分布式训练，则直接返回输入张量和 None
            return x, None
        # 使用 AllGatherFunction 进行前向传播
        return AllGatherFunction.apply(x, self.dim, sizes, self.all_reduce_grads)
