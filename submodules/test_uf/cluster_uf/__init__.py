import torch
from . import _C

def union_find(edges: torch.Tensor, N: int = None) -> torch.Tensor:
    """
    Args:
        edges: (M,2) int32 CUDA tensor
        N: number of vertices; if None, inferred as max(edges)+1

    Returns:
        roots: (N,) int32 CUDA tensor
    """
    if not torch.is_tensor(edges):
        raise TypeError("edges must be a torch.Tensor")
    if not edges.is_cuda:
        raise ValueError("edges must be a CUDA tensor")
    if edges.dtype != torch.int32:
        raise ValueError("edges must be torch.int32")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must have shape (M,2)")
    if not edges.is_contiguous():
        edges = edges.contiguous()

    if N is None:
        if edges.numel() == 0:
            raise ValueError("Cannot infer N from empty edges; please pass N explicitly.")
        # infer N as max index + 1
        N = int(edges.max().item()) + 1

    return _C.union_find_cuda(edges, int(N))
