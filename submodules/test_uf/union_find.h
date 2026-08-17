#pragma once
#include <torch/extension.h>

// edges: int32 CUDA tensor, shape [M, 2]
// N: number of vertices
// return: int32 CUDA tensor, shape [N], each is representative/root id (compressed)
torch::Tensor union_find_cuda(torch::Tensor edges, int64_t N);
