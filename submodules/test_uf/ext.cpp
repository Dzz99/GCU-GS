#include <torch/extension.h>
#include "union_find.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "union_find_cuda",
    &union_find_cuda,
    "Union-Find on GPU (edges int32 CUDA (M,2), N int) -> roots int32 CUDA (N)"
  );
}
