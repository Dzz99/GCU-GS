#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda.h>
#include <stdexcept>


#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err = (call);                                                   \
    if (err != cudaSuccess) {                                                   \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
                   cudaGetErrorString(err));                                    \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

static inline int div_up(int a, int b) { return (a + b - 1) / b; }

// -----------------------------
// Device: find with path halving
// -----------------------------
__device__ __forceinline__ int dsu_find_halving(int* parent, int x) {
  // Path halving: parent[x] = parent[parent[x]]
  while (true) {
    int p = parent[x];
    int gp = parent[p];
    if (p == gp) return p;       // reached a root or near-root
    parent[x] = gp;             // compress one step
    x = gp;
  }
}

// -----------------------------
// Kernel: init parent[i]=i
// -----------------------------
__global__ void dsu_init_parent(int* parent, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N) parent[tid] = tid;
}

// -----------------------------
// Kernel: one sweep union over all edges
// Rule: always attach higher root id -> lower root id (monotonic).
// Uses atomicCAS to avoid races when multiple threads update parent.
// -----------------------------
__global__ void dsu_union_sweep(const int2* edges, int M, int* parent) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= M) return;

  int u = edges[tid].x;
  int v = edges[tid].y;

  // Union loop: may need retries due to contention / root changes.
  while (true) {
    int ru = dsu_find_halving(parent, u);
    int rv = dsu_find_halving(parent, v);
    if (ru == rv) break;

    int hi = (ru > rv) ? ru : rv;
    int lo = (ru > rv) ? rv : ru;


    // Try to make hi point to lo, only if hi is still a root (parent[hi] == hi)
    int old = atomicCAS(&parent[hi], hi, lo);
    if (old == hi) {
      // success: merged
      break;
    }
    // else: someone else already changed parent[hi], retry
  }
}

// -----------------------------
// Kernel: compress parent chains
// parent[i] = parent[parent[i]]; repeated several rounds
// -----------------------------
__global__ void dsu_compress_round(int* parent, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N) return;
  int p = parent[tid];
  int gp = parent[p];
  parent[tid] = gp;
}

// -----------------------------
// Kernel: write roots = find(i)
// -----------------------------
__global__ void dsu_write_roots(int* parent_mut, int* roots, int N) {
  // parent_in is unused (kept signature flexible); we will use parent_mut to allow halving writes.
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N) return;
  int r = dsu_find_halving(parent_mut, tid);
  roots[tid] = r;
}

// -----------------------------
// Host DSU runner
// -----------------------------
struct UnionFindGPU {
  int N = 0;
  int M = 0;

  int* d_parent = nullptr;
  int2* d_edges = nullptr;
  int* d_roots = nullptr;

  void allocate(int N_, int M_) {
    N = N_;
    M = M_;
    CUDA_CHECK(cudaMalloc(&d_parent, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_edges, sizeof(int2) * M));
    CUDA_CHECK(cudaMalloc(&d_roots, sizeof(int) * N));
  }

  void release() {
    if (d_parent) CUDA_CHECK(cudaFree(d_parent));
    if (d_edges) CUDA_CHECK(cudaFree(d_edges));
    if (d_roots) CUDA_CHECK(cudaFree(d_roots));
    d_parent = nullptr;
    d_edges = nullptr;
    d_roots = nullptr;
    N = M = 0;
  }

  void upload_edges(const std::vector<int2>& edges) {
    if ((int)edges.size() != M) {
      std::fprintf(stderr, "upload_edges: size mismatch\n");
      std::exit(1);
    }
    CUDA_CHECK(cudaMemcpy(d_edges, edges.data(), sizeof(int2) * M, cudaMemcpyHostToDevice));
  }

  // Run DSU:
  // - init parent
  // - do union sweeps for num_iters
  // - compress rounds
  // - output roots
//   void run(int num_union_sweeps = 10, int num_compress_rounds = 5) {
//     constexpr int BS = 256;

//     // init
//     dsu_init_parent<<<div_up(N, BS), BS>>>(d_parent, N);
//     CUDA_CHECK(cudaGetLastError());

//     // union sweeps
//     for (int it = 0; it < num_union_sweeps; ++it) {
//       dsu_union_sweep<<<div_up(M, BS), BS>>>(d_edges, M, d_parent);
//       CUDA_CHECK(cudaGetLastError());
//       // Optional: compress a bit each iteration to accelerate convergence
//       dsu_compress_round<<<div_up(N, BS), BS>>>(d_parent, N);
//       CUDA_CHECK(cudaGetLastError());
//     }

//     // extra compress
//     for (int k = 0; k < num_compress_rounds; ++k) {
//       dsu_compress_round<<<div_up(N, BS), BS>>>(d_parent, N);
//       CUDA_CHECK(cudaGetLastError());
//     }

//     // write roots
//     dsu_write_roots<<<div_up(N, BS), BS>>>(d_parent, d_parent, d_roots, N);
//     CUDA_CHECK(cudaGetLastError());

//     CUDA_CHECK(cudaDeviceSynchronize());
//   }
  void run(cudaStream_t stream, int num_union_sweeps = 20, int num_compress_rounds = 16) {
    constexpr int BS = 256;

    dsu_init_parent<<<div_up(N, BS), BS, 0, stream>>>(d_parent, N);
    CUDA_CHECK(cudaGetLastError());

    for (int it = 0; it < num_union_sweeps; ++it) {
      dsu_union_sweep<<<div_up(M, BS), BS, 0, stream>>>(d_edges, M, d_parent);
      CUDA_CHECK(cudaGetLastError());

      dsu_compress_round<<<div_up(N, BS), BS, 0, stream>>>(d_parent, N);
      CUDA_CHECK(cudaGetLastError());
    }

    for (int k = 0; k < num_compress_rounds; ++k) {
      dsu_compress_round<<<div_up(N, BS), BS, 0, stream>>>(d_parent, N);
      CUDA_CHECK(cudaGetLastError());
    }

    dsu_write_roots<<<div_up(N, BS), BS, 0, stream>>>(d_parent, d_roots, N);
    CUDA_CHECK(cudaGetLastError());
  }

  std::vector<int> download_roots() const {
    std::vector<int> roots(N);
    CUDA_CHECK(cudaMemcpy(roots.data(), d_roots, sizeof(int) * N, cudaMemcpyDeviceToHost));
    return roots;
  }

  std::vector<int> download_parent() const {
    std::vector<int> p(N);
    CUDA_CHECK(cudaMemcpy(p.data(), d_parent, sizeof(int) * N, cudaMemcpyDeviceToHost));
    return p;
  }
};

// --------------------
// Input checks
// --------------------
static void check_edges_tensor(const torch::Tensor& edges) {
  TORCH_CHECK(edges.is_cuda(), "edges must be a CUDA tensor");
  TORCH_CHECK(edges.dtype() == torch::kInt32, "edges must be int32 (torch.int32)");
  TORCH_CHECK(edges.dim() == 2 && edges.size(1) == 2, "edges must have shape (M, 2)");
  TORCH_CHECK(edges.is_contiguous(), "edges must be contiguous");
}

// --------------------
// Main entry
// --------------------
torch::Tensor union_find_cuda(torch::Tensor edges, int64_t N) {
  check_edges_tensor(edges);
  TORCH_CHECK(N > 0, "N must be > 0");

  const int64_t M64 = edges.size(0);
  TORCH_CHECK(M64 >= 0, "M must be >= 0");
  TORCH_CHECK(M64 <= std::numeric_limits<int>::max(), "M too large for int");
  TORCH_CHECK(N <= std::numeric_limits<int>::max(), "N too large for int");

  const int M = static_cast<int>(M64);
  const int Ni = static_cast<int>(N);

  // Output roots tensor on GPU
  auto opts = torch::TensorOptions().dtype(torch::kInt32).device(edges.device());
  torch::Tensor roots = torch::empty({Ni}, opts);

  // CUDA stream from PyTorch
  cudaStream_t stream = at::cuda::getDefaultCUDAStream().stream();

  UnionFindGPU uf;
  uf.allocate(Ni, M);

  // Copy edges from torch tensor to uf.d_edges
  // edges is (M,2) int32; interpret as int2 array
  const int2* src_edges = reinterpret_cast<const int2*>(edges.data_ptr<int>());
  CUDA_CHECK(cudaMemcpyAsync(uf.d_edges, src_edges, sizeof(int2) * M, cudaMemcpyDeviceToDevice, stream));

  // Run union-find core
  uf.run(stream);

  // Copy d_roots -> torch roots
  CUDA_CHECK(cudaMemcpyAsync(roots.data_ptr<int>(), uf.d_roots, sizeof(int) * Ni, cudaMemcpyDeviceToDevice, stream));

  // Ensure completion before freeing uf memory
  CUDA_CHECK(cudaStreamSynchronize(stream));
  uf.release();

  return roots;
}

static void print_tensor_i32_cpu(const torch::Tensor& t, const char* name) {
  auto cpu = t.to(torch::kCPU);
  auto acc = cpu.accessor<int, 1>();
  std::cout << name << " (size=" << cpu.numel() << "): ";
  for (int64_t i = 0; i < cpu.numel(); ++i) {
    std::cout << acc[i] << (i + 1 == cpu.numel() ? "" : ", ");
  }
  std::cout << "\n";
}

int main() {
  // 1) 初始化 PyTorch（C++端）
  // 注：一般不需要显式 init；但我们至少检查 CUDA 可用
  if (!torch::cuda::is_available()) {
    std::cerr << "CUDA is not available. Please build/run with CUDA.\n";
    return 1;
  }

  // 2) 构造一个可预期的图
  // N = 6
  // component A: 0-1-2  -> root should become 0
  // component B: 3-4    -> root should become 3
  // component C: 5      -> root should become 5
  const int64_t N = 6;

  // edges list (undirected edges; DSU union doesn't care direction)
  // We'll create M=3 edges: (0,1), (1,2), (3,4)
  std::vector<int> edges_host = {
    0, 1,
    0, 5,
    1, 2,
    3, 4
  };
  const int64_t M = 3;

  // 3) 创建 edges tensor on CPU then move to CUDA
  // shape (M,2), dtype int32
  auto edges_cpu = torch::from_blob(edges_host.data(), {M, 2}, torch::TensorOptions().dtype(torch::kInt32)).clone();
  auto edges = edges_cpu.to(torch::kCUDA);

  // 4) 调用 union_find_cuda
  auto roots = union_find_cuda(edges, N);

  // 5) 打印 roots（拷回 CPU）
  print_tensor_i32_cpu(roots, "roots");

  // 6) 额外：验证预期（非必须）
  auto roots_cpu = roots.to(torch::kCPU);
  auto r = roots_cpu.accessor<int, 1>();

  bool ok = true;
  // nodes 0,1,2 should have root 0
  ok &= (r[0] == 0 && r[1] == 0 && r[2] == 0);
  // nodes 3,4 should have root 3
  ok &= (r[3] == 3 && r[4] == 3);
  // node 5 should have root 5
  ok &= (r[5] == 5);

  if (!ok) {
    std::cerr << "[FAIL] Unexpected roots.\n";
    return 2;
  }

  std::cout << "[OK] Union-Find result matches expected components.\n";
  return 0;
}