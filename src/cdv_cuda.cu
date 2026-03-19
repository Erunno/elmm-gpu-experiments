#include <iostream>
#include <cuda_runtime.h>

#define IDX(i, j, k, ldx, ldy) (((i) + 2) + ((j) + 2)*(ldx) + ((k) + 2)*(ldx)*(ldy))

#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// ---------------------------------------------------------
// 1. The FUSED CUDA Kernel (Runs on the GPU)
// ---------------------------------------------------------
template <typename T>
__global__ void cdv_fused_kernel(
    T* V2, const T* U, const T* V, const T* W,
    T dxmin, T dymin, T dzmin,
    int Vnx, int Vny, int Vnz,
    int ldx, int ldy) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i <= Vnx && j <= Vny && k <= Vnz) {
        
        // --- PART 1: CDVdiv ---
        T Ax_div = (T)0.25 / dxmin;
        T Ay_div = (T)0.25 / dymin;
        T Az_div = (T)0.25 / dzmin;

        T div_val = - (
            (Ay_div*(V[IDX(i,j+1,k,ldx,ldy)] + V[IDX(i,j,k,ldx,ldy)]) * (V[IDX(i,j+1,k,ldx,ldy)] + V[IDX(i,j,k,ldx,ldy)])
           - Ay_div*(V[IDX(i,j,k,ldx,ldy)] + V[IDX(i,j-1,k,ldx,ldy)]) * (V[IDX(i,j,k,ldx,ldy)] + V[IDX(i,j-1,k,ldx,ldy)]))
          + (Ax_div*(V[IDX(i+1,j,k,ldx,ldy)] + V[IDX(i,j,k,ldx,ldy)]) * (U[IDX(i,j+1,k,ldx,ldy)] + U[IDX(i,j,k,ldx,ldy)])
           - Ax_div*(V[IDX(i,j,k,ldx,ldy)] + V[IDX(i-1,j,k,ldx,ldy)]) * (U[IDX(i-1,j+1,k,ldx,ldy)] + U[IDX(i-1,j,k,ldx,ldy)]))
          + (Az_div*(V[IDX(i,j,k+1,ldx,ldy)] + V[IDX(i,j,k,ldx,ldy)]) * (W[IDX(i,j+1,k,ldx,ldy)] + W[IDX(i,j,k,ldx,ldy)])
           - Az_div*(V[IDX(i,j,k,ldx,ldy)] + V[IDX(i,j,k-1,ldx,ldy)]) * (W[IDX(i,j+1,k-1,ldx,ldy)] + W[IDX(i,j,k-1,ldx,ldy)]))
        );

        // --- PART 2: CDVadv ---
        T Ax_adv = (T)0.125 / dxmin;
        T Ay_adv = (T)0.5 / dymin;
        T Az_adv = (T)0.125 / dzmin;

        T Uadv = ( U[IDX(i,j,k,ldx,ldy)] + U[IDX(i,j+1,k,ldx,ldy)] + U[IDX(i-1,j,k,ldx,ldy)] + U[IDX(i-1,j+1,k,ldx,ldy)] );
        T Wadv = ( W[IDX(i,j,k,ldx,ldy)] + W[IDX(i,j+1,k,ldx,ldy)] + W[IDX(i,j,k-1,ldx,ldy)] + W[IDX(i,j+1,k-1,ldx,ldy)] );
        
        // We subtract the advection term from the divergence term we just calculated!
        T adv_val = div_val - (
             Ax_adv*(V[IDX(i+1,j,k,ldx,ldy)] - V[IDX(i-1,j,k,ldx,ldy)]) * Uadv
           + Ay_adv*(V[IDX(i,j+1,k,ldx,ldy)] - V[IDX(i,j-1,k,ldx,ldy)]) * V[IDX(i,j,k,ldx,ldy)]
           + Az_adv*(V[IDX(i,j,k+1,ldx,ldy)] - V[IDX(i,j,k-1,ldx,ldy)]) * Wadv 
        );

        // --- PART 3: Multiply by 0.5 and write to global memory ONCE ---
        V2[IDX(i,j,k,ldx,ldy)] = adv_val * (T)0.5;
    }
}

// ---------------------------------------------------------
// 2. The C++ Wrapper
// ---------------------------------------------------------
template <typename RealType>
void launch_fused_tmpl(RealType* V2, RealType* U, RealType* V, RealType* W, 
                       double dxmin, double dymin, double dzmin, 
                       std::size_t Vnx, std::size_t Vny, std::size_t Vnz, 
                       std::size_t ldx, std::size_t ldy, std::size_t total_size) 
{
    size_t bytes = total_size * sizeof(RealType);
    RealType *d_V2, *d_U, *d_V, *d_W;

    CHECK_CUDA_ERROR( cudaMalloc(&d_V2, bytes) );
    CHECK_CUDA_ERROR( cudaMalloc(&d_U, bytes) );
    CHECK_CUDA_ERROR( cudaMalloc(&d_V, bytes) );
    CHECK_CUDA_ERROR( cudaMalloc(&d_W, bytes) );

    // Notice we do NOT need to Memcpy V2 to the GPU, because the kernel completely overwrites it!
    // We also don't need cudaMemset, because every cell calculates its own final value directly.
    CHECK_CUDA_ERROR( cudaMemcpy(d_U, U, bytes, cudaMemcpyHostToDevice) );
    CHECK_CUDA_ERROR( cudaMemcpy(d_V, V, bytes, cudaMemcpyHostToDevice) );
    CHECK_CUDA_ERROR( cudaMemcpy(d_W, W, bytes, cudaMemcpyHostToDevice) );

    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((Vnx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (Vny + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (Vnz + threadsPerBlock.z - 1) / threadsPerBlock.z);

    // Cast the double spacing to the correct precision template type
    cdv_fused_kernel<RealType><<<numBlocks, threadsPerBlock>>>(
        d_V2, d_U, d_V, d_W, (RealType)dxmin, (RealType)dymin, (RealType)dzmin, Vnx, Vny, Vnz, ldx, ldy
    );
    CHECK_CUDA_ERROR( cudaGetLastError() );
    CHECK_CUDA_ERROR( cudaDeviceSynchronize() );

    // Bring the final fused array back to the CPU
    CHECK_CUDA_ERROR( cudaMemcpy(V2, d_V2, bytes, cudaMemcpyDeviceToHost) );

    CHECK_CUDA_ERROR( cudaFree(d_V2) ); CHECK_CUDA_ERROR( cudaFree(d_U) ); 
    CHECK_CUDA_ERROR( cudaFree(d_V) ); CHECK_CUDA_ERROR( cudaFree(d_W) );
}

extern "C" {
    void launch_cdv_fused_cuda(void* V2, void* U, void* V, void* W, 
                               double dxmin, double dymin, double dzmin, 
                               std::size_t Vnx, std::size_t Vny, std::size_t Vnz, 
                               std::size_t ldx, std::size_t ldy, std::size_t total_size, 
                               std::size_t knd_bytes) 
    {
        if (knd_bytes == 4) {
            launch_fused_tmpl<float>((float*)V2, (float*)U, (float*)V, (float*)W, dxmin, dymin, dzmin, Vnx, Vny, Vnz, ldx, ldy, total_size);
        } else if (knd_bytes == 8) {
            launch_fused_tmpl<double>((double*)V2, (double*)U, (double*)V, (double*)W, dxmin, dymin, dzmin, Vnx, Vny, Vnz, ldx, ldy, total_size);
        }
    }
}