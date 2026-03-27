#include <iostream>
#include <cuda_runtime.h>

// Macro to handle Fortran 1D array indexing with -2 starting bounds.
// Adjust this if your actual memory allocation bounds differ!
#define IDX(i, j, k, ldx, ldy) (((i) + 2) + ((j) + 2)*(ldx) + ((k) + 2)*(ldx)*(ldy))

// ---------------------------------------------------------
// 1. The CUDA Kernel (Runs on the GPU)
// ---------------------------------------------------------
__global__ void cdu_div_kernel(
    float* U2, const float* U, const float* V, const float* W,
    float Ax, float Ay, float Az,
    int Unx, int Uny, int Unz,
    int ldx, int ldy) 
{
    // Flatten the loops: map thread IDs to 3D grid coordinates
    // We add 1 because Fortran loops for this function start at 1
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    // Boundary check (equivalent to the loop minimums in Fortran)
    if (i <= Unx && j <= Uny && k <= Unz) {
        
        // Calculate U2 just like the Fortran code
        float val = - (
            (Ax * (U[IDX(i+1,j,k,ldx,ldy)] + U[IDX(i,j,k,ldx,ldy)]) * (U[IDX(i+1,j,k,ldx,ldy)] + U[IDX(i,j,k,ldx,ldy)])
           - Ax * (U[IDX(i,j,k,ldx,ldy)] + U[IDX(i-1,j,k,ldx,ldy)]) * (U[IDX(i,j,k,ldx,ldy)] + U[IDX(i-1,j,k,ldx,ldy)]))
          + (Ay * (U[IDX(i,j+1,k,ldx,ldy)] + U[IDX(i,j,k,ldx,ldy)]) * (V[IDX(i+1,j,k,ldx,ldy)] + V[IDX(i,j,k,ldx,ldy)])
           - Ay * (U[IDX(i,j,k,ldx,ldy)] + U[IDX(i,j-1,k,ldx,ldy)]) * (V[IDX(i+1,j-1,k,ldx,ldy)] + V[IDX(i,j-1,k,ldx,ldy)]))
          + (Az * (U[IDX(i,j,k+1,ldx,ldy)] + U[IDX(i,j,k,ldx,ldy)]) * (W[IDX(i+1,j,k,ldx,ldy)] + W[IDX(i,j,k,ldx,ldy)])
           - Az * (U[IDX(i,j,k,ldx,ldy)] + U[IDX(i,j,k-1,ldx,ldy)]) * (W[IDX(i+1,j,k-1,ldx,ldy)] + W[IDX(i,j,k-1,ldx,ldy)]))
        );
        
        U2[IDX(i,j,k,ldx,ldy)] = val;
    }
}
// ---------------------------------------------------------
// 2. The C++ Wrapper (Called by Fortran)
// ---------------------------------------------------------

// Helper macro for catching CUDA errors
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

extern "C" {
    void launch_cdudiv_cuda(
        float* U2, float* U, float* V, float* W, 
        float Ax, float Ay, float Az, 
        int Unx, int Uny, int Unz, 
        int ldx, int ldy, int total_size) 
    {
        size_t bytes = total_size * sizeof(float);
        float *d_U2, *d_U, *d_V, *d_W;

        // Check if a GPU is even available in the container
        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount == 0) {
            std::cerr << "   [CUDA ERROR] No CUDA-capable devices found! Did you pass the GPU flag to Charliecloud/srun?" << std::endl;
            return;
        }

        // Allocate GPU memory with error checking
        CHECK_CUDA_ERROR( cudaMalloc(&d_U2, bytes) );
        CHECK_CUDA_ERROR( cudaMalloc(&d_U, bytes) );
        CHECK_CUDA_ERROR( cudaMalloc(&d_V, bytes) );
        CHECK_CUDA_ERROR( cudaMalloc(&d_W, bytes) );

        // Copy data from Host
        CHECK_CUDA_ERROR( cudaMemcpy(d_U, U, bytes, cudaMemcpyHostToDevice) );
        CHECK_CUDA_ERROR( cudaMemcpy(d_V, V, bytes, cudaMemcpyHostToDevice) );
        CHECK_CUDA_ERROR( cudaMemcpy(d_W, W, bytes, cudaMemcpyHostToDevice) );

        // Define grid and block sizes
        dim3 threadsPerBlock(8, 8, 8); // 512 threads per block
        dim3 numBlocks(
            (Unx + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (Uny + threadsPerBlock.y - 1) / threadsPerBlock.y,
            (Unz + threadsPerBlock.z - 1) / threadsPerBlock.z
        );

        // Launch the kernel
        cdu_div_kernel<<<numBlocks, threadsPerBlock>>>(
            d_U2, d_U, d_V, d_W, Ax, Ay, Az, Unx, Uny, Unz, ldx, ldy
        );

        // Check for launch errors (e.g., invalid grid dimensions)
        CHECK_CUDA_ERROR( cudaGetLastError() );

        // Wait for GPU to finish and check for execution errors (e.g., memory out of bounds)
        CHECK_CUDA_ERROR( cudaDeviceSynchronize() );

        // Copy result back to Host
        CHECK_CUDA_ERROR( cudaMemcpy(U2, d_U2, bytes, cudaMemcpyDeviceToHost) );

        // Free GPU memory
        CHECK_CUDA_ERROR( cudaFree(d_U2) );
        CHECK_CUDA_ERROR( cudaFree(d_U) );
        CHECK_CUDA_ERROR( cudaFree(d_V) );
        CHECK_CUDA_ERROR( cudaFree(d_W) );
    }
}