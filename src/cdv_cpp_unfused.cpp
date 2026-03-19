#include <cstddef>

// Exact same column-major mapping
#define IDX(i, j, k, ldx, ldy) (((i) + 2) + ((j) + 2)*(ldx) + ((k) + 2)*(ldx)*(ldy))

// ---------------------------------------------------------
// 1. Equivalent to `call set(V2, 0)`
// ---------------------------------------------------------
template <typename T>
void set_cpp_tmpl(T* __restrict__ V2, T val, int Vnx, int Vny, int Vnz, int ldx, int ldy) {
    #pragma omp parallel for collapse(3) schedule(static)
    for (int k = 1; k <= Vnz; ++k) {
        for (int j = 1; j <= Vny; ++j) {
            for (int i = 1; i <= Vnx; ++i) {
                V2[IDX(i,j,k,ldx,ldy)] = val;
            }
        }
    }
}

// ---------------------------------------------------------
// 2. Equivalent to `call CDVdiv(V2, U, V, W)`
// ---------------------------------------------------------
template <typename T>
void CDVdiv_cpp_tmpl(T* __restrict__ V2, const T* __restrict__ U, const T* __restrict__ V, const T* __restrict__ W,
                     T dxmin, T dymin, T dzmin, int Vnx, int Vny, int Vnz, int ldx, int ldy) 
{
    T Ax_div = (T)0.25 / dxmin;
    T Ay_div = (T)0.25 / dymin;
    T Az_div = (T)0.25 / dzmin;

    #pragma omp parallel for collapse(3) schedule(static)
    for (int k = 1; k <= Vnz; ++k) {
        for (int j = 1; j <= Vny; ++j) {
            for (int i = 1; i <= Vnx; ++i) {
                T div_val = (Ay_div*(V[IDX(i,j+1,k,ldx,ldy)] + V[IDX(i,j,k,ldx,ldy)]) * (V[IDX(i,j+1,k,ldx,ldy)] + V[IDX(i,j,k,ldx,ldy)])
                           - Ay_div*(V[IDX(i,j,k,ldx,ldy)] + V[IDX(i,j-1,k,ldx,ldy)]) * (V[IDX(i,j,k,ldx,ldy)] + V[IDX(i,j-1,k,ldx,ldy)]))
                          + (Ax_div*(V[IDX(i+1,j,k,ldx,ldy)] + V[IDX(i,j,k,ldx,ldy)]) * (U[IDX(i,j+1,k,ldx,ldy)] + U[IDX(i,j,k,ldx,ldy)])
                           - Ax_div*(V[IDX(i,j,k,ldx,ldy)] + V[IDX(i-1,j,k,ldx,ldy)]) * (U[IDX(i-1,j+1,k,ldx,ldy)] + U[IDX(i-1,j,k,ldx,ldy)]))
                          + (Az_div*(V[IDX(i,j,k+1,ldx,ldy)] + V[IDX(i,j,k,ldx,ldy)]) * (W[IDX(i,j+1,k,ldx,ldy)] + W[IDX(i,j,k,ldx,ldy)])
                           - Az_div*(V[IDX(i,j,k,ldx,ldy)] + V[IDX(i,j,k-1,ldx,ldy)]) * (W[IDX(i,j+1,k-1,ldx,ldy)] + W[IDX(i,j,k-1,ldx,ldy)]));
                
                V2[IDX(i,j,k,ldx,ldy)] = -div_val;
            }
        }
    }
}

// ---------------------------------------------------------
// 3. Equivalent to `call CDVadv(V2, U, V, W)`
// ---------------------------------------------------------
template <typename T>
void CDVadv_cpp_tmpl(T* __restrict__ V2, const T* __restrict__ U, const T* __restrict__ V, const T* __restrict__ W,
                     T dxmin, T dymin, T dzmin, int Vnx, int Vny, int Vnz, int ldx, int ldy) 
{
    T Ax_adv = (T)0.125 / dxmin;
    T Ay_adv = (T)0.5 / dymin;
    T Az_adv = (T)0.125 / dzmin;

    #pragma omp parallel for collapse(3) schedule(static)
    for (int k = 1; k <= Vnz; ++k) {
        for (int j = 1; j <= Vny; ++j) {
            for (int i = 1; i <= Vnx; ++i) {
                T Uadv = ( U[IDX(i,j,k,ldx,ldy)] + U[IDX(i,j+1,k,ldx,ldy)] + U[IDX(i-1,j,k,ldx,ldy)] + U[IDX(i-1,j+1,k,ldx,ldy)] );
                T Wadv = ( W[IDX(i,j,k,ldx,ldy)] + W[IDX(i,j+1,k,ldx,ldy)] + W[IDX(i,j,k-1,ldx,ldy)] + W[IDX(i,j+1,k-1,ldx,ldy)] );
                
                T adv_val = Ax_adv*(V[IDX(i+1,j,k,ldx,ldy)] - V[IDX(i-1,j,k,ldx,ldy)]) * Uadv
                          + Ay_adv*(V[IDX(i,j+1,k,ldx,ldy)] - V[IDX(i,j-1,k,ldx,ldy)]) * V[IDX(i,j,k,ldx,ldy)]
                          + Az_adv*(V[IDX(i,j,k+1,ldx,ldy)] - V[IDX(i,j,k-1,ldx,ldy)]) * Wadv;
                          
                V2[IDX(i,j,k,ldx,ldy)] = V2[IDX(i,j,k,ldx,ldy)] - adv_val;
            }
        }
    }
}

// ---------------------------------------------------------
// 4. Equivalent to `call multiply(V2, 0.5)`
// ---------------------------------------------------------
template <typename T>
void multiply_cpp_tmpl(T* __restrict__ V2, T val, int Vnx, int Vny, int Vnz, int ldx, int ldy) {
    #pragma omp parallel for schedule(static)
    for (int k = 1; k <= Vnz; ++k) {
        for (int j = 1; j <= Vny; ++j) {
            #pragma GCC ivdep
            for (int i = 1; i <= Vnx; ++i) {
                V2[IDX(i,j,k,ldx,ldy)] *= val;
            }
        }
    }
}

// ---------------------------------------------------------
// 5. The Main Wrapper: Calls them in sequence just like Fortran
// ---------------------------------------------------------
template <typename T>
void launch_cdv_unfused_cpp_tmpl(T* __restrict__ V2, const T* __restrict__ U, const T* __restrict__ V, const T* __restrict__ W,
                                 T dxmin, T dymin, T dzmin, int Vnx, int Vny, int Vnz, int ldx, int ldy) 
{
    set_cpp_tmpl<T>(V2, (T)0.0, Vnx, Vny, Vnz, ldx, ldy);
    CDVdiv_cpp_tmpl<T>(V2, U, V, W, dxmin, dymin, dzmin, Vnx, Vny, Vnz, ldx, ldy);
    CDVadv_cpp_tmpl<T>(V2, U, V, W, dxmin, dymin, dzmin, Vnx, Vny, Vnz, ldx, ldy);
    multiply_cpp_tmpl<T>(V2, (T)0.5, Vnx, Vny, Vnz, ldx, ldy);
}

extern "C" {
    void launch_cdv_unfused_cpp(void* V2, void* U, void* V, void* W, 
                                double dxmin, double dymin, double dzmin, 
                                std::size_t Vnx, std::size_t Vny, std::size_t Vnz, 
                                std::size_t ldx, std::size_t ldy, 
                                std::size_t knd_bytes) 
    {
        if (knd_bytes == 4) {
            launch_cdv_unfused_cpp_tmpl<float>((float*)V2, (float*)U, (float*)V, (float*)W, (float)dxmin, (float)dymin, (float)dzmin, Vnx, Vny, Vnz, ldx, ldy);
        } else if (knd_bytes == 8) {
            launch_cdv_unfused_cpp_tmpl<double>((double*)V2, (double*)U, (double*)V, (double*)W, (double)dxmin, (double)dymin, (double)dzmin, Vnx, Vny, Vnz, ldx, ldy);
        }
    }
}