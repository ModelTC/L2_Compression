#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/ATen.h>

__device__
int lower_bound_cu(const float *array, int size, float key)
{
    int first = 0, len = size;
    int half, middle;

    while(len > 0){
        half = len >> 1;
        middle = first + half;
        if(array[middle] < key){
            first = middle + 1;
            len = len - half - 1;
        }
        else{
            len = half;
        }
    }
    return first;
}

__global__ 
void CosStatKernel(const int m , const int n, const float* sorted_matrix_data, const float* queries_data, const float delta, int* rescdf_i, float* rescdf_f, float* respdf_f)
{
    const int batch=1024;
    __shared__ float buf[batch];
    for (int k2=0;k2<m;k2+=batch){
        int end_k=min(m,k2+batch)-k2;
        for (int j=threadIdx.x;j<end_k;j+=blockDim.x){
            buf[j]=sorted_matrix_data[k2+j];
        }
        __syncthreads();
        for (int i=threadIdx.x+blockIdx.y*blockDim.x+blockIdx.x*blockDim.x*gridDim.y;i<n;i+=blockDim.x*gridDim.y*gridDim.x) {
            float query_data_i = queries_data[i];
            int start = lower_bound_cu(buf, end_k, query_data_i - delta);
            int rescdf_i_i = start;
            float rescdf_f_i = 0.0;
            float respdf_f_i = 0.0;
            for (int j = start; j < end_k; j++) {
                if (query_data_i > buf[j] + delta) {
                    rescdf_i_i += 1;
                }
                else if (query_data_i < buf[j] - delta){
                    break;
                }
                else{
                    rescdf_f_i += -0.5 * cos(((query_data_i - buf[j] + delta) * M_PI / delta) / 2) + 0.5;
                    respdf_f_i += ((M_PI/delta) / 4) * sin(((query_data_i - buf[j] + delta) * M_PI /  delta) / 2);
                }
            }
            rescdf_i[i] += rescdf_i_i;
            rescdf_f[i] += rescdf_f_i;
            respdf_f[i] += respdf_f_i;
        }
        __syncthreads();

    }
}

__global__
void CosStatGradKernel(const int m , const int n, const float* matrix_data, const float* queries_data, const float delta, const float* grad_cdf_f, float* res_grad_matrix_f)
{    
    const int batch=512;
    __shared__ float buf1[batch];
    __shared__ float buf2[batch];
    for (int k2=0;k2<n;k2+=batch){
        int end_k=min(n,k2+batch)-k2;
        for (int j=threadIdx.x;j<end_k;j+=blockDim.x){
            buf1[j]=queries_data[k2+j];
            buf2[j]=grad_cdf_f[k2+j];
        }
        __syncthreads();
        for (int j = threadIdx.x+blockIdx.y*blockDim.x+blockIdx.x*blockDim.x*gridDim.y; j < m; j+=blockDim.x*gridDim.y*gridDim.x) {
            float matrix_data_j = matrix_data[j];
            int start = lower_bound_cu(buf1, end_k, matrix_data_j - delta);
            float res_grad_matrix_f_j = 0.0;
            for (int i = start; i < end_k; i++) {
                if (matrix_data_j > buf1[i] + delta){
                    continue;
                }
                else if (matrix_data_j < buf1[i] - delta){
                    break;
                }
                else{
                    res_grad_matrix_f_j += -buf2[i] * ((M_PI/delta) / 4) * sin(((buf1[i] - matrix_data_j + delta) * M_PI /  delta) / 2);
                }
            }
            res_grad_matrix_f[j] += res_grad_matrix_f_j;
        }
        __syncthreads();
    }

}

void CosStatKernelLauncher(const int m , const int n, const float* sorted_matrix_data, const float* queries_data, const float delta, int* rescdf_i, float* rescdf_f, float* respdf_f)
{
	CosStatKernel<<<dim3(32,16,1),512>>>(m, n, sorted_matrix_data, queries_data, delta, rescdf_i, rescdf_f, respdf_f);

// 	cudaError_t err = cudaGetLastError();
// 	if (err != cudaSuccess)
// 	    printf("error in cosstat Output: %s\n", cudaGetErrorString(err));
}

void CosStatGradKernelLauncher(const int m , const int n, const float* matrix_data, const float* queries_data, const float delta, const float* grad_cdf_f, float* res_grad_matrix_f)
{
	CosStatGradKernel<<<dim3(32,16,1),512>>>(m, n, matrix_data, queries_data, delta, grad_cdf_f, res_grad_matrix_f);

// 	cudaError_t err = cudaGetLastError();
// 	if (err != cudaSuccess)
// 	    printf("error in cosstat Output: %s\n", cudaGetErrorString(err));
}

