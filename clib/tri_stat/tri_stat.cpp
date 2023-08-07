#include <torch/torch.h>

// CUDA forward declarations
void TriStatKernelLauncher(const int m , const int n, const float* sorted_matrix_data, const float* queries_data, const float delta, int* rescdf_i, float* rescdf_f, float* respdf_f);
void TriStatGradKernelLauncher(const int m , const int n, const float* matrix_data, const float* queries_data, const float delta, const float* grad_cdf_f, float* res_grad_matrix_f);

void tri_stat_forward_cuda(
    const at::Tensor sorted_matrix,
    const at::Tensor queries,
    const float delta,
    at::Tensor rescdf_i,
    at::Tensor rescdf_f,
    at::Tensor respdf_f)
{
    TriStatKernelLauncher(sorted_matrix.size(0), queries.size(0), sorted_matrix.data_ptr<float>(), queries.data_ptr<float>(), delta, rescdf_i.data_ptr<int>(), rescdf_f.data_ptr<float>(), respdf_f.data_ptr<float>());
}

void tri_stat_backward_cuda(
    const at::Tensor matrix,
    const at::Tensor queries,
    const float delta,
    at::Tensor grad_cdf_f,
    at::Tensor res_grad_matrix_f)
{
    TriStatGradKernelLauncher(matrix.size(0), queries.size(0), matrix.data_ptr<float>(), queries.data_ptr<float>(), delta, grad_cdf_f.data_ptr<float>(), res_grad_matrix_f.data_ptr<float>());
}

int lower_bound(const float *array, int size, float key)
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

void tri_stat(const int m , const int n, const float* sorted_matrix_data, const float* queries_data, const float delta, int* rescdf_i, float* rescdf_f, float* respdf_f)
{
    for (int i = 0; i < n; i++) {
        int start = lower_bound(sorted_matrix_data, m, queries_data[i] - delta);
        rescdf_i[i] += start;
        for (int j = start; j < m; j++) {
            if (queries_data[i] > sorted_matrix_data[j] + delta) {
                rescdf_f[i] += 1;
            }
            else if (queries_data[i] < sorted_matrix_data[j] - delta){
                break;
            }
            else{
                if (queries_data[i] < sorted_matrix_data[j]){
                    rescdf_f[i] += ((queries_data[i] - sorted_matrix_data[j] + delta) * (queries_data[i] - sorted_matrix_data[j] + delta) / delta / delta) / 2;
                    respdf_f[i] += (queries_data[i] - sorted_matrix_data[j] + delta) / delta / delta;
                }
                else{
                    rescdf_f[i] += 1 - ((sorted_matrix_data[j] - queries_data[i] + delta) * (sorted_matrix_data[j] - queries_data[i] + delta) / delta / delta) / 2;
                    respdf_f[i] += (sorted_matrix_data[j] - queries_data[i] + delta) / delta / delta;
                }

            }
        }
    }
}

void tri_stat_back(const int m , const int n, const float* matrix_data, const float* queries_data, const float delta, const float* grad_cdf_f, float* res_grad_matrix_f)
{
    for (int j = 0; j < m; j++) {
        int start = lower_bound(queries_data, n, matrix_data[j] - delta);
        for (int i = start; i < n; i++) {
            if (matrix_data[j] > queries_data[i] + delta){
                continue;
            }
            else if (matrix_data[j] < queries_data[i] - delta){
                break;
            }
            else{
                if (queries_data[i] < matrix_data[j]){
                    res_grad_matrix_f[j] += -(grad_cdf_f[i]) * (queries_data[i] - matrix_data[j] + delta) / delta / delta;
                }
                else{
                    res_grad_matrix_f[j] += -(grad_cdf_f[i]) * (matrix_data[j] - queries_data[i] + delta) / delta / delta;
                }

            }
        }
    }
}




void tri_stat_backward(
    const at::Tensor matrix,
    const at::Tensor queries,
    const float delta,
    at::Tensor grad_cdf_f,
    at::Tensor res_grad_matrix_f)
{

    const int m = matrix.size(0);
    const int n = queries.size(0);

    const float* matrix_data = matrix.data_ptr<float>();
    const float* queries_data = queries.data_ptr<float>();
    float* grad_cdf_f_data = grad_cdf_f.data_ptr<float>();
    float* res_grad_matrix_f_data = res_grad_matrix_f.data_ptr<float>();

    tri_stat_back(m, n, matrix_data, queries_data, delta, grad_cdf_f_data, res_grad_matrix_f_data);
}

void tri_stat_forward(
    const at::Tensor sorted_matrix,
    const at::Tensor queries,
    const float delta,
    at::Tensor rescdf_i,
    at::Tensor rescdf_f,
    at::Tensor respdf_f)
{

    const int m = sorted_matrix.size(0);
    const int n = queries.size(0);

    const float* sorted_matrix_data = sorted_matrix.data_ptr<float>();
    const float* queries_data = queries.data_ptr<float>();
    int* rescdf_i_data = rescdf_i.data_ptr<int>();
    float* rescdf_f_data = rescdf_f.data_ptr<float>();
    float* respdf_f_data = respdf_f.data_ptr<float>();

    tri_stat(m, n, sorted_matrix_data, queries_data, delta, rescdf_i_data, rescdf_f_data, respdf_f_data);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &tri_stat_forward, "tri_stat forward");
    m.def("backward", &tri_stat_backward, "tri_stat backward");
    m.def("forward_cuda", &tri_stat_forward_cuda, "tri_stat forward (CUDA)");
    m.def("backward_cuda", &tri_stat_backward_cuda, "tri_stat backward (CUDA)");
}
