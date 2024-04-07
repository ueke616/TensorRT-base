#include <cuda_runtime.h> // 导入CUDA运行时库
#include <cmath>          // 导入数学库，用于expf函数

// CUDA设备函数，计算sigmoid函数
// 输入：
// - x: 浮点数输入
// 输出：
// - 返回sigmoid激活函数的结果
static __device__ float sigmoid(float x){
    return 1 / (1 + expf(-x)); // 计算sigmoid公式
}

// CUDA核函数，用于计算并应用SELU激活函数
// 输入：
// - x: 指向输入数组的指针
// - output: 指向输出数组的指针，用于存储激活后的结果
// - n: 输入数组的元素数量
static __global__ void myselu_kernel(const float* x, float* output, int n){
    int position = threadIdx.x + blockDim.x * blockIdx.x; // 计算当前线程对应的全局索引
    if(position >= n) // 如果索引超出数组范围，直接返回
        return;
    output[position] = x[position] * sigmoid(x[position]); // 应用SELU激活函数，并将结果存储在输出数组中
}

// 用于调用SELU核函数的接口函数
// 输入：
// - x: 指向输入数组的指针
// - output: 指向输出数组的指针
// - n: 输入数组的元素数量
// - stream: CUDA流，用于核函数的异步执行
void myselu_inference(const float* x, float* output, int n, cudaStream_t stream){
    const int nthreads = 512; // 定义每个CUDA块中的线程数
    int block_size = n < nthreads ? n : nthreads; // 计算实际每块的线程数，不能超过nthreads
    int grid_size = (n + block_size - 1) / block_size; // 计算需要的块数以覆盖所有元素
    myselu_kernel<<<grid_size, block_size, 0, stream>>>(x, output, n); // 启动核函数
}
