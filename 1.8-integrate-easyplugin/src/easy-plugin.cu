#include "onnx-tensorrt-release-8.6/onnxplugin.hpp"

using namespace ONNXPlugin;

static __device__ float sigmoid(float x){
    return 1 / (x + expf(-x));
}

static __global__ void MYSELU_kernel_fp32(const float* x, float* output, int edge) {
    int position = threadIdx.x + blockDim.x * blockIdx.x;
    if(position >= edge) return;

    output[position] = x[position] * sigmoid(x[position]);
}

class MYSELU: public TRTPlugin {
    public:
        SetupPlugin(MYSELU);

        virtual void config_finish() override{
            printf("\033[33minit MYSELU config: %s\033[0m\n", config_->info_.c_str());
        }

        // 主要使用enqueue, 在算子入队后进行一些操。不再实现序列和和反序列化的操作, 当然也可以实现, 覆盖掉默认方法即可
        int enqueue(const std::vector<GTensor>& inputs, std::vector<GTensor>& outputs, const std::vector<GTensor>& weights, void* workspace, cudaStream_t stream) override{
            int n = inputs[0].count();
            const int nthreads = 512;
            int block_size = n < nthreads ? n : nthreads;
            int grid_size  = (n + block_size - 1) / block_size; 

            MYSELU_kernel_fp32 <<<grid_size, block_size, 0, stream>>>(inputs[0].ptr<float>(), outputs[0].ptr<float>(), n);
            return 0;
        }
};

// creator我们将不在创建, 在下面的功能中我们将会创建一个默认的通用的creator, 主要通过onnxplugin类来实现
RegisterPlugin(MYSELU);
