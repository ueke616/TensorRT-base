// tensorRT include
// 编译用的头文件
#include <NvInfer.h>

// onnx解析器的头文件
#include <onnx-tensorrt-release-8.6/NvOnnxParser.h>

// 推理用的运行时头文件
#include <NvInferRuntime.h>

// cuda include
#include <cuda_runtime.h>

// system include
#include <stdio.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <functional>
#include <unistd.h>

#include <opencv2/opencv.hpp>

using namespace std;

#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line){
    if(code != cudaSuccess){
        const char* err_name    = cudaGetErrorName(code);
        const char* err_message = cudaGetErrorString(code);
        printf("runtime error %s:%d %s failed. \n code = %s, message = %s \n", file, line, op, err_name, err_message);
        return false;
    }
    return true;
}

inline const char* severity_string(nvinfer1::ILogger::Severity t){
    switch(t){
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
        case nvinfer1::ILogger::Severity::kERROR:   return "error";
        case nvinfer1::ILogger::Severity::kWARNING: return "warning";
        case nvinfer1::ILogger::Severity::kINFO:    return "info";
        case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
        default: return "unknow";
    }
}

class TRTLogger : public nvinfer1::ILogger{
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override{
        if(severity <= Severity::kINFO){
            // 打印带颜色的字符，格式如下：
            // printf("\033[47;33m打印的文本\033[0m");
            // 其中 \033[ 是起始标记
            //      47    是背景颜色
            //      ;     分隔符
            //      33    文字颜色
            //      m     开始标记结束
            //      \033[0m 是终止标记
            // 其中背景颜色或者文字颜色可不写
            // 部分颜色代码 https://blog.csdn.net/ericbar/article/details/79652086
            if(severity == Severity::kWARNING){
                printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else if(severity <= Severity::kERROR){
                printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else{
                printf("%s: %s\n", severity_string(severity), msg);
            }
        }
    }
} logger;

// 定义一个函数指针类型Int8Process，用于处理图像预处理和加载数据到Tensor的操作。
// 这个函数接受当前处理的图像索引、总图像数、图像文件列表、输入维度和一个指向浮点数组的指针。
typedef std::function<void(
    int current, int count, const std::vector<std::string>& files,
    nvinfer1::Dims dims, float* ptensor
)> Int8Process;

// int8 熵校准器: 用于评估量化前后的分布改变
// Int8EntropyCalibrator类用于INT8量化过程中的熵校准，继承自nvinfer1::IInt8EntropyCalibrator2接口。
class Int8EntropyCalibrator: public nvinfer1::IInt8EntropyCalibrator2 {
    public:
        // 构造函数，用于根据给定的图像文件和预处理函数初始化校准器
        // imagefiles: 输入图像的文件路径列表
        // dims: 输入Tensor的维度
        // preprocess: 图像预处理函数，将图像数据加载到Tensor中
        Int8EntropyCalibrator(const vector<string>& imagefiles, nvinfer1::Dims dims, const Int8Process& preprocess){
            assert(preprocess != nullptr);
            this->dims_         = dims;
            this->allimgs_      = imagefiles;
            this->preprocess_   = preprocess;
            this->fromCalibratorData_ = false;
            files_.resize(dims.d[0]);
        }

        // 从校准缓存数据加载的构造函数，允许直接使用之前的校准结果，避免重复计算
        // entropyCalibratorData: 之前校准的缓存数据
        Int8EntropyCalibrator(const vector<uint8_t>& entropyCalibratorData, nvinfer1::Dims dims, const Int8Process& preprocess) {
            assert(preprocess != nullptr); // 确保传入的预处理函数不为空
            this->dims_ = dims; // 保存输入维度
            this->entropyCalibratorData_ = entropyCalibratorData; // 保存校准缓存数据
            this->preprocess_ = preprocess;  // 对输入进行操作的预处理函数
            this->fromCalibratorData_ = true; // 标记为从校准缓存数据创建的对象
            files_.resize(dims.d[0]); // 根据batch大小调整files_的大小
        }

        // 析构函数，负责释放分配的资源
        virtual ~Int8EntropyCalibrator(){
            if(tensor_host_ != nullptr){
                checkRuntime(cudaFreeHost(tensor_host_)); // 释放主机内存
                checkRuntime(cudaFree(tensor_device_)); // 释放设备内存
                tensor_host_ = nullptr;
                tensor_device_ = nullptr;
            }
        }

        // 返回校准过程中使用的batch大小
        int getBatchSize() const noexcept {
            return dims_.d[0];
        }

        // 准备下一批次的数据，返回false表示没有更多数据可用于校准
        bool next() {
            int batch_size = dims_.d[0];
            if (cursor_ + batch_size > allimgs_.size())
                return false; // 检查是否还有足够的图像用于下一个batch

            // 为当前batch准备图像文件路径
            for(int i=0; i < batch_size; ++i) {
                files_[i] = allimgs_[cursor_++];
            }

            // 如果是第一次调用，分配主机和设备内存
            if(tensor_host_ == nullptr) {
                size_t volume = 1;
                for (int i=0; i < dims_.nbDims; ++i)
                    volume *= dims_.d[i]; // 计算Tensor总元素数
                
                bytes_ = volume * sizeof(float); // 计算需要的内存大小
                checkRuntime(cudaMallocHost(&tensor_host_, bytes_)); // 分配主机内存
                checkRuntime(cudaMalloc(&tensor_device_, bytes_)); // 分配设备内存
            }

            // 使用预处理函数加载图像数据到主机内存Tensor
            preprocess_(cursor_, allimgs_.size(), files_, dims_, tensor_host_);
            // 将数据从主机内存拷贝到设备内存
            checkRuntime(cudaMemcpy(tensor_device_, tensor_host_, bytes_, cudaMemcpyHostToDevice));
            return true;
        }

        // 获取当前批次的数据，用于校准
        bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept {
            if(!next()) return false; // 准备下一批数据，如果没有则返回false
            bindings[0] = tensor_device_; // 设置绑定到当前设备内存上的Tensor
            return true;
        }
        
        // 获取之前保存的校准缓存数据
        const vector<uint8_t>& getEntropyCalibratorData() {
            return entropyCalibratorData_;
        }

        // 读取校准缓存，用于初始化校准过程
        const void* readCalibrationCache(size_t& length) noexcept {
            if (fromCalibratorData_) {
                length = this->entropyCalibratorData_.size();
                return this->entropyCalibratorData_.data();
            }
            length = 0;
            return nullptr;
        }
    
        // 写入校准缓存，保存校准结果以便将来使用
        virtual void writeCalibrationCache(const void* cache, size_t length) noexcept {
            entropyCalibratorData_.assign((uint8_t*)cache, (uint8_t*)cache + length);
        }

    private:
        Int8Process preprocess_;    // 预处理函数
        vector<string> allimgs_;    // 输入图像文件路径列表
        size_t batchCudaSize_ = 0;  // 未使用, 可考虑移除
        int cursor_ = 0;            // 当前处理到的图像索引
        size_t bytes_ = 0;          // 当前batch的数据大小
        nvinfer1::Dims dims_;       // 输入数据的维度信息
        vector<string> files_;      // 存储当前batch中需要处理的文件图像的路径
        float* tensor_host_ = nullptr;          // 存储在cpu内存中的tensor数据. 校准过程中, 图像数据会首先被读取并预处理为浮点数格式, 然后存储在这个数据中
        float* tensor_device_ = nullptr;        // 存储在gpu内存中的tensor数据. 校准过程中, 主机内存中的数据被拷贝到这个设备内存中, 一边TensorRT可以利用GPU进行加速计算
        vector<uint8_t> entropyCalibratorData_; // 用于存储校准过程生成的熵校准数据. 在校准完成后，这些数据可以被保存到文件中，以便在将来的校准过程中直接加载，从而避免重复进行耗时的校准计算
        bool fromCalibratorData_ = false;       // 用于指示Int8EntropyCalibrator对象是直接从之前保存的熵校准数据创建的（true），还是需要通过处理图像数据来生成校准数据的（false）。这个标志影响校准器的行为，特别是在读取校准缓存数据时。
};


// 通过智能指针管理 nv 返回的指针参数
// 内存自动释放, 避免泄漏
template<typename _T>
static shared_ptr<_T> make_nvshared(_T* ptr){
    return shared_ptr<_T>(ptr, [](_T* p){p->destroy();});
}


static bool exists(const string& path) {
    #ifdef _WIN32
        return ::PathFileExistA(path.c_str());
    #else
        return access(path.c_str(), R_OK) == 0;
    #endif
}

// 创建模型
bool build_model(){
    if(exists("engine.trtmodel")){
        printf("Engine.trtmodel has exists.\n");
        return true;
    }

    TRTLogger logger;

    // 使用智能指针对基本组件进行管理
    auto builder = make_nvshared(nvinfer1::createInferBuilder(logger));
    auto config  = make_nvshared(builder->createBuilderConfig());

    // createNetworkV2(1)表示采用显性batch size，新版tensorRT(>=7.0)时，不建议采用0非显性batch size
    // 因此贯穿以后，请都采用createNetworkV2(1)而非createNetworkV2(0)或者createNetwork
    auto network = make_nvshared(builder->createNetworkV2(1));

    // 通过onnxparser解析器解析的结果会填充到network中，类似addConv的方式添加进去
    auto parser = make_nvshared(nvonnxparser::createParser(*network, logger));
    if(!parser->parseFromFile("classifier.onnx", 1)) {
        printf("Failed to parse classifier.onnx \n");

        return false;
    }

    int maxBatchSize = 10;
    printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f);
    config->setMaxWorkspaceSize(1 << 28);

    // 如果模型有多个执行上下文，则必须多个profile
    // 多个输入共用一个profile
    auto profile = builder->createOptimizationProfile();
    auto input_tensor = network->getInput(0);
    auto input_dims = input_tensor->getDimensions();

    // 设置了模型的批量大小为1，以及启用了INT8量化，为后续的模型优化和推理准备
    input_dims.d[0] = 1;
    config->setFlag(nvinfer1::BuilderFlag::kINT8);

    // 将图像数据标准化并调整到模型期望的输入尺寸和格式
    // Lambda表达式接受5个参数：当前处理的图像索引（current）、总图像数（count）、图像文件路径列表（files）、输入维度（dims）、指向预处理后的浮点数据的指针（ptensor）
    auto preprocess = [](
        int current, int count, const std::vector<std::string>& files, 
        nvinfer1::Dims dims, float* ptensor
    ){
        printf("Preprocess %d / %d\n", count, current);

        // 标定所采用的数据预处理必须与推理时一样
        // 解析输入维度
        int width = dims.d[3];
        int height = dims.d[2];
        // 定义均值和标准差
        float mean[] = {0.406, 0.456, 0.485};
        float std[]  = {0.225, 0.224, 0.229};

        // opencv读取图像, 缩放到期望尺寸, 将图像像素格式从unsigned char转换到标准化后的float格式, 并以BGR的格式存储在ptensor指向的内存中
        for(int i = 0; i < files.size(); ++i){
            auto image = cv::imread(files[i]);
            cv::resize(image, image, cv::Size(width, height));
            int image_area = width * height;
            unsigned char* pimage = image.data;
            float* phost_b = ptensor + image_area * 0;
            float* phost_g = ptensor + image_area * 1;
            float* phost_r = ptensor + image_area * 2;
            for(int i = 0; i < image_area; ++i, pimage += 3){
                // 注意这里的顺序rgb调换了
                *phost_r++ = (pimage[0] / 255.0f - mean[0]) / std[0];
                *phost_g++ = (pimage[1] / 255.0f - mean[1]) / std[1];
                *phost_b++ = (pimage[2] / 255.0f - mean[2]) / std[2];
            }
            ptensor += image_area * 3;
        }
    };

    /*
    在TensorRT的INT8量化流程中，Int8EntropyCalibrator对象用于处理校准数据集，生成校准表。这个校准表随后用于量化模型的权重和激活，从而在不显著牺牲精度的情况下提高模型的推理速度。使用校准表进行INT8量化是一种平衡推理性能和模型精度的有效方法。
    */
    // 配置INT8校准过程中的数据读取和预处理工具
    shared_ptr<Int8EntropyCalibrator> calib(new Int8EntropyCalibrator(
        {"dou.png"}, input_dims, preprocess
    ));
    config->setInt8Calibrator(calib.get());  // 添加量化器到配置中

    // 有关输入图像的设置
    // 配置最小允许batch
    input_dims.d[0] = 1;
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);

    // 配置最大允许batch
    // if networkDims.d[i] != -1, then minDims.d[i] == optDims.d[i] == maxDims.d[i] == networkDims.d[i]
    input_dims.d[0] = maxBatchSize;
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
    config->addOptimizationProfile(profile);

    auto engine = make_nvshared(builder->buildEngineWithConfig(*network, *config));
    if(engine == nullptr) {
        printf("builder engine failed \n");
        return false;
    }

    // 将模型序列化，并储存为文件
    auto model_data = make_nvshared(engine->serialize());
    FILE* f = fopen("engine.trtmodel", "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    f = fopen("calib.txt", "wb");
    auto calib_data = calib->getEntropyCalibratorData();
    fwrite(calib_data.data(), 1, calib_data.size(), f);
    fclose(f);

    // 卸载顺序按照构建顺序倒序
    printf("Done.\n");
    return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////

vector<unsigned char> load_file(const string& file){
    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0){
        in.seekg(0, ios::beg);
        data.resize(length);

        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
}

vector<string> load_labels(const char* file) {
    vector<string> lines;

    ifstream in(file, ios::in | ios::binary);
    if(!in.is_open()){
        printf("open %d failed. \n", file);
        return lines;
    }

    string line;
    while(getline(in, line)){
        lines.push_back(line);
    }
    in.close();
    return lines;
}

void inference() {
    TRTLogger logger;
    // 反序列化推理引擎
    auto engine_data = load_file("engine.trtmodel");
    auto runtime   = make_nvshared(nvinfer1::createInferRuntime(logger));
    auto engine = make_nvshared(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if(engine == nullptr){
        printf("Deserialize cuda engine failed.\n");
        runtime->destroy();
        return;
    }

    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));
    auto execution_context = make_nvshared(engine->createExecutionContext());

    int input_batch   = 1;
    int input_channel = 3;
    int input_height  = 224;
    int input_width   = 224;
    int input_numel   = input_batch * input_channel * input_height * input_width;
    float* input_data_host   = nullptr;
    float* input_data_device = nullptr;
    checkRuntime(cudaMallocHost(&input_data_host, input_numel * sizeof(float)));
    checkRuntime(cudaMalloc(&input_data_device, input_numel * sizeof(float)));

    ///////////////////////////////////////////////////
    // image to float
    auto image = cv::imread("dou.png");
    float mean[] = {0.406, 0.456, 0.485};
    float std[]  = {0.225, 0.224, 0.229};

    // 对应于pytorch的代码部分
    cv::resize(image, image, cv::Size(input_width, input_height));
    int image_area = image.cols * image.rows;
    unsigned char* pimage = image.data;
    float* phost_b = input_data_host + image_area * 0;
    float* phost_g = input_data_host + image_area * 1;
    float* phost_r = input_data_host + image_area * 2;
    for(int i = 0; i < image_area; ++i, pimage += 3){
        // 注意这里的顺序rgb调换了
        *phost_r++ = (pimage[0] / 255.0f - mean[0]) / std[0];
        *phost_g++ = (pimage[1] / 255.0f - mean[1]) / std[1];
        *phost_b++ = (pimage[2] / 255.0f - mean[2]) / std[2];
    }
    ///////////////////////////////////////////////////
    checkRuntime(cudaMemcpyAsync(input_data_device, input_data_host, input_numel * sizeof(float), cudaMemcpyHostToDevice, stream));

    // 3x3输入，对应3x3输出
    const int num_classes = 1000;
    float output_data_host[num_classes];
    float* output_data_device = nullptr;
    checkRuntime(cudaMalloc(&output_data_device, sizeof(output_data_host)));

    // 明确当前推理时，使用的数据输入大小
    auto input_dims = execution_context->getBindingDimensions(0);
    input_dims.d[0] = input_batch;

    execution_context->setBindingDimensions(0, input_dims);
    float* bindings[] = {input_data_device, output_data_device};
    bool success      = execution_context->enqueueV2((void**)bindings, stream, nullptr);
    checkRuntime(cudaMemcpyAsync(output_data_host, output_data_device, sizeof(output_data_host), cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));

    float* prob = output_data_host;
    int predict_label = std::max_element(prob, prob + num_classes) - prob;
    auto labels = load_labels("labels.imagenet.txt");
    auto predict_name = labels[predict_label];
    float confidence  = prob[predict_label];
    printf("Predict: %s, confidence = %f, label = %d\n", predict_name.c_str(), confidence, predict_label);

    checkRuntime(cudaStreamDestroy(stream));
    checkRuntime(cudaFreeHost(input_data_host));
    checkRuntime(cudaFree(input_data_device));
    checkRuntime(cudaFree(output_data_device));
}

int main(){
    if(!build_model()){
        return -1;
    }
    inference();
    return 0;
}
