// 编译用的头文件
#include <NvInfer.h>

// onnx解析器的头文件
#include <onnx-tensorrt-release-8.6/NvOnnxParser.h>

// 推理用的运行时头文件
#include <NvInferRuntime.h>

// cuda include
#include <cuda_runtime.h>

// system include
#include <memory>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

inline const char* severity_string(nvinfer1::ILogger::Severity t){
    switch(t) {
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
        case nvinfer1::ILogger::Severity::kERROR:   return "error";
        case nvinfer1::ILogger::Severity::kWARNING: return "warning";
        case nvinfer1::ILogger::Severity::kINFO:    return "info";
        case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
        default: return "unknow";
    }
}

class TRTLogger: public nvinfer1::ILogger {
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

//上一节的代码
bool build_model(){
    TRTLogger logger;

    /* 在内存堆区创建的变量, 返回时未销毁, 可能导致内存泄漏
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1);
    */
    // 这是基本需要的组件， 修改成智能指针来避免内存泄漏
    std::shared_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger), [](nvinfer1::IBuilder* ptr) { ptr->destroy(); });
    std::shared_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig(), [](nvinfer1::IBuilderConfig* ptr) { ptr->destroy(); });
    std::shared_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(1), [](nvinfer1::INetworkDefinition* ptr) { ptr->destroy(); });

    /*
    使用TensorRT的ONNX解析器创建一个用于解析ONNX模型的解析器对象。它是将ONNX模型导入到TensorRT中的第一步，以便之后可以将该模型编译成优化的推理引擎
    nvonnxparser::createParser: 这是一个函数，用于创建并返回一个指向新创建的IParser接口的指针。这个函数是nvonnxparser命名空间中定义的。
    参数列表:
        *network: 这是createParser函数的第一个参数，它是一个对nvinfer1::INetworkDefinition对象的引用。这个对象代表了TensorRT网络定义，解析器将会把ONNX模型中的网络结构和参数加载到这个网络定义中。
        logger: 这是createParser函数的第二个参数，它是一个nvinfer1::ILogger对象，用于记录解析过程中的日志信息。这允许用户获取解析过程中的警告、错误等日志信息，帮助调试。
    简单来说，这行代码通过指定的网络定义和日志记录器创建了一个ONNX解析器对象，以便加载和解析ONNX模型
    */
    // nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    std::shared_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network.get(), logger), [](nvonnxparser::IParser* ptr) { ptr->destroy(); });
    if (!parser->parseFromFile("demo.onnx", 1)) {
        printf("Failed to parse demo.onnx");

        // 注意这里的几个指针没有释放, 是有内存泄露的，后面考虑更优雅的解决
        /*
        存在内存泄漏的原因主要在于创建的对象（如IBuilder, IBuilderConfig, INetworkDefinition, 和 IParser）
        都是通过NVIDIA TensorRT提供的工厂方法分配在堆上的，但在函数返回前，并没有对它们进行适当的释放处理

        为什么存在内存泄漏？
        在C++中，当你使用new操作符或者特定的工厂方法（如TensorRT中的createInferBuilder）创建动态对象时，这些对象会被分配在堆上。
        与自动（栈）变量不同，堆上的对象不会在离开其作用域时自动销毁。因此，你需要显式地调用delete（对于单个对象）或delete[]（对于对象数组）来释放这些对象占用的内存。
        如果忘记释放这些动态分配的对象，程序就会出现内存泄漏——即内存得不到释放，仍然被分配状态，直到程序终止

        解决方法
        为了解决内存泄漏问题，你需要确保在对象不再需要时释放它们。
        一个常见的做法是使用智能指针，如std::unique_ptr或std::shared_ptr，这些智能指针可以自动管理对象的生命周期，从而避免内存泄漏。
        
        当智能指针超出其作用域时，它的析构函数会自动调用用户提供的销毁函数来释放资源。这样，即使在遇到错误返回时，智能指针也会确保资源被正确释放，从而避免内存泄漏。
        */
        return false;
    }

    int maxBatchSize = 10;
    printf("Workspace size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f);
    config->setMaxWorkspaceSize(1 << 28);

    // 如果模型有多个输入, 则必须多个profile
    auto profile = builder->createOptimizationProfile();
    auto input_tensor = network->getInput(0);
    int input_channel = input_tensor->getDimensions().d[1];

    // 配置输入的最小、最优、最大的范围
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, input_channel, 3, 3));
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, input_channel, 3, 3));
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(maxBatchSize, input_channel, 5, 5));
    config->addOptimizationProfile(profile);

    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    if(engine == nullptr) {
        printf("Build engine failed. \n");
        return false;
    }

    // 将模型序列化, 并存储为文件
    nvinfer1::IHostMemory* model_data = engine->serialize();
    FILE* f = fopen("engine.trtmodel", "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    // 卸载顺序按照构建顺序倒序
    model_data->destroy();
    // parser->destroy();
    engine->destroy();
    // network->destroy();
    // config->destroy();
    // builder->destroy();
    printf("Done.\n");
    return true; 
}

int main(){
    build_model();
    return 0;
}

