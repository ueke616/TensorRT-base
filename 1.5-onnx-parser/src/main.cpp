
// tensorRT include
// 编译用的头文件
#include <NvInfer.h>
// onnx解析器的头文件
#include <NvOnnxParser.h>
// 推理用的运行时头文件
#include <NvInferRuntime.h>
// cuda include
#include <cuda_runtime.h>
// system include
#include <cstdio>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

// TensorRT提供了一个ILogger接口，允许开发者自定义如何处理库产生的日志消息
// 定义一个内联函数，将TensorRT日志消息的严重性级别转换为字符串。
inline const char* severity_string(nvinfer1::ILogger::Severity t){
    // 使用switch语句来检查传入的严重性级别t。
    switch (t) {
        // 如果严重性级别是内部错误，返回对应的字符串。
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
        // 如果严重性级别是错误，返回对应的字符串。
        case nvinfer1::ILogger::Severity::kERROR: return "error";
        // 如果严重性级别是警告，返回对应的字符串。
        case nvinfer1::ILogger::Severity::kWARNING: return "warning";
        // 如果严重性级别是信息，返回对应的字符串。
        case nvinfer1::ILogger::Severity::kINFO: return "info";
        // 如果严重性级别是详细信息（冗长输出），返回对应的字符串。
        case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
        
        // 默认情况下（如果传入了一个未知的严重性级别），返回"unknown"。
        default: return "unknow";
    }
}



class TRTLogger: public nvinfer1::ILogger {
    public:
        virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override {
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
                } else if (severity <= Severity::kERROR) {
                    printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
                } else {
                    printf("%s: %s\n", severity_string(severity), msg);
                }
            }
        }
} logger;

// 上一节的代码
bool build_model(){
    TRTLogger logger;

    // 1. 定义 builder, config 和 network
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1);

    // 2. 输入, 模型结构和输出的基本信息
    // 通过 onnxparser 解析的结果会填充到 network 中, 类似 addConv 的方式添加进去
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    if(!parser->parseFromFile("demo.onnx", 1)){
        printf("Failed to parser demo.onnx\n");

        // 注意这里的几个指针还没有释放，是有内存泄漏的，后面考虑更优雅的解决
        return false;
    }

    int maxBatchSize = 10;
    printf("Workspace Size = %.2f MB \n", (1 << 28) / 1024.f / 1024.f);
    config->setMaxWorkspaceSize(1 << 28);

    // 2.1 关于 profile
    // 如果模型有多个输入, 则必须多个profile
    auto profile = builder->createOptimizationProfile();
    auto input_tensor = network->getInput(0);
    int input_channel = input_tensor->getDimensions().d[1];

    // 配置输入的最小、最优、最大的范围
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, input_channel, 3, 3));
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, input_channel, 3, 3));
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(maxBatchSize, input_channel, 5, 5));

    // 添加到配置
    config->addOptimizationProfile(profile);

    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    if(engine == nullptr){
        printf("Build engine failed. \n");
        return false;
    }
    
    // 3. 序列化
    // 将模型序列化, 并存储为文件
    nvinfer1::IHostMemory* model_data = engine->serialize();
    FILE* f = fopen("engine.trtmodel", "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    // 卸载顺序按照构建顺序倒序
    model_data->destroy();
    parser->destroy();
    engine->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
    printf("Done. \n");
    return true;
}

int main(){
    build_model();
    return 0;
}


