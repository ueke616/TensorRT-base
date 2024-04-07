// tensorRT include
#include <NvInfer.h>
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
// 上一节的代码

class TRTLogger: public nvinfer1::ILogger {
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override{
        if(severity <= Severity::kINFO){
            printf("%d: %s \n", severity, msg);
        }
    }
} logger;


// 通过提供一个浮点数数组的指针和数组中元素的数量，它能够创建一个nvinfer1::Weights对象，这个对象后续可以用于TensorRT网络的构建过程中，特别是在定义网络层时指定权重。
nvinfer1::Weights make_weights(float* ptr, int n){
    nvinfer1::Weights w;
    w.count = n;
    w.type = nvinfer1::DataType::kFLOAT;
    w.values = ptr;
    return w;
}


// 定义一个函数用于构建模型，成功构建返回 true，否则返回 false
bool build_model(){
    // 创建一个 TensorRT 日志对象，用于记录构建过程中的信息或错误
    TRTLogger logger;

    // 创建一个模型构建器，这是构建 TensorRT 模型的起点
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    // 创建一个构建配置对象，用于指定模型优化和运行时的配置
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    // 创建一个网络定义，这是模型的结构定义
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1);

    // 定义输入和输出的数量
    const int num_input  = 3;
    const int num_output = 2;
    // 定义第一层权重和偏置的值
    float layer1_weight_values[] = {1.0, 2.0, 0.5, 0.1, 0.2, 0.5};
    float layer1_bias_values[] = {0.3, 0.8};

    // 添加一个输入层到网络中，指定数据类型和维度
    nvinfer1::ITensor* input = network->addInput("image", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4(1, num_input, 1, 1));
    // 使用辅助函数将权重值转换为 TensorRT 可接受的格式
    // 全连接层的权重数量取决于输入节点数（num_input）和输出节点数（num_output）的乘积 3 * 2 = 6
    // 全连接层的偏置数量等于输出节点的数量 2
    nvinfer1::Weights layer1_weight  = make_weights(layer1_weight_values, 6);
    nvinfer1::Weights layer1_bias    = make_weights(layer1_bias_values, 2);
    // 添加一个全连接层到网络中
    auto layer1 = network->addFullyConnected(*input, num_output, layer1_weight, layer1_bias);
    // 添加一个激活层到网络中，使用 Sigmoid 激活函数
    auto prob   = network->addActivation(*layer1->getOutput(0), nvinfer1::ActivationType::kSIGMOID);

    // 标记网络的输出
    network->markOutput(*prob->getOutput(0));

    // 设置工作空间大小，这是 GPU 上用于中间数据存储的空间
    printf("Workspace Size = %.2f MB \n", (1 << 28) / 1024.0f / 1024.0f);
    config->setMaxWorkspaceSize(1 << 28);
    // 设置最大批量大小
    builder->setMaxBatchSize(1);

    // 根据网络定义和配置构建一个 CUDA 引擎
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    if(engine == nullptr){
        printf("Build engine failed. \n");
        return false;
    }

    // 序列化模型为一个流式格式，可以保存到磁盘上
    nvinfer1::IHostMemory* model_data = engine->serialize();
    // 将序列化的模型数据写入到文件中
    FILE* f = fopen("engine.trtmodel", "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    // 释放资源
    model_data->destroy();
    engine->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
    printf("Done. \n");
    return true;
}



// 定义一个函数，接受一个文件路径作为参数，返回一个包含文件内容的 unsigned char 类型的 vector
vector<unsigned char> load_file(const string& file) {
    // 使用 ifstream 打开文件，以输入模式和二进制模式
    ifstream in(file, ios::in | ios::binary);
    
    // 检查文件是否成功打开
    if(!in.is_open())
        // 如果文件未成功打开，返回一个空的 vector
        return {};
    
    // 将文件指针移动到文件末尾
    in.seekg(0, ios::end);
    // 获取当前文件指针的位置，即文件的总长度
    size_t length = in.tellg();

    // 定义一个用于存储文件数据的 vector
    std::vector<uint8_t> data;
    
    // 检查文件长度是否大于0
    if (length > 0){
        // 将文件指针重新移动到文件开头
        in.seekg(0, ios::beg);
        // 根据文件长度重新设置 vector 的大小
        data.resize(length);

        // 从文件中读取数据，存储到 vector 中
        // 使用 read 方法从文件中读取 length 个字节的数据到 data 容器中。这里将 data 的第一个元素的地址作为目标缓冲区的起始地址（需要将其从 uint8_t* 类型转换为 read 方法期望的 char* 类型
        in.read((char*)&data[0], length);
    }
    
    // 关闭文件
    in.close();
    
    // 返回包含文件数据的 vector
    return data;
}



void inference(){
    // 1. 准备模型并加载
    TRTLogger logger;
    auto engine_data = load_file("engine.trtmodel");
    /*
    创建一个用于执行模型推理的运行时环境。
    - **`createInferRuntime`**：这是一个函数，用于创建一个新的推理运行时环境。这个环境是必须的，以便在 NVIDIA GPU 上加载和执行经过优化的深度学习模型。
    参数列表
    - **`logger`**：这是一个日志记录器，用于记录运行时的信息、警告或错误。它帮助开发者了解运行时发生了什么，特别是在出现问题时用于调试。
    功能和目的
    通过执行这行代码，你会得到一个 `IRuntime` 实例，它允许你加载之前保存的、优化后的模型，并在 NVIDIA GPU 上执行这些模型进行推理。这是使用 TensorRT 进行高效深度学习推理的第一步。
    */
    nvinfer1::IRuntime* runtime     = nvinfer1::createInferRuntime(logger);
    /*
    把之前优化并保存下来的模型“复活”到内存中，准备在 GPU 上运行推理。
    - **`deserializeCudaEngine`**：这个方法的作用是将之前序列化（也就是转换成一串字节数据）的模型数据反序列化，创建出一个可用于执行推理的 CUDA 引擎。
    参数列表
    - **`engine_data.data()`**：指向包含序列化引擎数据的缓冲区的指针。`engine_data` 是一个包含序列化模型数据的容器（如 `std::vector`），`.data()` 方法返回这个容器内部数据的原始指针。
    - **`engine_data.size()`**：序列化数据的大小，即缓冲区中的字节数。这告诉 `deserializeCudaEngine` 方法有多少字节的数据需要被处理来恢复 CUDA 引擎。
    */
    nvinfer1::ICudaEngine* engine   = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    if(engine == nullptr){  // 反序列化模型失败
        printf("Deserialize cuda engine failed. \n");
        runtime->destroy();
        return;
    }
    /*
    创建一个执行上下文（`IExecutionContext`），用于在之前创建的推理引擎（`engine`）上运行模型推理。
    ### 方法解释
    - **`createExecutionContext`**：这是一个方法，用于从推理引擎（`engine`）创建一个执行上下文。执行上下文包含了执行推理所需的所有状态信息，使得模型可以在 GPU 上运行。
    ### 参数列表
    - 这个方法没有显式参数列表，因为它是直接调用在一个引擎实例上的，所以它使用的是该引擎的内部状态。
    ### 功能和目的
    执行上下文是执行模型推理的实际环境，它使用了之前通过引擎定义的模型结构和优化。你可以通过这个上下文来实际运行模型，进行数据的输入和获取预测结果。
    */
    nvinfer1::IExecutionContext* execution_context = engine->createExecutionContext();
    cudaStream_t stream = nullptr;
    // 创建CUDA流, 以确定这个batch的推理是独立的
    // CUDA 流是一个序列化的操作队列，用于管理在 NVIDIA GPU 上执行的一系列计算和内存操作。
    // 通过这种方式，它可以异步地执行多个操作，提高 GPU 的利用率和整体性能。
    cudaStreamCreate(&stream);

    // 2. 准备好要推理的数据并搬运到GPU
    float input_data_host[] = {1, 2, 3};
    float* input_data_device = nullptr;

    float output_data_host[2];
    float* output_data_device = nullptr;
    /*
    在 GPU 上分配足够的内存空间，以便存放与 `output_data_host` 大小相同的数据。
    - **`cudaMalloc`**：这是一个 CUDA 运行时 API 函数，会在 GPU 的全局内存中分配指定大小的空间。
    - **`&input_data_device`**：这是一个指向指针的指针，`cudaMalloc` 将修改它所指向的指针，让它指向新分配的 GPU 内存地址。
    - **`sizeof(output_data_host)`**：这表示要分配的内存大小，等于 `output_data_host` 数据结构所占用的字节数。
    */
    cudaMalloc(&input_data_device, sizeof(input_data_host));
    cudaMalloc(&output_data_device, sizeof(output_data_host));
    cudaMemcpyAsync(input_data_device, input_data_host, sizeof(input_data_host), cudaMemcpyHostToDevice, stream);

    // 两个指针分别指向在 GPU 内存中分配的输入数据和输出数据的位置
    // 用于将输入和输出数据的内存位置传递给执行上下文，以便模型知道从哪里读取输入数据，以及将推理结果写入到哪个位置
    float* bindings[] = {input_data_device, output_data_device};

    // 3. 推理并将结果搬运回CPU
    /*
    通过 TensorRT 的执行上下文 (`execution_context`) 
    将一个推理任务异步地排入一个 CUDA 流中执行，并立即返回一个标志，表明任务是否成功启动。
    - **`enqueueV2`**：这个方法用于将推理任务排入指定的 CUDA 流。它是异步执行的，意味着它会立即返回，而推理任务会在后台的 CUDA 流上执行。
    参数列表:
    - **`(void**)bindings`**：这是将 `bindings` 数组的地址转换为 `void**` 类型。`bindings` 数组包含了输入和输出数据在 GPU 内存中的地址。转换为 `void**` 是因为 `enqueueV2` 方法需要一个指向 `void` 指针的指针，以通用方式处理不同数据类型的内存地址。
    - **`stream`**：这是 CUDA 流的标识符，指定了推理任务应该在哪个 CUDA 流上执行。使用 CUDA 流可以实现并行执行多个任务，提高 GPU 的利用率。
    - **`nullptr`**：这个参数是用于事件回调的，由于在这个调用中不需要事件回调，所以传递了 `nullptr`。
    返回值:
    - **`bool success`**：`enqueueV2` 方法的返回值是一个布尔值，表示推理任务是否成功排入了 CUDA 流。如果成功，它会返回 `true`；如果有错误发生，比如内存不足或者参数不正确，它会返回 `false`。
    */
    bool success    = execution_context->enqueueV2((void**)bindings, stream, nullptr);
    /*
    使用 CUDA 异步拷贝功能，将数据从 GPU 内存（设备内存）复制回到 CPU 内存（主机内存）。
    - **`output_data_host`**：目标位置的指针，指向 CPU 内存中准备接收数据的位置。
    - **`output_data_device`**：源位置的指针，指向 GPU 内存中存储有数据的位置。
    - **`sizeof(output_data_host)`**：要复制的数据大小。这里使用 `sizeof` 来获取 `output_data_host` 指向的内存区域的大小，但注意，这种做法仅在 `output_data_host` 是数组或已知大小的对象时才正确。如果 `output_data_host` 是一个指针，这里应该使用实际的数据大小，而不是使用 `sizeof(output_data_host)`，因为这仅仅会返回指针的大小。
    - **`cudaMemcpyDeviceToHost`**：复制方向，从设备（GPU）到主机（CPU）。
    - **`stream`**：指定操作应该在哪个 CUDA 流上执行。这允许操作异步执行，即在不阻塞当前线程的情况下进行。
    这个函数调用是异步的，意味着它会立即返回，而数据复制操作可能在后台进行。复制完成后，`output_data_host` 将包含从 GPU 内存复制过来的数据。这通常在执行了 GPU 上的计算任务之后进行，以获取计算结果。
    */
    cudaMemcpyAsync(output_data_host, output_data_device, sizeof(output_data_host), cudaMemcpyDeviceToHost, stream);
    /*
    用于等待与指定 CUDA 流相关联的所有先前排队的操作完成。它是一个同步操作，意味着调用此函数的线程会阻塞，直到指定的 CUDA 流中的所有操作（如内存复制、核函数执行等）都完成执行。
    - **`stream`**：这是你想要同步的 CUDA 流的标识符。它指定了之前通过 `cudaStreamCreate` 创建并用于排队操作的流。
    简单来说，`cudaStreamSynchronize` 确保在程序中继续执行任何进一步操作之前，所有已经提交到 CUDA 流的任务都已经完成了。这对于确保数据完整性和正确的执行顺序非常重要，特别是在需要使用到流中操作结果的情况下。
    */
    cudaStreamSynchronize(stream);
    printf("output_data_host = %f, %f\n", output_data_host[0], output_data_host[1]);

    // 4. 释放内存
    printf("clean memory \n");
    cudaStreamDestroy(stream);  // 销毁流
    execution_context->destroy();
    engine->destroy();
    runtime->destroy();

    // 5. 手动推理进行验证
    const int num_input = 3;  // 输入
    const int num_output = 2;   // 输出
    float layer1_weight_values[] = {1.0, 2.0, 0.5, 0.1, 0.2, 0.5};  // 全连接层的权重和偏置
    float layer1_bias_values[] = {0.3, 0.8};

    printf("手动验证计算结果：\n");
    // 执行前向传播计算
    for(int io=0; io < num_output; ++io){
        float output_host = layer1_bias_values[io];
        // 内部循环通过每个输入（num_input），将输入值乘以对应的权重，并累加到当前输出值上
        for(int ii = 0; ii < num_input; ++ii) {
            output_host += layer1_weight_values[io * num_input + ii] * input_data_host[ii];
        }

        // sigmoid
        // 计算完成加权和后，通过 Sigmoid 函数转换输出值, 得到[0, 1]的概率值
        float prob = 1 / (1 + exp(-output_host));
        printf("output_prob[%d] = %f\n", io, prob);
    }
}

int main() {
    if(!build_model()){
        return -1;
    }
    inference();
    return 0;
}
