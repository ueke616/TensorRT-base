// tensorRT include
#include <NvInfer.h>
#include <NvInferRuntime.h>

// cuda include
#include <cuda_runtime.h>

// system include
#include <cstdio>

// 日志类，当TensorRT编译过程中出现了任何的消息，想让你知道它做了什么事情，可以用于问题的排查和调试
// TRTLogger类继承自nvinfer1::ILogger，nvinfer1是TensorRT提供的命名空间，ILogger是一个接口用于实现日志记录功能。
class TRTLogger: public nvinfer1::ILogger{

// noexcept：这个关键字用于指定函数不会抛出异常		它告诉编译器和程序的使用者，你可以安全地假定这个函数不会因为异常而退出
// override：这个关键字用于明确表示某个成员函数覆盖了基类中的虚函数		使用 override 关键字可以让编译器帮助你检查函数签名。如果你声明的函数并没有实际覆盖任何基类中的虚函数，编译器将报错
public:
    // 重写ILogger接口中的log方法。这是一个虚函数，用于处理日志消息。
    // TensorRT在运行时会调用这个方法来记录各种日志消息。
    virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override{
        // 通过日志的级别决定打印哪些内容
        // 检查日志消息的严重性等级。TensorRT定义了不同的日志严重性等级，
        // 包括kINTERNAL_ERROR, kERROR, kWARNING, kINFO, 和kVERBOSE。
        // 这里的if条件检查日志消息的严重性是否小于或等于kVERBOSE，
        // 即是否是冗余信息或更重要的消息。
        if(severity <= Severity::kVERBOSE){
            // 如果条件为真，使用printf打印日志消息。
            // %d: 将严重性等级作为整数打印。
            // %s: 打印指向日志消息字符串的指针。
            // 注意：日志的严重性等级也是以整数形式提供，它们映射到TensorRT定义的枚举值。
            printf("%d: %s\n", severity, msg);
        }
    }
};

// 辅助函数，用于简化TensorRT中权重对象的创建过程。
// 通过提供一个浮点数数组的指针和数组中元素的数量，它能够创建一个nvinfer1::Weights对象，这个对象后续可以用于TensorRT网络的构建过程中，特别是在定义网络层时指定权重。
// 定义一个函数 make_weight，它接收一个浮点数指针和一个整数，返回一个 nvinfer1::Weights 对象。
nvinfer1::Weights make_weights(float* ptr, int n){
    // 创建一个 nvinfer1::Weights 对象 w。
    nvinfer1::Weights w;
    // 设置 w 的 count 属性，这表示权重数组中元素的数量。
    w.count = n;
    // 设置 w 的 type 属性为 kFLOAT，表示这些权重是浮点数类型。
    w.type  = nvinfer1::DataType::kFLOAT;
    // 将传入的浮点数数组指针赋值给 w 的 values 属性，这个指针指向的内存包含了权重的实际数据。
    w.values = ptr;
    // 返回构造好的 Weights 对象。
    return w;
}

int main() {
    // 本代码主要实现一个最简单的神经网络 figure/simple_fully_connected_net.png 
    TRTLogger logger;   // logger是必要的, 用来捕捉warning和info等
    
    // ----------------------------- 1. 定义 builder, config 和network -----------------------------
    // 这是基本需要的组件
    // 形象的理解是你需要一个builder去build这个网络，网络自身有结构，这个结构可以有不同的配置
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    // 创建一个构建配置，指定TensorRT应该如何优化模型，tensorRT生成的模型只能在特定配置下运行
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    // 创建网络定义，其中createNetworkV2(1)表示采用显性batch size，新版tensorRT(>=7.0)时，不建议采用0非显性batch size
    // 因此贯穿以后，请都采用createNetworkV2(1)而非createNetworkV2(0)或者createNetwork
    // 0 或不传递任何参数通常表示创建一个标准网络，没有额外的属性。
    // 1 << 0 （等同于 1）通常表示启用 EXPLICIT_BATCH，即显式定义批量大小。
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1);

    // 构建一个模型
    /*
        Network definition:

        image
          |
        linear (fully connected)  input = 3, output = 2, bias = True     w=[[1.0, 2.0, 0.5], [0.1, 0.2, 0.5]], b=[0.3, 0.8]
          |
        sigmoid
          |
        prob
    */

    // ----------------------------- 2. 输入，模型结构和输出的基本信息 -----------------------------
    const int num_input = 3;    // in_channel
    const int num_output = 2;   // out_channel
    float layer1_weight_values[] = {1.0, 2.0, 0.5, 0.1, 0.2, 0.5};  // 前3个给w1的rgb, 后3个给w2的rgb
    float layer1_bias_values[] = {0.3, 0.8};

    // 输入指定数据的名称、数据类型和完整维度，将输入层添加到网络
    /*
    在 TensorRT 中，`addInput` 方法用于向网络定义中添加一个输入张量。这个方法的参数决定了输入张量的名字、数据类型和维度。
    这里是 `addInput` 方法的具体参数解释：
    - `"image"`：这是输入张量的名称。在后续构建和推理过程中，你可以通过这个名称引用这个输入。
    - `nvinfer1::DataType::kFLOAT`：这指定了输入数据的类型。在这个例子中，`kFLOAT` 表示使用 32位浮点数。TensorRT 支持不同的数据类型，比如半精度浮点数（kHALF）等。
    - `nvinfer1::Dims4(-1, num_input, -1, -1)`：这是输入张量的维度，`Dims4` 表示一个4维的维度对象。这些维度对应于 NCHW 格式，其中 N 是批量大小，C 是通道数，H 是高度，W 是宽度。
    - `-1`：在 N 和 H、W 的位置使用 `-1` 表示这些维度在网络创建时是动态的，即它们可以在推理时指定。这是显式批处理模式的一个特征，允许在不同的推理调用中使用不同的批量大小和输入尺寸。
    - `num_input`：这里指的是输入张量的通道数。在一些上下文中，比如处理彩色图像时，这个数字可能是3（对应于RGB通道）。
    当你调用 `addInput` 并使用动态尺寸（即 `-1`），你告诉 TensorRT，当实际推理时，你将提供具体的批量大小、高度和宽度。这使得模型能够更加灵活地处理不同大小的输入。
    在构建实际的推理引擎之后，你需要在推理时为这些动态维度指定具体的值，这通常是通过调用 `setBindingDimensions` 来完成的。
    */
    nvinfer1::ITensor* input = network->addInput("image", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4(1, num_input, 1, 1));
    nvinfer1::Weights layer1_weight = make_weights(layer1_weight_values, 6);
    nvinfer1::Weights layer1_bias   = make_weights(layer1_bias_values, 2);
    // 添加全连接层          注意对input及逆行了解引用
    auto layer1 = network->addFullyConnected(*input, num_output, layer1_weight, layer1_bias);
    // 添加激活层       注意更严谨的写法是*(layer1->getOutput(0)) 即对getOutput返回的指针进行解引用
    /*
    在 TensorRT 的 API 中，`getOutput(0)` 方法是用于从一个网络层中获取输出张量的。这个方法接受一个整数参数，表示你想要获取的输出张量的索引。
    在大多数情况下，网络层只会有一个输出（比如卷积层、池化层等），在这种情况下，索引 `0` 就是用来获取那个唯一的输出。但是，对于某些特殊的层，比如分支（branching）层或者某些自定义层，可能会有多个输出。这时候，你可以通过不同的索引（0, 1, 2, ...）来获取不同的输出张量。
    因此，在调用 `layer1->getOutput(0)` 时，参数 `0` 意味着你想要从 `layer1` 获取第一个（或唯一的）输出张量。这个输出张量随后被用作激活函数层的输入，`addActivation` 方法用于在这个输出上应用一个激活函数，本例中是 Sigmoid 激活函数。
    这种设计允许 TensorRT 构建的网络有更大的灵活性和复杂性，因为它可以处理多输出层的情况。
    */
    auto prob = network->addActivation(*layer1->getOutput(0), nvinfer1::ActivationType::kSIGMOID);

    // 将我们需要的prob标记为输出
    network->markOutput(*prob->getOutput(0));

    printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f); // 256Mib
    /*
    1. `config->setMaxWorkspaceSize(1 << 28);`
    这一行设置了构建优化器在优化网络时可以使用的最大工作空间大小。工作空间是一块临时内存区域，TensorRT 在优化和执行层操作时用于存放临时数据。`1 << 28` 表示将 1 左移 28 位，相当于 \(2^{28}\) 字节，或者 256MB。这意味着 TensorRT 在构建和优化网络时，最多可以使用 256MB 的内存作为工作空间。
    分配足够的工作空间对于确保网络能够被成功优化是非常重要的。如果分配的工作空间太小，可能导致构建过程失败或者运行时性能下降。
    2. `builder->setMaxBatchSize(1);`
    这行代码设置了网络支持的最大批量大小（batch size）。在这个例子中，将最大批量大小设置为 1 表示网络被优化以一次处理一个数据样本。批量大小是指在网络前向传播过程中一次性并行处理的数据样本数量。较小的批量大小可以减少内存使用，但可能会影响处理效率；较大的批量大小可以提高吞吐量，但会增加内存需求。
    设置最大批量大小是在使用 TensorRT 早期版本中，构建显式批处理模式之前的引擎时必须要做的。在更现代的 TensorRT 版本中，通常建议使用显式批量大小（通过 `createNetworkV2` 的标志参数指定），这可以提供更大的灵活性和控制。
    总之，这两行代码是在配置 TensorRT 构建过程中的内存使用和批量处理能力，以确保网络可以高效地在特定的硬件上运行。
    */
    config->setMaxWorkspaceSize(1 << 28);
    builder->setMaxBatchSize(1);    // 推理 engine 模型文件

    // ----------------------------- 3. 生成engine模型文件 -----------------------------
    //TensorRT 7.1.0版本已弃用buildCudaEngine方法，统一使用buildEngineWithConfig方法
    /*
    这行代码是使用 NVIDIA TensorRT 的 API 在 C++ 环境中构建一个推理引擎 (`ICudaEngine`)。构建过程中会考虑网络定义 (`INetworkDefinition`) 和构建配置 (`IBuilderConfig`)，以优化模型的推理性能。
    ### 解读代码
    - `nvinfer1::ICudaEngine* engine`：声明了一个指向 `ICudaEngine` 类型的指针，名为 `engine`。`ICudaEngine` 是 TensorRT 中的一个核心接口，代表了一个优化后可用于执行推理的引擎。
    - `builder->buildEngineWithConfig(*network, *config);`：这是调用 `builder` 对象的 `buildEngineWithConfig` 方法，该方法负责根据提供的网络定义 (`network`) 和构建配置 (`config`) 来构建一个优化后的推理引擎。
    ### 参数列表
    - `*network`：这是一个 `INetworkDefinition` 类型的对象，它包含了模型的网络结构定义，包括模型的层、输入、输出等信息。这里通过解引用 (`*`) 传递了一个 `INetworkDefinition` 对象的引用。
    - `*config`：这是一个 `IBuilderConfig` 类型的对象，它包含了构建过程中的各种配置选项，如最大工作空间大小 (`setMaxWorkspaceSize`)，精度设置，优化配置文件等。通过解引用传递了一个 `IBuilderConfig` 对象的引用。
    ### 功能和目的
    这行代码的功能是将网络定义 (`INetworkDefinition`) 和构建配置 (`IBuilderConfig`) 结合起来，构建一个用于执行推理的优化后的 CUDA 引擎 (`ICudaEngine`)。这个过程包括分析和优化网络结构，选择最适合当前硬件的算法，以及可能的内存优化等，最终目的是提高模型在特定硬件上的推理性能。
    构建完成的引擎可以被用于执行推理，它封装了所有GPU计算的细节，使得开发者可以高效地在 NVIDIA GPU 上部署其深度学习模型。
    */
    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    if(engine == nullptr){
        printf("Build engine failed. \n");
        return -1;
    }
    // ----------------------------- 4. 序列化模型文件并存储 -----------------------------
    // 将模型序列化，并储存为文件
    /*
    这行代码是在使用 NVIDIA TensorRT 的 API 来序列化（serialize）一个优化后的推理引擎（engine），并将序列化的数据保存在 `IHostMemory` 对象中。这个对象之后可以用来保存到磁盘上或在网络上传输，以便之后可以快速加载和执行推理，而不需要重新进行模型优化过程。
    具体来说：
    - `nvinfer1::IHostMemory* model_data`：这是一个指针，指向 `IHostMemory` 类型的对象。`IHostMemory` 是 TensorRT 提供的一个接口，用于访问和管理主机（CPU）内存中的数据。
    - `engine->serialize()`：这个方法调用是在对 `engine` 对象执行序列化操作。`engine` 是一个指向 `ICudaEngine` 类型的对象，代表了一个优化后的推理引擎。序列化操作将这个引擎的所有配置、网络模型结构和权重等转换成一个连续的内存块，这样就可以在不需要原始模型定义的情况下进行推理。
    在序列化过程完成后，`model_data` 指针就指向了包含序列化引擎数据的内存区域。通过这种方式，你可以将序列化的引擎数据保存到文件中，例如使用文件 I/O 操作，或者将其发送到另一个系统进行推理。
    简单来说，这行代码的作用是将 TensorRT 的推理引擎序列化为一块内存区域，方便持久化存储或网络传输，以便后续快速加载和使用。
    */
    nvinfer1::IHostMemory* model_data = engine->serialize();
    FILE* f = fopen("engine.trtmodel", "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    // 卸载顺序按照构建顺序倒序
    /*
    在使用 NVIDIA TensorRT 或其他需要显式内存管理的框架时，销毁（destroy）对象是一种重要的资源管理实践。这些 `destroy` 调用的必要性来自于几个方面：
    ### 1. **内存管理**
    最直接的原因是内存管理。在 C++ 中，许多资源，如动态分配的内存、文件句柄、网络连接等，不会自动管理。如果不显式释放这些资源，就会导致内存泄漏，随着程序的运行时间增长，消耗的内存量也会不断增加，最终可能耗尽系统资源，导致程序崩溃或影响系统性能。
    ### 2. **资源释放**
    除了内存之外，程序还可能使用其他有限资源，比如 GPU 上的缓冲区、硬件加速器的特定能力等。及时释放这些资源可以确保它们被其他进程或程序部分再次利用，优化系统资源的使用。
    ### 3. **对象生命周期管理**
    在复杂的系统中，正确管理对象的生命周期对于维持程序的稳定性和可预测性至关重要。显式销毁对象有助于清晰地定义对象的使用范围，减少由于对象状态不明确导致的错误。
    ### 4. **依赖关系管理**
    在一些情况下，对象之间可能存在依赖关系，错误的销毁顺序可能导致运行时错误。例如，一个对象可能依赖于另一个对象的状态。正确的销毁顺序有助于维护这些依赖关系，确保程序的正确性。
    ### 5. **编程实践**
    良好的编程习惯要求开发者对使用的资源负责，无论是在高级语言中还是在像 C++ 这样的低级语言中。显式销毁对象反映了代码的清晰结构和负责任的资源管理。
    ### 总结
    `destroy` 调用是确保资源被正确管理和释放的重要手段，特别是在直接操作底层资源的环境中，如使用 TensorRT 这类高性能计算库时。遵循这些实践可以帮助避免资源泄漏，提高程序的效率和稳定性。
    */
    model_data->destroy();
    engine->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
    printf("Done.\n");
    return 0;
}
