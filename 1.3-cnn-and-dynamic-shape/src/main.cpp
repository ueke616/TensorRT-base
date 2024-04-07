// TensorRT include
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

class TRTLogger: public nvinfer1::ILogger{
public:
	virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override{
		if(severity <= Severity::kINFO){
			printf("%d: %s\n", severity, msg);
		}
	}
} logger;

nvinfer1::Weights make_weights(float* ptr, int n){
	nvinfer1::Weights w;
	w.count = n;
	w.type	= nvinfer1::DataType::kFLOAT;
	w.values = ptr;
	return w;
}

bool build_model(){
	TRTLogger logger;

	// 1. 定义 buffer, config 和 network
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1);

	// 2. 输入，模型结构和输出的基本信息
	const int num_input		= 1;
	const int num_output	= 1;
	float layer1_weight_values[] = {
		1.0, 2.0, 3.1,
		0.1, 0.1, 0.1,
		0.2, 0.2, 0.2
	};
	float layer1_bias_values[] = {0.0};

	// 使用动态 shape
	nvinfer1::ITensor* input = network->addInput("image", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4(-1, num_input, -1, -1));
	nvinfer1::Weights layer1_weight = make_weights(layer1_weight_values, 9);
	nvinfer1::Weights layer1_bias	= make_weights(layer1_bias_values, 1);
	// nvinfer1::DimsHW(3, 3) 创建了一个 DimsHW 对象，其高度和宽度都被设置为 3。这在实际应用中常见于定义 3x3 的卷积核大小
	auto layer1 = network->addConvolution(*input, num_output, nvinfer1::DimsHW(3, 3), layer1_weight, layer1_bias);
	layer1->setPadding(nvinfer1::DimsHW(1, 1));

	auto prob = network->addActivation(*layer1->getOutput(0), nvinfer1::ActivationType::kRELU);
	// 将我们需要的prob标记为输出
	network->markOutput(*prob->getOutput(0));

	int maxBatchSize = 10;
	printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f);
    // 配置暂存存储器，用于layer实现的临时存储，也用于保存中间激活值
    config->setMaxWorkspaceSize(1 << 28);  // 2048 MB

	// 2.1 关于 profile(配置文件)
	// 如果模型有多个输入, 则必须有多个 profile
	auto profile = builder->createOptimizationProfile();
	// 配置最小允许 1x1x3x3  最优配置和最小配置一样
	profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, num_input, 3, 3));
	profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, num_input, 3, 3));
	// 配置最大允许10x1x5x5
	// if networkDims.d[i] != -1, then minDims.d[i] == optDims.d[i] == maxDims.d[i] == networkDims.d[i]
	profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(maxBatchSize, num_input, 5, 5));
	config->addOptimizationProfile(profile);

	nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);  // 构建一个用于执行推理的优化后的 CUDA 引擎 
	if(engine == nullptr){
		printf("Build engine failed. \n");
		return false;
	}

	// 3. 序列化模型并保存
	nvinfer1::IHostMemory* model_data = engine->serialize();
	FILE* f = fopen("engine.trtmodel", "wb");
	fwrite(model_data->data(), 1, model_data->size(), f);
	fclose(f);

	// 卸载顺序按照构建顺序倒序
	model_data->destroy();
	engine->destroy();
	network->destroy();
    config->destroy();
    builder->destroy();
    printf("Done.\n");
    return true;
}

vector<unsigned char> load_file(const string& file){
	ifstream in(file, ios::in | ios::binary);
	if(!in.is_open())
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

void inference(){
	// 1. 加载 model并反序列化
	TRTLogger logger;
	auto engine_data = load_file("engine.trtmodel");
	nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
	nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
	if(engine == nullptr){
		printf("Deserialize cuda engine failed. \n");
		runtime->destroy();
		return;
	}

	nvinfer1::IExecutionContext* execution_context = engine->createExecutionContext();
	cudaStream_t stream = nullptr;
	cudaStreamCreate(&stream);

	// 2. 输入与输出
	float input_data_host[] = {
		1, 	1, 	1,	// batch 0
		1, 	1, 	1,
		1, 	1, 	1,

		-1,	 1,	 1,	// batch 1
		1,	 0,	 1,
		1,	 1,  -1
	};
	float* input_data_device = nullptr;

	// 3x3输入, 对应3x3输出
	int ib = 2;
	int iw = 3;
	int ih = 3;
	float output_data_host[ib * iw * ih];
	float* output_data_device = nullptr;
	cudaMalloc(&input_data_device, sizeof(input_data_host));
	cudaMalloc(&output_data_device, sizeof(output_data_host));
	cudaMemcpyAsync(input_data_device, input_data_host, sizeof(input_data_host), cudaMemcpyHostToDevice, stream);

	// 3. 推理
	// 明确当前推理时, 使用的数据输入大小
	execution_context->setBindingDimensions(0, nvinfer1::Dims4(ib, 1, ih, iw)); // 绝大部分只考虑batch的动态
	float* bindings[] = {input_data_device, output_data_device};
	bool success	= execution_context->enqueueV2((void**)bindings, stream, nullptr);
	cudaMemcpyAsync(output_data_host, output_data_device, sizeof(output_data_host), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

	// 4. 输出结果
	for(int b = 0; b < ib; ++b) {
		printf("batch %d. output_data_host = \n", b);
		for(int i = 0; i < iw * ih; ++i){
			printf("%f, ", output_data_host[b * iw * ih + i]);
			if((i + 1) % iw == 0)
				printf("\n");
		}
	}

	printf("Clean memory \n");
	cudaStreamDestroy(stream);
	cudaFree(input_data_device);
	cudaFree(output_data_device);
	execution_context->destroy();
	engine->destroy();
	runtime->destroy();	
}

int main(){
	if(!build_model()){
		return -1;
	}
	inference();
	return 0;
}


