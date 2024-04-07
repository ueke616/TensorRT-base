#!/bin/bash

# 请修改protoc为你要使用的版本
unset CUDA_HOME
unset LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/data/nvidia/TensorRT-8.6.1.6/lib:/data/nvidia/TensorRT-8.6.1.6/include
protoc=/usr/bin/protoc

rm -rf workspace/pbout
mkdir -p workspace/pbout

$protoc onnx-ml.proto --cpp_out=workspace/pbout --python_out=workspace/pbout