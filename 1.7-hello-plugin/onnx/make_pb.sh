#!/bin/bash

# 请修改protoc为你要使用的版本protoc
unset LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/data/nvidia/TensorRT-8.6.1.6/lib:/data/nvidia/TensorRT-8.6.1.6/include
protoc=/usr/local/bin/protoc

cd onnx

echo Create directory "pbout"
rm -rf pbout
mkdir -p pbout

$protoc onnx-ml.proto --cpp_out=pbout
$protoc onnx-operators-ml.proto --cpp_out=pbout

echo Copy pbout/onnx-ml.pb.cc to ../src/onnx/onnx-ml.pb.cpp
cp pbout/onnx-ml.pb.cc           ../src/onnx/onnx-ml.pb.cpp

echo Copy pbout/onnx-operators-ml.pb.cc to ../src/onnx/onnx-operators-ml.pb.cpp
cp pbout/onnx-operators-ml.pb.cc ../src/onnx/onnx-operators-ml.pb.cpp

echo Copy pbout/onnx-ml.pb.h to ../src/onnx/onnx-ml.pb.h
cp pbout/onnx-ml.pb.h           ../src/onnx/onnx-ml.pb.h

echo Copy pbout/onnx-operators-ml.pb.h to ../src/onnx/onnx-operators-ml.pb.h
cp pbout/onnx-operators-ml.pb.h ../src/onnx/onnx-operators-ml.pb.h

echo Remove directory "pbout"
rm -rf pbout
