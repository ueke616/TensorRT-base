import onnx
import torch
import torch.nn as nn
import onnx.helper as helper
import numpy as np


class Preprocess(nn.Module):
    def __init__(self):
        super(Preprocess, self).__init__()
        self.mean = torch.rand(1, 1, 1, 3)
        self.std  = torch.rand(1, 1, 1, 3)
    
    def forward(self, x):
        # x = B x H x W x C     Uint8
        # y = B x C x H x W     Float32     减去均值除以标准差
        x = (x.float() / 255.0 - self.mean) / self.std
        x = x.permute(0, 3, 1, 2)       # 换轴
        return x


def read_conv_node_weight(model, node_name="model.0.conv.weight"):
    for item in model.graph.initializer:
        if item.name == node_name:
            print("shape: ", item.dims)
            
            weight = np.frombuffer(item.raw_data, dtype=np.float32).reshape(*item.dims)
            print(weight.shape)


def read_node_output_data(model, next_node="/model.24/Constant_3_output_0"):
    # output: "/model.24/Constant_3_output_0"
    # name: "/model.24/Constant_3"
    # op_type: "Constant"
    # attribute {
    #   name: "value"
    #   type: TENSOR
    #   t {
    #     dims: 1
    #     dims: 80
    #     data_type: 1
    #     raw_data: "\000\000\000\000\000\000\200?\000\000\000@\000\000@@\000\000\200@\000\000\240@\000\000\300@\000\000\340@\000\000\000A\000\000\020A\000\000 A\000\0000A\000\000@A\000\000PA\000\000`A\000\000pA\000\000\200A\000\000\210A\000\000\220A\000\000\230A\000\000\240A\000\000\250A\000\000\260A\000\000\270A\000\000\300A\000\000\310A\000\000\320A\000\000\330A\000\000\340A\000\000\350A\000\000\360A\000\000\370A\000\000\000B\000\000\004B\000\000\010B\000\000\014B\000\000\020B\000\000\024B\000\000\030B\000\000\034B\000\000 B\000\000$B\000\000(B\000\000,B\000\0000B\000\0004B\000\0008B\000\000<B\000\000@B\000\000DB\000\000HB\000\000LB\000\000PB\000\000TB\000\000XB\000\000\\B\000\000`B\000\000dB\000\000hB\000\000lB\000\000pB\000\000tB\000\000xB\000\000|B\000\000\200B\000\000\202B\000\000\204B\000\000\206B\000\000\210B\000\000\212B\000\000\214B\000\000\216B\000\000\220B\000\000\222B\000\000\224B\000\000\226B\000\000\230B\000\000\232B\000\000\234B\000\000\236B"
    #   }
    # }
    for item in model.graph.node:
        if item.op_type == "Constant":
            if next_node in item.output:
                t = item.attribute[0].t
                data = np.frombuffer(t.raw_data, dtype=np.float32).reshape(*t.dims)
                print(data.shape)
                print(data)

def edit_output_data(model, next_node="/model.24/Constant_4_output_0"):
    for item in model.graph.node:
        if item.op_type == "Constant":
            if next_node in item.output:
                t = item.attribute[0].t
                # print(np.frombuffer(t.raw_data, dtype=np.int64))
                t.raw_data = np.array([100], dtype=np.int64).tobytes()


def add_property_to_node(model, node_name="/model.24/Reshape_4"):
    for item in model.graph.node:
        if item.name == node_name:
            # print(item)
            print(helper.make_node(
                op_type="Reshape", 
                inputs=["/model.24/m.2/Conv_output_0", "/model.24/Constant_36_output_0"], 
                outputs=["/model.24/Reshape_4_output_0"], 
                name="/model.24/Reshape_4"
            ))
            newitem = helper.make_node(
                op_type="Reshape", 
                inputs=["/model.24/m.2/Conv_output_0", "/model.24/Constant_36_output_0"], 
                outputs=["/model.24/Reshape_4_output_0"], 
                name="/model.24/Reshape_4",
                myname="like"
            )
            item.CopyFrom(newitem)


def del_node(model, node_name="/model.24/Transpose_1"):
    # 如果输入输出有多个, 那只保留第一个
    find_node_with_input = lambda name: [item for item in model.graph.node if name in item.input][0]
    find_node_with_output = lambda name: [item for item in model.graph.node if name in item.output][0]

    remove_nodes = []
    for item in model.graph.node:
        if item.name == node_name:
            # 上一个节点的输出是当前节点的输入
            prev = find_node_with_output(item.input[0])
            
            # 下一个节点的输入是当前节点的输出
            next = find_node_with_input(item.output[0])
            next.input[0] = prev.output[0]
            remove_nodes.append(item)

    for item in remove_nodes[::-1]:
        print(item)
        model.graph.node.remove(item)

    onnx.save(model, "workspace/new.onnx")
    return model


def edit_static_input_to_dynamic(model):
    static_batch_size = 10
    # print(model.graph.input)
    # 先找到 input 节点的下一个节点记录一下, 以免信息丢失
    find_input_next_node = lambda name: [item for item in model.graph.node if name == 'input'][0]
    input_out = find_input_next_node(model.graph.input[0].name)
    input_out.input[0] = 'images'
    # print(input_out.input)
    
    # 修改 input 节点的名字, 并修改输入图片的H,W, 改成动态的
    new_input = helper.make_tensor_value_info("images", 1, [static_batch_size, 3, "height", "width"])
    model.graph.input[0].CopyFrom(new_input)

    new_output = helper.make_tensor_value_info("output", 1, [static_batch_size, "anchors", "classes"])
    model.graph.output[0].CopyFrom(new_output)


def add_node_to_graph(model):
    pre = Preprocess()
    torch.onnx.export(
        pre,
        (torch.zeros(1, 640, 640, 3, dtype=torch.uint8),),
        "workspace/preprocess.onnx"
    )
    pre_onnx = onnx.load("workspace/preprocess.onnx")
    
    # 0. 先把 pre_onnx 的所有节点以及输入输出名称都加上前缀
    # 1. 先把 yolov5s 中的image为输入的节点, 修改为 pre_onnx 的输出节点
    # 2. 把 pre_onnx 的 node 全部放到 yolov5s 的 node 中
    # 3. 把 pre_onnx 的输入名称作为yolov5s的 input 名称
    for n in pre_onnx.graph.node:
        n.name = f"pre/{n.name}"
        
        for i in range(len(n.input)):
            n.input[i]     = f"pre/{n.input[i]}"
        
        for i in range(len(n.output)):
            n.output[i]    = f"pre/{n.output[i]}"
    
    for n in model.graph.node:
        if n.name == "/model.0/conv/Conv":
            n.input[0] = "pre/" + pre_onnx.graph.output[0].name
    
    for n in pre_onnx.graph.node:
        model.graph.node.append(n)
    
    input_name = "pre/" + pre_onnx.graph.input[0].name
    model.graph.input[0].CopyFrom(pre_onnx.graph.input[0])
    model.graph.input[0].name = input_name
    
    onnx.save(model, "workspace/new.onnx")

if __name__ == "__main__":
    # model = onnx.load("./1.4-onnx-editor/workspace/yolov5s.onnx")
    model = onnx.load("./workspace/yolov5s.onnx")
    # model = del_node(model, node_name="/model.24/Transpose_1")
    add_node_to_graph(model)
    

