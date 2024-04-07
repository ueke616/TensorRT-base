import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.conv.bias, 0)
    
    def forward(self, x):
        x = self.conv(x)
        return F.relu(x)


if __name__ == "__main__":
    # 这个包对应opset11的导出代码，如果想修改导出的细节，可以在这里修改代码
    # import torch.onnx.symbolic_opset11
    print("对应 opset 文件夹在这里: ", os.path.dirname(torch.onnx.__file__))

    model = Model()
    dummy = torch.zeros(1, 1, 3, 3)
    torch.onnx.export(
        model,
        (dummy,),       # 这里的args，是指输入给model的参数，需要传递tuple，因此用括号
        "workspace/demo.onnx",      # 储存的文件路径
        verbose=True,       # 打印详细信息
        input_names=["image"],      # 为输入和输出节点指定名称，方便后面查看或者操作
        output_names=["output"],
        opset_version=11,       # 这里的opset，指，各类算子以何种方式导出，对应于symbolic_opset11
        dynamic_axes={      # 表示他有batch、height、width3个维度是动态的，在onnx中给其赋值为-1
            "image": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        }
    )
    print("Done. !")
