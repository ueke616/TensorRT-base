import torch
from torch import nn, onnx, autograd
import torch.nn.functional as F
import os


# pytorch自定义算子导出onnx计算图——————多输入的计算图
class MyStrangeOp2(autograd.Function):
    # reference: https://pytorch.org/docs/1.10/onnx.html#torch-autograd-functions
    @staticmethod
    def symbolic(g, input1, input2, weight, floatAttr, intAttr, strAttr):
        """
        这个方法可以把pytorch模型中的算子转成onnx中的节点
        g: graph,
        """
        print("================= call symbolic ====================")
        return g.op("MyStrangeOp2", input1, input2, weight, float_attr1_f=floatAttr, int_attr2_i=intAttr, str_attr3_s=strAttr)
    
    @staticmethod
    def forward(ctx, input1, input2, weight, floatAttr, intAttr, strAttr):
        return (input1 + input2) * weight 


class MyStrangeOp2Layer(nn.Module):
    def __init__(self, weight, floatAttr, intAttr, strAttr):
        super(MyStrangeOp2Layer, self).__init__()
        self.weight = weight
        self.floatAttr = floatAttr
        self.intAttr = intAttr
        self.strAttr = strAttr
    
    def forward(self, x1, x2):
        return MyStrangeOp2.apply(x1, x2, self.weight, self.floatAttr, self.intAttr, self.strAttr)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.myLayer1 = MyStrangeOp2Layer(weight=nn.Parameter(torch.ones(1, 3, 4, 4)), floatAttr=[0.1, 0.5], intAttr=[2, 2], strAttr="你好")
        self.myLayer2 = MyStrangeOp2Layer(weight=nn.Parameter(torch.ones(1, 3, 4, 4)), floatAttr=[0.5, 0.5], intAttr=[3, 3], strAttr="世界")
        self.conv1    = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv2    = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, stride=1, bias=True)
    
    def forward(self, in1, in2, in3, in4):
        x1 = self.myLayer1(in1, in2)
        x2 = self.myLayer2(in3, in4)
        x1     = self.conv1(x1)
        x2     = self.conv1(x2)
        return x1 + x2


if __name__ == '__main__':
    # 这个包对应opset11的导出代码, 如果想修改导出的细节, 可以在这里修改代码
    # import torch.onnx.symbolic_opset11
    print("对应 opset 文件夹代码在这里: ", os.path.dirname(onnx.__file__))

    model = Model().eval()
    input1 = torch.ones(1, 3, 4, 4, dtype=torch.float32)
    input2 = torch.ones(1, 3, 4, 4, dtype=torch.float32)
    input3 = torch.ones(1, 3, 4, 4, dtype=torch.float32)
    input4 = torch.ones(1, 3, 4, 4, dtype=torch.float32)

    output = model(input1, input2, input3, input4)
    print(f"iterence output = \n{output}")

    dummy1 = torch.randn(1, 3, 4, 4)
    dummy2 = torch.randn(1, 3, 4, 4)
    dummy3 = torch.randn(1, 3, 4, 4)
    dummy4 = torch.randn(1, 3, 4, 4)
    
    torch.onnx.export(
        model, 
        (dummy1, dummy2, dummy3, dummy4),   # 这里的args，是指输入给model的参数，需要传递tuple，因此用括号
        "workspace/demo2.onnx",     # 储存的文件路径
        verbose=True,     # 打印详细信息
        input_names=["image"],     # 为输入和输出节点指定名称，方便后面查看或者操作
        output_names=["output"], 
        opset_version=15,     # 这里的opset，指，各类算子以何种方式导出，对应于symbolic_opset11
        # 表示他有batch、height、width3个维度是动态的，在onnx中给其赋值为-1
        # 通常，我们只设置batch为动态，其他的避免动态
        dynamic_axes={
            "image": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        },
        # 对于插件，需要禁用onnx检查 https://pytorch.org/docs/2.0/onnx.html?highlight=torch+nn+export#torch.onnx.export
        operator_export_type=onnx.OperatorExportTypes.ONNX_FALLTHROUGH
    )
    
    print("Done!")
