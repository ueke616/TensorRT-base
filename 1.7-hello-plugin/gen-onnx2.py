import torch
from torch import nn, onnx, autograd
import torch.nn.functional as F
import os


# pytorch自定义算子导出onnx计算图——————单输入的计算图
class MyStrangeOp(autograd.Function):
    # reference: https://pytorch.org/docs/1.10/onnx.html#torch-autograd-functions
    @staticmethod
    def symbolic(g, input, weight, bias, floatAttr, intAttr):
        """
        这个方法可以把pytorch模型中的算子转成onnx中的节点
        g: graph,
        """
        print("================= call symbolic ====================")
        return g.op("MyStrangeOp", input, weight, bias, float_attr_f=floatAttr, int_attr_i=intAttr),  \
            g.op("MyStringOp", input, weight, bias, float_attr_f=floatAttr, int_attr_i=intAttr)
    
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight, bias, floatAttr, intAttr):
        return input + weight, input * weight + bias


class MyStrangeOpLayer(nn.Module):
    def __init__(self, weight, bias, floatAttr, intAttr):
        super(MyStrangeOpLayer, self).__init__()
        self.weight = weight
        self.bias = bias
        self.floatAttr = floatAttr
        self.intAttr = intAttr
    
    def forward(self, x):
        return MyStrangeOp.apply(x, self.weight, self.bias, self.floatAttr, self.intAttr)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.myLayer1 = MyStrangeOpLayer(weight=nn.Parameter(torch.ones(1, 3, 4, 4)), bias=nn.Parameter(torch.ones(1, 3, 4, 4)), floatAttr=[0.1, 0.5], intAttr=[2, 2])
        self.myLayer2 = MyStrangeOpLayer(weight=nn.Parameter(torch.ones(1, 3, 4, 4)), bias=nn.Parameter(torch.ones(1, 3, 4, 4)), floatAttr=[0.5, 0.5], intAttr=[3, 3])
        self.conv1    = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv2    = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv3    = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv4    = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, stride=1, bias=True)
    
    def forward(self, x):
        x1, x2 = self.myLayer1(x)
        x3, x4 = self.myLayer2(x)
        x1     = self.conv1(x1)
        x2     = self.conv1(x2)
        x3     = self.conv1(x3)
        x4     = self.conv1(x4)
        return x1 + x2 + x3 + x4


if __name__ == '__main__':
    # 这个包对应opset11的导出代码, 如果想修改导出的细节, 可以在这里修改代码
    # import torch.onnx.symbolic_opset11
    print("对应 opset 文件夹代码在这里: ", os.path.dirname(onnx.__file__))

    model = Model().eval()
    input = torch.ones(1, 3, 4, 4, dtype=torch.float32)

    output = model(input)
    print(f"iterence output = \n{output}")

    dummy = torch.randn(1, 3, 4, 4)
    
    torch.onnx.export(
        model, 
        (dummy,),   # 这里的args，是指输入给model的参数，需要传递tuple，因此用括号
        "workspace/demo.onnx",     # 储存的文件路径
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
