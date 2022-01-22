'''
Hyperparameters     Values

Kernel size         3
Kernel initializer  uniform
Pool method         max
Activations         relu
Learning rate       1.0
'''

import torch
from torch.nn import BatchNorm2d as bn

def maxPool_2x2():
    return torch.nn.MaxPool2d(kernel_size=2, stride=2)

# 固定为3x3的 torch.nn.Conv2d
def conv3x3(in_channels, out_channels, kernel_size=3, stride=1, padding=(1, 1)):
    return torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    # def __init__(self, in_channels, out_channels,
    # kernel_size, stride=1,
    # padding=0, dilation=1,
    # groups=1, bias=True, padding_mode='zeros'):


class TrackNet3(torch.nn.Module):

    def __init__(self, input_height=288, input_width=512):
        super(TrackNet3, self).__init__()

        # Layer1
        self.conv1 = conv3x3(9, 64)
        self.nor1 = bn(64)

        # Layer2
        self.conv2 = conv3x3(64, 64)
        self.nor2 = bn(64)

        # Layer3
        # self.max3 = maxPool_2by2()
        self.conv3 = conv3x3(64, 64, stride=2)
        self.nor3 = bn(64)



        # Layer4
        self.conv4 = conv3x3(64, 128)
        self.nor4 = bn(128)

        # Layer5
        self.conv5 = conv3x3(128, 128)
        self.nor5 = bn(128)

        # Layer6
        # self.max6 = maxPool_2x2()
        self.conv6 = conv3x3(128, 128, stride=2)
        self.nor6 = bn(128)


        # Layer7
        self.conv7 = conv3x3(128, 256)
        self.nor7 = bn(256)

        # Layer8
        self.conv8 = conv3x3(256, 256)
        self.nor8 = bn(256)

        # Layer9
        self.conv9 = conv3x3(256, 256)
        self.nor9 = bn(256)

        # Layer10
        # self.max10 = maxPool_2x2()
        self.conv10 =conv3x3(256, 256, stride=2)
        self.nor10 = bn(256)


        # Layer11
        self.conv11 = conv3x3(256, 512)
        self.nor11 = bn(512)

        # Layer12
        self.conv12 = conv3x3(512, 512)
        self.nor12 = bn(512)

        # Layer13
        self.conv13 = conv3x3(512, 512)
        self.nor13 = bn(512)

        # Layer14  没有learnable参数，不用初始化
        # forward中：x = torch.cat((self.leo_upsample(x), x3), 1)
        # 沿着 x3是layer_9的output

        # Layer15
        self.conv15 = conv3x3(768, 256)
        self.nor15 = bn(256)

        # Layer16
        self.conv16 = conv3x3(256, 256)
        self.nor16 = bn(256)

        # Layer17
        self.conv17 = conv3x3(256, 256)
        self.nor17 = bn(256)

        # Layer18
        # 对应forward中的：
        # x = torch.cat((self.leo_upsample(x), x某), 1)

        # Layer19
        self.conv19 = conv3x3(384, 128)
        self.nor19 = bn(128)

        # Layer20
        self.conv20 = conv3x3(128, 128)
        self.nor20 = bn(128)

        # Layer21
        # 对应forward中的：
        # x = torch.cat((self.leo_upsample(x), x某), 1)

        # Layer22
        self.conv22 = conv3x3(192, 64)
        self.nor22 = bn(64)

        # Layer23
        self.conv23 = conv3x3(64, 64)
        self.nor23 = bn(64)

        # Layer24
        self.conv24 = torch.nn.Conv2d(64, 3, kernel_size=1)
        # 默认：stride=1, padding=0, bias=True, padding_mode='zeros'

        # 其他
        # This operation may produce nondeterministic gradients when given tensors on a CUDA device.
        # search Reproducibility for more information.
        self.leo_upsample = torch.nn.UpsamplingBilinear2d(scale_factor=2)
        # 等价于： nn.functional.interpolate(scale_factor=2, mode='bilinear', align_corners=True)

        # 一般都用转置卷积？这里是为了省计算量？

        self.act = torch.nn.ReLU()
        #  self.act = torch.nn.ReLU(inplace=True)    #  inplace省内存，快？可能有坑？

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.nor1(x)

        x = self.conv2(x)
        x = self.act(x)
        x1 = self.nor2(x)

        x = self.conv3(x1)
        x = self.act(x)
        x = self.nor3(x)

        x = self.conv4(x)
        x = self.act(x)
        x = self.nor4(x)

        x = self.conv5(x)
        x = self.act(x)
        x2 = self.nor5(x)

        x = self.conv6(x2)
        x = self.act(x)
        x = self.nor6(x)

        x = self.conv7(x)
        x = self.act(x)
        x = self.nor7(x)

        x = self.conv8(x)
        x = self.act(x)
        x = self.nor8(x)

        x = self.conv9(x)
        x = self.act(x)
        x3 = self.nor9(x)

        x = self.conv10(x3)
        x = self.act(x)
        x = self.nor10(x)

        x = self.conv11(x)
        x = self.act(x)
        x = self.nor11(x)

        x = self.conv12(x)
        x = self.act(x)
        x = self.nor12(x)

        x = self.conv13(x)
        x = self.act(x)
        x = self.nor13(x)

        x = torch.cat((self.leo_upsample(x), x3), 1)

        x = self.conv15(x)
        x = self.act(x)
        x = self.nor15(x)

        x = self.conv16(x)
        x = self.act(x)
        x = self.nor16(x)

        x = self.conv17(x)
        x = self.act(x)
        x = self.nor17(x)

        x = torch.cat((self.leo_upsample(x), x2), 1)

        x = self.conv19(x)
        x = self.act(x)
        x = self.nor19(x)

        x = self.conv20(x)
        x = self.act(x)
        x = self.nor20(x)

        x = torch.cat((self.leo_upsample(x), x1), 1)

        x = self.conv22(x)
        x = self.act(x)
        x = self.nor22(x)

        x = self.conv23(x)
        x = self.act(x)
        x = self.nor23(x)

        x = self.conv24(x)  # 输出为3通道
        x = torch.sigmoid(x)

        return x
