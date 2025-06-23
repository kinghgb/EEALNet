import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import math
import sys
from mmcv.cnn import build_norm_layer
sys.path.insert(0, '../../')
from mmcv.ops.carafe import CARAFEPack
from pvtv2 import pvt_v2_b0,pvt_v2_b1,pvt_v2_b2,pvt_v2_b2_li,pvt_v2_b3,pvt_v2_b4,pvt_v2_b5

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)
affine_par=True
class EEALNet(nn.Module):
    def __init__(self, fun_str = 'pvt_v2_b0'):
        super().__init__()
        self.backbone,embedding_dims = eval(fun_str)()
        self.uaspp=GPM()
        self.egrm2 = EGRM(embedding_dims[1]//2, embedding_dims[3] // 8,
                                     opr_kernel_size=7, iterations=1)
        self.egrm1 = EGRM(embedding_dims[0], embedding_dims[1] // 2,
                                     opr_kernel_size=7,
                                     iterations=1)
        self.fgrm0 = FGRM(embedding_dims[0], embedding_dims[0],focus_background=False,
                                     opr_kernel_size=7, iterations=1)
        self.mfm0 = MFM(cur_in_channels=embedding_dims[0], low_in_channels=embedding_dims[0],
                                  out_channels=embedding_dims[0])
        self.mfm1 = MFM(cur_in_channels=embedding_dims[1], low_in_channels=embedding_dims[0],
                                  out_channels=embedding_dims[0])
        self.mfm2 = MFM(cur_in_channels=embedding_dims[2], low_in_channels=embedding_dims[0],
                                  out_channels=embedding_dims[0])

        self.cbr = CBR(in_channels=embedding_dims[3], out_channels=embedding_dims[3] // 8,
                                           kernel_size=3, stride=1,
                                           dilation=1, padding=1)


        self.predict_conv = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dims[0], out_channels=1, kernel_size=3, padding=1, stride=1))

        # self.pix3 = Pix(cur_in_channels=embedding_dims[0]//2, low_in_channels=embedding_dims[3] // 8,
        #                                      out_channels=embedding_dims[0] // 4)  # 16



    def forward(self, x):
        # byxhz
        layer = self.backbone(x)
        s5 = self.cbr(layer[3])
        s5 = self.uaspp(s5)
        s4 = self.mfm2(layer[2],s5)
        # s4 = self.uaspp(s4)
        s3 = self.mfm1(layer[1],s4)
        # s3 = self.uaspp(s3)
        s2 = self.mfm0(layer[0],s3)

        # s2 = self.uaspp(s2)
        predict3=F.interpolate(s2, size=s5.size()[2:], mode='bilinear', align_corners=True)
        predict3=self.predict_conv(predict3)
        # focus
        fgc3, predict2 = self.egrm2(s4, s5, predict3)

        fgc2, predict1 = self.egrm1(s3, fgc3, predict2)

        fgc1, predict0 = self.fgrm0(s2, fgc2, predict1)

        # rescale

        predict3 = F.interpolate(predict3, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict2 = F.interpolate(predict2, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict1 = F.interpolate(predict1, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict0 = F.interpolate(predict0, size=x.size()[2:], mode='bilinear', align_corners=True)

        return predict3, predict2, predict1, predict0
class SENet(nn.Module):

    def __init__(self, in_dim, ratio=2):
            super(SENet, self).__init__()
            self.dim = in_dim
            self.fc = nn.Sequential(nn.Linear(in_dim, self.dim // ratio, bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(self.dim // ratio, in_dim, bias=False))
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
            b, c, _, _ = x.size()
            y = F.adaptive_avg_pool2d(x, (1, 1)).view(b, c)
            y = self.sigmoid(self.fc(y)).view(b, c, 1, 1)

            output = y.expand_as(x)*x

            return output
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, need_relu=True,
                 bn=nn.BatchNorm2d):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = bn(out_channels)
        self.relu = nn.ReLU()
        self.need_relu = need_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.need_relu:
            x = self.relu(x)
        return x
class PAM(nn.Module):
    def __init__(self, in_channels):
        super(PAM, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x

        return out
class GPM(nn.Module):
    def __init__(self, dilation_series=[6, 12, 18], padding_series=[6, 12, 18], depth=64):
        # def __init__(self, dilation_series=[2, 5, 7], padding_series=[2, 5, 7], depth=128):
        super(GPM, self).__init__()
        self.se = SENet(64)
        self.branch_main = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            BasicConv2d(64, depth, kernel_size=1, stride=1)
        )
        self.branch0 = BasicConv2d(64, depth, kernel_size=1, stride=1)
        self.branch1 = BasicConv2d(64, depth, kernel_size=3, stride=1, padding=padding_series[0],
                                   dilation=dilation_series[0])
        self.branch2 = BasicConv2d(64, depth, kernel_size=3, stride=1, padding=padding_series[1],
                                   dilation=dilation_series[1])
        self.branch3 = BasicConv2d(64, depth, kernel_size=3, stride=1, padding=padding_series[2],
                                   dilation=dilation_series[2])
        self.head = nn.Sequential(
            BasicConv2d(depth*5, 64, kernel_size=3, padding=1),
            PAM(64)
        )
        self.channels=nn.Conv2d(in_channels=64, out_channels=depth, kernel_size=1)
        self.channels1 = nn.Conv2d(in_channels=128, out_channels=depth, kernel_size=1)
        # self.out = nn.Sequential(
        #     nn.Conv2d(256, 64, 3, padding=1),
        #     nn.BatchNorm2d(64, affine=affine_par),
        #     nn.PReLU(),
        #     nn.Dropout2d(p=0.1),
        #     nn.Conv2d(64, 1, 1)
        # )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        size = x.shape[2:]
        branch_avg = self.branch_main(x)
        branch_avg = self.se(branch_avg)
        branch_avg = F.interpolate(branch_avg, size=size, mode='bilinear', align_corners=True)
        branch0 = self.branch0(x)
        branchse0=self.se(branch0)
        branch1=self.channels(x)
        branch1 = self.branch1(branch1+branch0)
        branchse1 = self.se(branch1)
        branch2 = self.channels(x)
        branch2 = self.branch2(branch2+branch1)
        branchse2 = self.se(branch2)
        branch3 = self.channels(x)
        branch3 = self.branch3(branch3+branch2)
        branchse3 = self.se(branch3)
        branch32=torch.cat([branchse3,branchse2], 1)
        branch32=self.channels1(branch32)
        branch21=torch.cat([branch32,branchse1], 1)
        branch21 = self.channels1(branch21)
        branch10=torch.cat([branch21,branchse0], 1)
        branch10 = self.channels1(branch10)
        # x1=branch0*branch1
        # x2=branch1*branch2
        # x3=branch2*branch3
        # y2=x2*x1
        # z2=x2*x3
        # y3=y2*z2
        # out=branch0+x1+y2+y3
        out = torch.cat([branch_avg, branch10, branch21, branch32, branchse3], 1)
        out = self.head(out)
        # out = self.out(out)
        return out
class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,dilation=1):
        super(CBR, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding,dilation=dilation)
        self.norm_cfg = {'type': 'BN', 'requires_grad': True}
        _, self.bn = build_norm_layer(self.norm_cfg, out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)

        return x


class MFM(nn.Module):
    def __init__(self,cur_in_channels=64,low_in_channels=32,out_channels=64):
        super(MFM,self).__init__()
        self.cur_in_channels = cur_in_channels
        self.cur_conv = nn.Sequential(
            nn.Conv2d(in_channels=cur_in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.GELU()
        )
        self.low_conv = nn.Sequential(
            nn.Conv2d(in_channels=low_in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.GELU()
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=2 * out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.GELU()
        )
        self.sigmord=nn.Sigmoid()
        self.cbr = CBR(in_channels=16, out_channels=1,
                   kernel_size=3, stride=1,
                   dilation=1, padding=1)
        self.down = CBR(in_channels=out_channels, out_channels=1,
                       kernel_size=3, stride=2,
                       dilation=1, padding=1)
        self.channel = CBR(in_channels=16, out_channels=64,
                       kernel_size=3, stride=1,
                       dilation=1, padding=1)
        self.se = SENet(64)
    def forward(self,x_cur,x_low):
        pixel_shuffle = torch.nn.PixelShuffle(2)
        x_cur = self.cur_conv(x_cur)
        x_low = self.low_conv(x_low)
        x_low1 = pixel_shuffle(x_low)
        x_low1 = self.cbr(x_low1)
        x_low1 = self.sigmord(x_low1)
        x1=x_low1*x_cur+x_cur
        x_cur1=self.down(x_cur)
        x_cur1=self.sigmord(x_cur1)
        x2=x_cur1*x_low+x_low
        x2=pixel_shuffle(x2)
        x2=self.channel(x2)
        x = torch.cat((x1,x2),dim=1)
        x = self.out_conv(x)
        x = self.se(x)
        return x


import numpy as np
import cv2

def get_open_map(input,kernel_size,iterations):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    open_map_list = map(lambda i: cv2.dilate(i.permute(1, 2, 0).detach().numpy(), kernel=kernel, iterations=iterations), input.cpu())
    open_map_tensor = torch.from_numpy(np.array(list(open_map_list)))
    return open_map_tensor.unsqueeze(1).cuda()

class Basic_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Basic_Conv, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class FGRM(nn.Module):
    def __init__(self, channel1, channel2,focus_background = True, opr_kernel_size = 3,iterations = 1):
        super(FGRM, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2
        self.focus_background = focus_background
        self.up = nn.Sequential(nn.Conv2d(self.channel2, 4*self.channel1, 3, 1, 1),
                                nn.BatchNorm2d(4*self.channel1), nn.ReLU())
        self.input_map = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), nn.Sigmoid())

        #只用来查看参数
        self.increase_input_map = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=1))
        self.output_map = nn.Conv2d(self.channel1, 1, 7, 1, 3)
        self.beta = nn.Parameter(torch.ones(1))


        self.conv2 = nn.Conv2d(in_channels=self.channel1, out_channels=self.channel1, kernel_size=3, padding=1,
                               stride=1)

        self.conv_cur_dep1 = Basic_Conv(self.channel1, self.channel1, 3, 1, 1)

        self.conv_cur_dep2 = Basic_Conv(in_channels=self.channel1, out_channels=self.channel1, kernel_size=3,
                                       padding=1, stride=1)

        self.conv_cur_dep3 = Basic_Conv(in_channels=self.channel1, out_channels=self.channel1, kernel_size=3,
                                       padding=1, stride=1)

        self.opr_kernel_size = opr_kernel_size

        self.iterations = iterations


    def forward(self, cur_x, dep_x, in_map):
        # x; current-level features     cur_x
        # y: higher-level features    dep_x
        # in_map: higher-level prediction
        pixel_shuffle = torch.nn.PixelShuffle(2)
        dep_x = self.up(dep_x)
        dep_x=pixel_shuffle(dep_x)
        input_map = self.input_map(in_map)

        if self.focus_background:
            self.increase_map = self.increase_input_map(get_open_map(input_map, self.opr_kernel_size, self.iterations) - input_map)
            b_feature = cur_x * self.increase_map #当前层中,关注深层部分没有关注的部分

        else:
            b_feature = cur_x * input_map  #在当前层中，对深层部分关注的部分更加关注，同时也关注一下其他部分
        #b_feature = cur_x
        fn = self.conv2(b_feature)


        refine2 = self.conv_cur_dep1(dep_x+self.beta * fn)
        refine2 = self.conv_cur_dep2(refine2)
        refine2 = self.conv_cur_dep3(refine2)

        output_map = self.output_map(refine2)

        return refine2, output_map

class EGRM(nn.Module):
    def __init__(self, channel1, channel2,opr_kernel_size = 3,iterations = 1):
        super(EGRM, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2
        #self.focus_background = focus_background
        self.up = nn.Sequential(nn.Conv2d(self.channel2, 4*self.channel1, 7, 1, 3),
                                nn.BatchNorm2d(4*self.channel1), nn.ReLU())
        self.input_map = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), nn.Sigmoid())

        #只用来查看参数
        self.increase_input_map = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=1))
        self.output_map = nn.Conv2d(self.channel1, 1, 7, 1, 3)
        self.beta = nn.Parameter(torch.ones(1))


        self.conv2 = nn.Conv2d(in_channels=self.channel1, out_channels=self.channel1, kernel_size=3, padding=1,
                               stride=1)

        self.conv_cur_dep1 = Basic_Conv( self.channel1, self.channel1, 3, 1, 1)

        self.conv_cur_dep2 = Basic_Conv(in_channels=self.channel1, out_channels=self.channel1, kernel_size=3,
                                       padding=1, stride=1)

        self.conv_cur_dep3 = Basic_Conv(in_channels=self.channel1, out_channels=self.channel1, kernel_size=3,
                                       padding=1, stride=1)

        self.opr_kernel_size = opr_kernel_size

        self.iterations = iterations


    def forward(self, cur_x, dep_x, in_map):
        # x; current-level features     cur_x
        # y: higher-level features    dep_x
        # in_map: higher-level prediction

        pixel_shuffle = torch.nn.PixelShuffle(2)
        dep_x = self.up(dep_x)
        dep_x = pixel_shuffle(dep_x)#fi+1
        in_map1=1-in_map
        input_map = self.input_map(in_map)#pi+1
        input_map1 = self.input_map(in_map1)


        increase_map = get_open_map(input_map, self.opr_kernel_size, self.iterations) - input_map
        increase_map1 =get_open_map(increase_map, self.opr_kernel_size, self.iterations) - increase_map
        # b_feature = cur_x * self.increase_map #当前层中,关注深层部分没有关注的部分
        increase_map2 = get_open_map(input_map1, self.opr_kernel_size, self.iterations) - input_map1
        # a_feature =cur_x*self.increase_map1
        # inmap2=1-increase_map1
        # increase_map2 = get_open_map(inmap2, self.opr_kernel_size, self.iterations) - inmap2
        outmap=increase_map+increase_map1+increase_map2
        c_feature=outmap*cur_x
        fn = self.conv2(c_feature)


        refine2 = self.conv_cur_dep1(dep_x+self.beta * fn)
        refine2 = self.conv_cur_dep2(refine2)
        refine2 = self.conv_cur_dep3(refine2)

        output_map = self.output_map(refine2)

        return refine2, output_map




if __name__ =='__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    from thop import profile
    net = SARNet('pvt_v2_b3').cuda()
    data = torch.randn(1, 3, 672, 672).cuda()
    flops, params = profile(net, (data,))
    print('flops: %.2f G, params: %.2f M' % (flops / (1024*1024*1024), params / (1024*1024)))
    y = net(data)
    for i in y:
        print(i.shape)

