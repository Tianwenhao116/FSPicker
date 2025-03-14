import torch.nn as nn
from torch.nn import functional as F
import torch
from functools import partial
import sys
import os
from torch.nn import Conv3d, Module, Linear, BatchNorm3d, ReLU
from torch.nn.modules.utils import _pair, _triple

sys.path.append("..")
from rsaBlock3d import rsaBlock
from CSA import CSA
from GLSA3d import GLSA3d
from DWConv3d import SeparableConv3D
from DMDS3D import DMDS3D_ECA

# from SoftPool import SoftPool3d
try:
    from model.sync_batchnorm import SynchronizedBatchNorm3d
except:
    pass


def normalization(planes, norm='bn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    elif norm == 'sync_bn':
        m = SynchronizedBatchNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m


# Residual 3D UNet
class ResidualUNet3D(nn.Module):
    def __init__(self, f_maps=[32, 64, 128, 256], in_channels=1, out_channels=13,
                 args=None, use_paf=None, use_uncert=None,use_rsa=False,use_d2conv3d=False,use_csa=False,use_glsa=True,use_dw=False):
        super(ResidualUNet3D, self).__init__()
        print('f_maps:',f_maps)

        norm = args.norm
        act = args.act
        use_lw = args.use_lw
        lw_kernel = args.lw_kernel
        self.pif_sigmoid = args.pif_sigmoid
        self.paf_sigmoid = args.paf_sigmoid
        self.use_tanh = args.use_tanh
        self.use_IP =True
        self.out_channels = out_channels
        self.csa=use_csa
        self.dw=use_dw
        self.use_glsa=use_glsa
        if self.out_channels > 1:
            self.use_softmax = args.use_softmax
        else:
            self.use_sigmoid = args.use_sigmoid
        self.use_coord = False

        self.use_rsa = use_rsa
        self.use_softpool = args.use_softpool

        self.use_paf = use_paf
        self.use_uncert = use_uncert

        if self.use_softpool:
            # pool_layer = SoftPool3d
            pass
        else:
            pool_layer = nn.AvgPool3d

        if self.use_IP:
            pools = []
            for _ in range(len(f_maps) - 1):
                pools.append(pool_layer(2))
            self.pools = nn.ModuleList(pools)

        if self.use_glsa:
            self.glsa_block=GLSA3d(f_maps[-1],f_maps[-1])
        if self.use_rsa:
            self.rsablock1=rsaBlock(f_maps[-1])

        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, use_IP=False,
                                  use_coord=self.use_coord,
                                  pool_layer=pool_layer, norm=norm, act=act,
                                  use_lw=use_lw, lw_kernel=lw_kernel,use_d2conv3d=use_d2conv3d,use_rsa=False,use_dw=use_dw)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, use_IP=self.use_IP, use_coord=self.use_coord,
                                  pool_layer=pool_layer, norm=norm, act=act,
                                  use_lw=use_lw, lw_kernel=lw_kernel,use_d2conv3d=use_d2conv3d,use_rsa=False,use_dw=use_dw)

            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)
        if self.csa:
          self.attentions = nn.ModuleList([
              CSA(f_maps[2],f_maps[3],f_maps[0],4),
              CSA(f_maps[1],f_maps[2],f_maps[0],2),
              None
        ])


        self.se_loss = args.use_se_loss
        if self.se_loss:
            self.avgpool = nn.AdaptiveAvgPool3d(1)
            self.fc1 = nn.Linear(f_maps[-1], f_maps[-1])
            self.fc2 = nn.Linear(f_maps[-1], out_channels)

        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i]
            out_feature_num = reversed_f_maps[i + 1]
            # TODO: if non-standard pooling was used, make sure to use correct striding for transpose conv
            # currently strides with a constant stride: (2, 2, 2)
            decoder = Decoder(in_feature_num, out_feature_num, use_coord=self.use_coord, norm=norm, act=act,
                               use_lw=use_lw, lw_kernel=lw_kernel,use_d2conv3d=use_d2conv3d,use_rsa=False,use_csa=self.csa)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        if args.final_double:
            self.final_conv = nn.Sequential(
                nn.Conv3d(f_maps[0], f_maps[0] // 2, kernel_size=1),
                nn.Conv3d(f_maps[0] // 2, out_channels, 1)
            )
            if self.use_paf:
                self.paf_conv = nn.Sequential(
                    nn.Conv3d(f_maps[0], f_maps[0] // 2, kernel_size=1),
                    nn.Conv3d(f_maps[0] // 2, 1, 1)
                )
            self.dropout = nn.Dropout3d
        else:
            self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
            if self.use_paf:
                self.paf_conv = nn.Conv3d(f_maps[0], 1, 1)
            self.dropout = nn.Dropout3d

        if self.use_paf:
            if self.use_uncert:
                self.logsigma = nn.Parameter(torch.FloatTensor([0.5] * 2))
            else:
                self.logsigma = torch.FloatTensor([0.5] * 2)

    def forward(self, x):
        if self.use_IP:
            img_pyramid = []
            img_d = x
            for pool in self.pools:
                img_d = pool(img_d)

                img_pyramid.append(img_d)



        encoders_features = []
        for idx, encoder in enumerate(self.encoders):

            if self.use_IP and idx > 0:
                if idx ==3 and self.use_glsa:
                    x = encoder(x, img_pyramid[idx - 1])
                    x=self.glsa_block(x)
                else:
                   x = encoder(x, img_pyramid[idx - 1])
            else:
                if idx ==3 and self.use_glsa:
                    x = encoder(x)
                    x=self.glsa_block(x)
                else:
                    x = encoder(x)
            encoders_features.insert(0, x)


        first_layer_feature = encoders_features[-1]
        encoders_feature = encoders_features[1:]

        if self.se_loss:
            se_out = self.avgpool(x)
            se_out = se_out.view(se_out.size(0), -1)
            se_out = self.fc1(se_out)
            se_out = self.fc2(se_out)
        if self.csa:
          for decoder, attention, encoder_feature in zip(self.decoders, self.attentions, encoders_feature):
            if attention:
                features_after_att = attention(encoder_feature, x, first_layer_feature)
            else:    # no attention opr in first layer
                features_after_att = first_layer_feature
            x = decoder(features_after_att, x)
        else:
          for  decoder, encoder_features in zip(self.decoders, encoders_feature):
              x = decoder(encoder_features, x)


        out = self.final_conv(x)
        if self.out_channels > 1:
            if self.use_softmax:
                out = torch.softmax(out, dim=1)
            elif self.pif_sigmoid:
                out = torch.sigmoid(out)
            elif self.use_tanh:
                out = torch.tanh(out)
        else:
            if self.use_sigmoid:
                out = torch.sigmoid(out)
            elif self.use_tanh:
                out = torch.tanh(out)

        if self.use_paf:
            paf_out = self.paf_conv(x)
            if self.paf_sigmoid:
                paf_out = torch.sigmoid(paf_out)

        if self.se_loss:#使用了SE损失
            return [out, se_out]
        else:
            if self.use_paf:#使用了残差注意力图
                return [out, paf_out, self.logsigma]
            else:
                return out


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, apply_pooling=True, use_IP=False, use_coord=False,
                 pool_layer=nn.MaxPool3d, norm='bn', act='relu',
                 use_lw=False, lw_kernel=3, input_channels=1, use_d2conv3d=False, use_rsa=False, use_dw=False,use_dmds=True):
        super(Encoder, self).__init__()
        if apply_pooling:
            self.pooling = pool_layer(kernel_size=2)
            if use_dmds:
                self.pooling=DMDS3D_ECA(in_channels=in_channels, out_channels=in_channels, k_size=3)

        else:
            self.pooling = None

        self.use_IP = use_IP
        self.use_coord = use_coord
        self.use_dw=use_dw
        inplaces = in_channels + input_channels if self.use_IP else in_channels
        inplaces = inplaces + 3 if self.use_coord else inplaces

        if use_lw:
            self.basic_module = ExtResNetBlock_lightWeight(inplaces, out_channels, lw_kernel=lw_kernel)
        elif use_dw:
            self.basic_module = DWExtResNetBlock(inplaces, out_channels, norm=norm, act=act,use_d2conv3d=use_d2conv3d,use_rsa=use_rsa)
        else:
            self.basic_module = ExtResNetBlock(inplaces, out_channels, norm=norm, act=act,
                                                     use_d2conv3d=use_d2conv3d, use_rsa=use_rsa)

    def forward(self, x, scaled_img=None):
        if self.pooling is not None:
            x = self.pooling(x)
        if self.use_IP:
            x = torch.cat([x, scaled_img], dim=1)
        if self.use_coord:
            x = self.coord_conv(x)

        x = self.basic_module(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=(2, 2, 2), mode='nearest',
                 padding=1, use_coord=False, norm='bn', act='relu',
                 use_lw=False, lw_kernel=3,use_d2conv3d=False,use_rsa=False,use_csa=False):
        super(Decoder, self).__init__()
        self.use_coord = use_coord
        self.use_d2conv3d=use_d2conv3d
        self.use_csa=use_csa;

        self.upsampling = Upsampling(transposed_conv=True, in_channels=in_channels, out_channels=out_channels,
                                     scale_factor=scale_factor, mode=mode)
        # sum joining
        self.joining = partial(self._joining, concat=self.use_csa)
        if(self.use_csa):
            in_channels = out_channels * 2 + 3 if self.use_coord else out_channels * 2
        else:
            in_channels = out_channels  + 3 if self.use_coord else out_channels


        if use_lw:
            self.basic_module = ExtResNetBlock_lightWeight(in_channels, out_channels, lw_kernel=lw_kernel)
        else:
            self.basic_module = ExtResNetBlock(in_channels, out_channels, norm='bn',act=act,use_d2conv3d=self.use_d2conv3d,use_rsa=use_rsa)

    def forward(self, encoder_features, x, ReturnInput=False):
        x = self.upsampling(encoder_features, x)
        #print(x.shape);
        x = self.joining(encoder_features, x)
        #print(x.shape)
        if self.use_coord:
            x = self.coord_conv(x)
            #print(x.shape)
        if ReturnInput:
            x1 = self.basic_module(x)
            return x1, x
        x = self.basic_module(x)
        return x

    @staticmethod
    def _joining(encoder_features, x, concat):
        if concat:
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features + x


class ExtResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm='bn', act='relu',use_d2conv3d=False,use_rsa=False):
        super(ExtResNetBlock, self).__init__()
        # first convolution
        self.conv1 = SingleConv(in_channels, out_channels, norm=norm, act=act,use_d2conv3d=use_d2conv3d)
        # residual block
        self.conv2 = SingleConv(out_channels, out_channels, norm=norm, act=act,use_d2conv3d=use_d2conv3d)
        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        self.conv3 = SingleConv(out_channels, out_channels, norm=norm, act=act,use_d2conv3d=use_d2conv3d)
        self.use_rsa=use_rsa
        if self.use_rsa:
          self.rsa=rsaBlock(out_channels)
        self.non_linearity = nn.ELU(inplace=False)

    def forward(self, x):
        # apply first convolution and save the output as a residual
        out = self.conv1(x)
        residual = out
        # residual block
        out = self.conv2(out)
        out = self.conv3(out)

        out += residual
        out = self.non_linearity(out)
        if self.use_rsa:
          out=self.rsa(out)
        return out
class DWExtResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm='bn', act='relu',use_d2conv3d=False,use_rsa=False):
        super(DWExtResNetBlock, self).__init__()
        # # first convolution
        # self.conv1 = SingleConv(in_channels, out_channels, norm=norm, act=act,use_d2conv3d=use_d2conv3d)
        # # residual block
        # self.conv2 = SingleConv(out_channels, out_channels, norm=norm, act=act,use_d2conv3d=use_d2conv3d)
        # # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        # self.conv3 = SingleConv(out_channels, out_channels, norm=norm, act=act,use_d2conv3d=use_d2conv3d)
        # self.use_rsa=use_rsa
        # if self.use_rsa:
        #   self.rsa=rsaBlock(out_channels)

        self.conv1=SeparableConv3D(in_channels,out_channels,kernel_size=3, padding=1)
        self.conv2=SeparableConv3D(out_channels,out_channels,kernel_size=3, padding=1)
        self.conv3=SeparableConv3D(out_channels,out_channels,kernel_size=3, padding=1)
        self.non_linearity = nn.ELU(inplace=False)

    def forward(self, x):
        # apply first convolution and save the output as a residual
        out = self.conv1(x)
        residual = out
        # residual block
        out = self.conv2(out)
        out = self.conv3(out)

        out += residual
        out = self.non_linearity(out)
        return out


class SingleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, norm='bn', act='relu',use_d2conv3d=False):
        super(SingleConv, self).__init__()
        if use_d2conv3d:
            # print(use_d2conv3d)
            self.add_module('d2conv3d',XYZSizeConditionedDConv3d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            self.add_module('conv', nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
        self.add_module('batchnorm', normalization(out_channels, norm=norm))
        if act == 'relu':
            self.add_module('relu', nn.ReLU(inplace=False))
        elif act == 'lrelu':
            self.add_module('lrelu', nn.LeakyReLU(negative_slope=0.1, inplace=False))
        elif act == 'elu':
            self.add_module('elu', nn.ELU(inplace=False))
        elif act == 'gelu':
            self.add_module('elu', nn.GELU(inplace=False))


class ExtResNetBlock_lightWeight(nn.Module):
    def __init__(self, in_channels, out_channels, lw_kernel=3):
        super(ExtResNetBlock_lightWeight, self).__init__()
        # first convolution
        self.conv1 = SingleConv_lightWeight(in_channels, out_channels, lw_kernel=lw_kernel)
        # residual block
        self.conv2 = SingleConv_lightWeight(out_channels, out_channels, lw_kernel=lw_kernel)
        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        self.conv3 = SingleConv_lightWeight(out_channels, out_channels, lw_kernel=lw_kernel)
        self.non_linearity = nn.ELU(inplace=False)

    def forward(self, x):
        # apply first convolution and save the output as a residual
        out = self.conv1(x)
        residual = out
        # residual block
        out = self.conv2(out)
        out = self.conv3(out)

        out += residual
        out = self.non_linearity(out)
        return out


class SingleConv_lightWeight(nn.Sequential):
    def __init__(self, in_channels, out_channels, lw_kernel=3, layer_scale_init_value=1e-6):
        super(SingleConv_lightWeight, self).__init__()

        self.dwconv = nn.Conv3d(in_channels, in_channels, kernel_size=lw_kernel, padding=lw_kernel//2, groups=in_channels)
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)
        self.pwconv1 = nn.Linear(in_channels, 2 * in_channels)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(2 * in_channels, out_channels)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_channels)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, 1)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)  # (N, C, D, H, W) -> (N, D, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)  # (N, H, W, C) -> (N, C, H, W)

        x = x + (input if self.in_channels == self.out_channels else self.skip(input))
        return x


class Upsampling(nn.Module):
    def __init__(self, transposed_conv, in_channels=None, out_channels=None, scale_factor=(2, 2, 2), mode='nearest'):
        super(Upsampling, self).__init__()

        if transposed_conv:
            # make sure that the output size reverses the MaxPool3d from the corresponding encoder
            # (D_out = (D_in − 1) ×  stride[0] − 2 ×  padding[0] +  kernel_size[0] +  output_padding[0])
            self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=scale_factor, padding=1)
        else:
            self.upsample = partial(self._interpolate, mode=mode)

    def forward(self, encoder_features, x):
        output_size = encoder_features.size()[2:]
        return self.upsample(x, output_size)

    @staticmethod
    def _interpolate(x, size, mode):
        return F.interpolate(x, size=size, mode=mode)




class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Training for 3D U-Net models')
    parser.add_argument('--use_IP', type=bool, help='whether use image pyramid', default=True)
    parser.add_argument('--use_DS', type=bool, help='whether use deep supervision', default=False)
    parser.add_argument('--use_Res', type=bool, help='whether use residual connectivity', default=False)
    parser.add_argument('--use_bg', type=bool, help='whether use batch generator', default=False)
    parser.add_argument('--use_coord', type=bool, help='whether use coord conv', default=True)
    parser.add_argument('--use_rsa', type=bool, help='whether use rsaBlock3d', default=False)
    parser.add_argument('--use_softmax', type=bool, help='whether use softmax', default=False)
    parser.add_argument('--use_softpool', type=bool, help='whether use softpool', default=False)
    parser.add_argument('--use_se_loss', type=bool, help='whether use aspp', default=False)
    parser.add_argument('--pif_sigmoid', type=bool, help='whether use aspp', default=False)
    parser.add_argument('--paf_sigmoid', type=bool, help='whether use aspp', default=False)
    parser.add_argument('--final_double', type=bool, help='whether use aspp', default=False)
    parser.add_argument('--use_tanh', type=bool, help='whether use aspp', default=False)
    parser.add_argument('--norm', help='type of normalization', type=str, default='bn',
                        choices=['bn', 'gn', 'in', 'sync_bn'])
    parser.add_argument('--use_lw', type=bool, help='whether use lightweight', default=False)
    parser.add_argument('--lw_kernel', type=int, default=5)
    parser.add_argument('--act', help='type of activation function', type=str, default='relu',
                        choices=['relu', 'lrelu', 'elu', 'gelu'])
    args = parser.parse_args()

    net = ResidualUNet3D(args=args,f_maps=[24, 48, 72, 108],use_rsa=False,use_d2conv3d=False,use_csa=False).cuda()
    print(net)

    # conv = SplAtConv3d(64, 32, 3)
    # print(conv)
    data = torch.rand([2, 1, 56, 56, 56]).cuda()
    print(data.shape)
    out = net(data)
    print('_'*100)
    # print(out)
