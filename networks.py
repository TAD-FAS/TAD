# -*- coding: utf-8 -*-

"""
Created on 2021/10/3 18:55
@author: Acer
@description: 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


##################################################################################
# ---------------------------------Discriminator----------------------------------
##################################################################################


class MsDiscriminator(nn.Module):
    '''
    多尺度鉴别器是Pix2PixHD提出来的,大概就是将要判别的图像下采样成三种不同的尺寸在进行区分
    '''
    def __init__(self, in_channels, params, sn=False):
        '''
        n_scale: 鉴别器划分为几种尺度
        sn:是否使用spectral_norm
        '''
        super(MsDiscriminator, self).__init__()
        num_features = params['num_features']
        num_scale = params['num_scale']
        padding_mode = params['padding_mode']
        norm = params['norm']
        act = params['act']
        n_layer = params['n_layer']

        self.down_sample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
        self.Diss_bone = nn.ModuleList()  # 存储鉴别器的主干部分
        for _ in range(num_scale):
            self.Diss_bone.append(self._make_net(in_channels, num_features, padding_mode, norm, act, n_layer, sn))
        
        if sn:
            # self.dis_ls = spectral_norm(nn.Conv2d(num_features*2**(n_layer-1), 1, 1, 1, 0))
            self.dis_rs = spectral_norm(nn.Conv2d(num_features*2**(n_layer-1), 1, 1, 1, 0))
        else:
            # self.dis_ls = nn.Conv2d(num_features*2**(n_layer-1), 1, 1, 1, 0)  # 区分live or spoof
            self.dis_rs = nn.Conv2d(num_features*2**(n_layer-1), 1, 1, 1, 0)  # 区分real or synthetic

    def _make_net(self, in_channels, num_features, padding_mode, norm, act, n_layer, sn):
        model = []
        model += [ConvBlock(in_channels, num_features, 4, 2, 1,
                            padding_mode=padding_mode, norm=norm, act=act, sn=sn)]
        for i in range(1, n_layer):
            model += [ConvBlock(num_features, num_features*2, 4, 2, 1,
                                padding_mode=padding_mode, norm=norm, act=act, sn=sn)]
            num_features *= 2
        return nn.Sequential(*model)

    def forward(self, x):
        dis_rs_outs=[]
        for Dis in self.Diss_bone:
            disbone_out = Dis(x)
            out_rs = self.dis_rs(disbone_out)
            dis_rs_outs.append(out_rs)
            x = self.down_sample(x)
        return dis_rs_outs


class Dis_content(nn.Module):
    def __init__(self, in_channels):
        super(Dis_content, self).__init__()
        model = []
        model += [ConvBlock(in_channels, in_channels, 3, 2, 1, norm='none')] 
        model += [ConvBlock(in_channels, in_channels, 3, 2, 1, norm='none')]
        model += [ConvBlock(in_channels, in_channels, 3, 2, 1, norm='none')]
        model += [ConvBlock(in_channels, in_channels, 3, 2, 1, norm='none')]
        model += [nn.AdaptiveAvgPool2d(1)]
        model += [nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        out = out.view(-1)
        return out

##################################################################################
# -----------------------------------Generator------------------------------------
##################################################################################

class Generator(nn.Module):
    def __init__(self, in_channels, params):
        super(Generator, self).__init__()        
        self.enc = Encoder(in_channels=in_channels, norm='bn', params=params)  
        self.dec = Decoder(self.enc.output_dim, 3, norm='bn', params=params)


    def forward(self, image):
        latent = self.enc(image)
        # 生成live face和spoof trace
        gen_limage = self.dec(latent)
        return gen_limage, latent



##################################################################################
# --------------------------------------Encoder-----------------------------------
##################################################################################
class Encoder(nn.Module):
    def __init__(self, in_channels, norm, params):
        super(Encoder, self).__init__()
        num_features = params['num_features']
        # num_downsample = params['num_downsample']
        num_res = params['num_res']
        act = params['act']
        padding_mode = params['padding_mode']

        self.conv = ConvBlock(in_channels, 64, 7, 1, 3, padding_mode=padding_mode, norm=norm, act=act)
        # branch of encode spoof-irrelevant information
        self.block1 = nn.Sequential(
            ConvBlock(64, 128, 3, 1, 1, padding_mode=padding_mode, norm=norm, act=act),
            ConvBlock(128, 196, 3, 1, 1, padding_mode=padding_mode, norm=norm, act=act),
            ConvBlock(196, 128, 3, 1, 1, padding_mode=padding_mode, norm=norm, act=act), 
            nn.MaxPool2d(2, 2),  # downsample 128x128x128
        )
        self.block2 = nn.Sequential(
            ConvBlock(128, 128, 3, 1, 1, padding_mode=padding_mode, norm=norm, act=act),
            ConvBlock(128, 196, 3, 1, 1, padding_mode=padding_mode, norm=norm, act=act),
            ConvBlock(196, 128, 3, 1, 1, padding_mode=padding_mode, norm=norm, act=act), 
            nn.MaxPool2d(2, 2),  # downsample 128x64x64
        )
        self.block3 = nn.Sequential(
            ConvBlock(128, 128, 3, 1, 1, padding_mode=padding_mode, norm=norm, act=act), 
            ConvBlock(128, 196, 3, 1, 1, padding_mode=padding_mode, norm=norm, act=act),
            ConvBlock(196, 128, 3, 1, 1, padding_mode=padding_mode, norm=norm, act=act), 
            nn.MaxPool2d(2, 2),  # downsample 128x32x32
        )
        
        self.output_dim = 384  # 解码时用到

    def forward(self, x):
        x = self.conv(x)
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        b1 =  F.interpolate(x1, (x3.size(2), x3.size(3)), mode='bilinear', align_corners=True)
        b2 =  F.interpolate(x2, (x3.size(2), x3.size(3)), mode='bilinear', align_corners=True)
        latent = torch.cat((b1, b2, x3), dim=1)
        return latent
    

##################################################################################
# --------------------------------------Decoder-----------------------------------
##################################################################################

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, norm, params):  # in_channels = 384
        super(Decoder, self).__init__()
        # num_unsample = params['num_downsample']
        num_res = params['num_res']
        act = params['act']
        padding_mode = params['padding_mode']
        self.apply_shotcut = params['apply_shotcut']
        self.up1 = nn.Sequential(
            ConvBlock(in_channels, 128, 3, 1, 1, padding_mode=padding_mode, norm=norm, act=act),  # 128x32x32
            ConvBlock(128, 196, 3, 1, 1, padding_mode=padding_mode, norm=norm, act=act),  # 196x32x32
            nn.Upsample(scale_factor=2),
            ConvBlock(196, 128, 5, 1, 2, padding_mode=padding_mode, norm=norm, act=act),  # 128x64x64  
        )
        self.up2 = nn.Sequential(
            ConvBlock(128, 128, 3, 1, 1, padding_mode=padding_mode, norm=norm, act=act),  # 128x64x64
            ConvBlock(128, 196, 3, 1, 1, padding_mode=padding_mode, norm=norm, act=act),  # 196x64x64
            nn.Upsample(scale_factor=2),
            ConvBlock(196, 128, 5, 1, 2, padding_mode=padding_mode, norm=norm, act=act),  # 128x128x128  
        )
        self.up3 = nn.Sequential(
            ConvBlock(128, 128, 3, 1, 1, padding_mode=padding_mode, norm=norm, act=act),  # 128x128x128  
            ConvBlock(128, 196, 3, 1, 1, padding_mode=padding_mode, norm=norm, act=act),  # 196x128x128  
            nn.Upsample(scale_factor=2),
            ConvBlock(196, 128, 5, 1, 2, padding_mode=padding_mode, norm=norm, act=act),  # 128x256x256  
        )

        self.out = ConvBlock(128, out_channels, 7, 1, 3, padding_mode=padding_mode, norm='none', act='tanh')

    def forward(self, latent):
        up1 = self.up1(latent)
        up2 = self.up2(up1)
        up3 = self.up3(up2)
        image = self.out(up3)
        return image


class MSM(nn.Module):
    def __init__(self, params):
        super().__init__()
        act = params['act']
        padding_mode = params['padding_mode']
        # depth
        self.depth = nn.Sequential(
            ConvBlock(384, 128, 3, 1, 1, padding_mode=padding_mode, norm='bn', act=act),
            ConvBlock(128, 64, 3, 1, 1, padding_mode=padding_mode, norm='bn', act=act),
            ConvBlock(64, 1, 3, 1, 1, padding_mode=padding_mode, norm='none', act='tanh'),  # 注意使用relu还是tanh
        )
        # patch
        self.patch = nn.Sequential(
            ConvBlock(384, 128, 3, 1, 1, padding_mode=padding_mode, norm='bn', act=act),
            ConvBlock(128, 64, 3, 1, 1, padding_mode=padding_mode, norm='bn', act=act),
            ConvBlock(64, 1, 3, 1, 1, padding_mode=padding_mode, norm='none', act='tanh'),  # 注意使用relu还是tanh
        )
        self.avg = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, latent):
        # patch_feat, depth_feat = torch.split(latent, 192, dim=1)  # 从第一维度划分，每个块channel = 192
        # print('\n',patch_feat.size())
        patch_map = self.patch(latent)
        depth_map = self.depth(latent)
        center_feat = self.avg(latent).view(latent.size(0), -1)
        # print('\n',center_feat)
        return depth_map, patch_map, center_feat


class Classifier(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cls = nn.Sequential(
            LinearBlock(in_channels, 512, 'none', activation='relu'),
            LinearBlock(512, 2, 'none', activation='none'),
        )

    def forward(self, depth_map, patch_map, center_feat):
        depth_feat = depth_map.view(depth_map.size(0), -1)
        patch_feat = patch_map.view(patch_map.size(0), -1)
        x =torch.cat((depth_feat, patch_feat), dim=1)
        x = x.view(x.size(0), -1) 
        x = self.cls(x)   
        return x


##################################################################################
# -----------------------------------BasicBlock-----------------------------------
##################################################################################
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding=0, padding_mode='reflect', act='relu', norm='none', sn=False, apply_dropout=False):
        '''
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param padding_type:
        :param act: active function
        :param norm: normalization
        :param sn: use spectral_norm or not  目前没有实现spectral_norm,实现请参考MUNIT
        '''
        super(ConvBlock, self).__init__()
        norm_dim = out_channels
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'lrelu':
            self.act = nn.LeakyReLU(0.2, inplace=True)
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'none':
            self.act = None
        else:
            assert 0, "Unsupported activation: {}".format(act)
        self.use_bias = True
        if sn:
            self.conv = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode=padding_mode)

        if apply_dropout == True:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class ResBlocks(nn.Module):
    def __init__(self, dim, num_blocks, norm='in', act='relu', padding_mode='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm, act, padding_mode)]

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', act='relu', padding_mode='zero'):
        super(ResBlock, self).__init__()
        model = []
        model += [ConvBlock(dim, dim, 3, 1, 1, norm=norm, act=act, padding_mode=padding_mode)]
        model += [ConvBlock(dim, dim, 3, 1, 1, norm=norm, act='none', padding_mode=padding_mode)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        # out = F.relu(residual+out)  # MUNIT, LIR, DRIT都没有act
        out = residual + out
        return out



class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu', apply_dropout=False):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)
        
        if apply_dropout == True:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = None

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        if self.dropout is not None:
            out = self.dropout(out)
        return out



##################################################################################
# -------------------------------Normalization layers-----------------------------
# copy from MUNIT and LIR
##################################################################################


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

    
####################################################################
# --------------------- Spectral Normalization ---------------------
#  This part of code is copied from pytorch master branch (0.5.0)
####################################################################
class SpectralNorm(object):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """
    def __init__(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(self.dim,
                                            *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        weight_mat = weight_mat.reshape(height, -1)
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                v = F.normalize(torch.matmul(weight_mat.t(), u), dim=0, eps=self.eps)
                u = F.normalize(torch.matmul(weight_mat, v), dim=0, eps=self.eps)
        sigma = torch.dot(u, torch.matmul(weight_mat, v))
        weight = weight / sigma
        return weight, u

    def remove(self, module):
        weight = getattr(module, self.name)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_orig')
        module.register_parameter(self.name, torch.nn.Parameter(weight))

    def __call__(self, module, inputs):
        if module.training:
            weight, u = self.compute_weight(module)
            setattr(module, self.name, weight)
            setattr(module, self.name + '_u', u)
        else:
            r_g = getattr(module, self.name + '_orig').requires_grad
            getattr(module, self.name).detach_().requires_grad_(r_g)

    @staticmethod
    def apply(module, name, n_power_iterations, dim, eps):
        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = module._parameters[name]
        height = weight.size(dim)
        u = F.normalize(weight.new_empty(height).normal_(0, 1), dim=0, eps=fn.eps)
        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        module.register_buffer(fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_forward_pre_hook(fn)
        return fn


def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d,
                               torch.nn.ConvTranspose2d,
                               torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module


def remove_spectral_norm(module, name='weight'):
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module
    raise ValueError("spectral_norm of '{}' not found in {}".format(name, module))



def test():
    # from torch.utils.tensorboard import SummaryWriter
    from utils import get_config
    from torchsummary import summary
    config = get_config('./configs/oulu_npu.yaml')
    print(type(config['gen']['num_features']))
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = Generator(3, config).to(device)
    # # summary(model, input_size=(3, 256, 256))
    # inputs = torch.randn(4, 3, 256, 256).to(device)
    # _, out, _ = model(inputs)
    # print(out)
    # target = torch.ones_like(out)
    # target[1][0] = 0
    # print(target)
    # print(out.size())
    # writer = SummaryWriter('logs')
    # decoder = MsDiscriminator(3, config['dis'], False).to(device)
    # inputs = torch.rand(4, 3, 256, 256).to(device)
    # writer.add_graph(decoder, inputs)
    # writer.close()



if __name__ == '__main__':
    test()









