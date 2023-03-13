
import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthNet(nn.Module):
    def __init__(self, in_channels, params):
        super().__init__()
        act = params['act']
        padding_mode = params['padding_mode']
        self.conv = ConvBlock(in_channels, 64, 7, 1, 3, padding_mode=padding_mode, norm='bn', act=act)
        self.block1 = nn.Sequential(
            ConvBlock(64, 128, 3, 1, 1, padding_mode=padding_mode, norm='bn', act=act),
            ConvBlock(128, 196, 3, 1, 1, padding_mode=padding_mode, norm='bn', act=act),
            ConvBlock(196, 128, 3, 1, 1, padding_mode=padding_mode, norm='bn', act=act),  
            nn.MaxPool2d(2, 2)  # downsample 128x128x128
        )
        self.block2 = nn.Sequential(
            ConvBlock(128, 128, 3, 1, 1, padding_mode=padding_mode, norm='bn', act=act),
            ConvBlock(128, 196, 3, 1, 1, padding_mode=padding_mode, norm='bn', act=act),
            ConvBlock(196, 128, 3, 1, 1, padding_mode=padding_mode, norm='bn', act=act),  
            nn.MaxPool2d(2, 2)  # downsample 128x64x64
        )
        self.block3 = nn.Sequential(
            ConvBlock(128, 128, 3, 1, 1, padding_mode=padding_mode, norm='bn', act=act), 
            ConvBlock(128, 196, 3, 1, 1, padding_mode=padding_mode, norm='bn', act=act),
            ConvBlock(196, 128, 3, 1, 1, padding_mode=padding_mode, norm='bn', act=act),  
            nn.MaxPool2d(2, 2)  # downsample 128x32x32
        )

        self.conv1 = nn.Sequential(
            ConvBlock(384, 128, 3, 1, 1, padding_mode=padding_mode, norm='bn', act=act),
            ConvBlock(128, 64, 3, 1, 1, padding_mode=padding_mode, norm='bn', act=act),
            ConvBlock(64, 1, 3, 1, 1, padding_mode=padding_mode, norm='bn', act='tanh'),  # 注意使用relu还是tanh
        )
        
    def forward(self, x):
        x = self.conv(x)
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        b1 =  F.interpolate(x1, (x3.size(2), x3.size(3)), mode='bilinear', align_corners=True)
        b2 =  F.interpolate(x2, (x3.size(2), x3.size(3)), mode='bilinear', align_corners=True)
        latent = torch.cat((b1, b2, x3), dim=1)
        x = self.conv1(latent)
        return x


##################################################################################
# -----------------------------------BasicBlock-----------------------------------
##################################################################################
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding=0, padding_mode='reflect', act='relu', norm='none', apply_dropout=False):
        '''
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param padding_type:
        :param act: active function
        :param norm: normalization
        :param sn: use spectral_norm or not  目前没有实现spectral_norm，实现请参考MUNIT
        '''
        super(ConvBlock, self).__init__()
        norm_dim = out_channels
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
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