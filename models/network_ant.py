# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange 
from einops.layers.torch import Rearrange, Reduce
from timm.models.layers import trunc_normal_, DropPath
import torch.nn.functional as F
from timm.models.layers import to_2tuple, trunc_normal_
from collections import OrderedDict

def model_size_summary(model: nn.Module, verbose: bool = True) -> str:
    """
    Return model summary as a string. Optionally print it to standard output.

    Args:
        model (nn.Module): PyTorch model to summarize.
        verbose (bool): If True, print the summary to standard output. Defaults to True.

    Returns:
        str: The model summary as a string.
    """

    def count_params(module: nn.Module, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))

    total_params = count_params(model, trainable_only=False)
    trainable_params = count_params(model, trainable_only=True)

    named_children = list(model.named_children())
    if not named_children:
        named_children = [("No submodules", None)]

    name_max_len = max(len(n) for n, _ in named_children)
    idx_width = len(str(len(named_children)))
    header = f'{"Idx".ljust(idx_width)} | {"Name".ljust(name_max_len)} | Params (ratio %)'

    lines = []
    lines.append(f"Model summary: {getattr(model, 'name_or_path', '')}")
    lines.append(header)
    lines.append("-" * len(header))

    for i, (child_name, child_module) in enumerate(named_children):
        if child_module is None:
            child_params = 0
        else:
            child_params = count_params(child_module, trainable_only=False)

        ratio = (child_params / total_params * 100) if total_params else 0
        lines.append(
            f"{str(i).ljust(idx_width)} | " f"{child_name.ljust(name_max_len)} | " f"{child_params:>11,} ({ratio:.2f}%)"
        )

    lines.append("-" * len(header))
    lines.append(f"Trainable params     : {trainable_params:,}")
    lines.append(f"Non-trainable params : {total_params - trainable_params:,}")
    lines.append(f"Total params         : {total_params:,}")

    # memory footprint
    if hasattr(model, "get_memory_footprint"):
        size_in_bytes = model.get_memory_footprint()
        device = getattr(model, "device", "unknown")
        lines.append(f"Memory footprint     : {size_in_bytes / 10**6:,.2f} MB (device={device})")
    else:
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
        size_in_mb = (param_size + buffer_size) / 10**6
        device = getattr(model, "device", next(model.parameters()).device)
        lines.append(f"Memory footprint     : {size_in_mb:,.2f} MB (device={device})")

    dtype = getattr(model, "dtype", next(model.parameters()).dtype)
    lines.append(f"Model dtype          : {dtype}")

    summary = "\n".join(lines)

    if verbose:
        print(summary)

    return summary



class WMSA(nn.Module):
    """ Self-attention module in Swin Transformer
    """

    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim 
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim//head_dim
        self.window_size = window_size
        self.type=type
        self.embedding_layer = nn.Linear(self.input_dim, 3*self.input_dim, bias=True)

        # TODO recover
        # self.relative_position_params = nn.Parameter(torch.zeros(self.n_heads, 2 * window_size - 1, 2 * window_size -1))
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size -1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1,2).transpose(0,1))

    def generate_mask(self, h, w, p, shift):
        """ generating the mask of SW-MSA
        Args:
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: should be (1 1 w p p),
        """
        # supporting sqaure.
        attn_mask = torch.zeros(h, w, p, p, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(attn_mask, 'w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)')
        return attn_mask

    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True; 
        Returns:
            output: tensor shape [b h w c]
        """
        if self.type!='W': x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window_size//2)), dims=(1,2))
        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window_size)
        h_windows = x.size(1)
        w_windows = x.size(2)
        # sqaure validation
        # assert h_windows == w_windows

        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window_size)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        # Adding learnable relative embedding
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        # Using Attn Mask to distinguish different subwindows.
        if self.type != 'W':
            attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window_size//2)
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type!='W': output = torch.roll(output, shifts=(self.window_size//2, self.window_size//2), dims=(1,2))
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window_size)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size -1
        # negative is allowed
        return self.relative_position_params[:, relation[:,:,0].long(), relation[:,:,1].long()]


class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer Block
        """
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        if input_resolution <= window_size:
            self.type = 'W'

        print("Block Initial Type: {}, drop_path_rate:{:.6f}".format(self.type, drop_path))
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.ln2 = nn.LayerNorm(input_dim)
        # self.mlp = nn.Sequential(
        #     nn.Linear(input_dim, 4 * input_dim),
        #     nn.GELU(),
        #     nn.Linear(4 * input_dim, output_dim),
        # )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        # x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

# --------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# --------------------------------------------
def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CBR',
         negative_slope=0.2):
    L = []
    for t in mode:
        if t == 'C':
            L.append(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                          padding=padding, bias=bias))
        elif t == 'T':
            L.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, bias=bias))
        elif t == 'B':
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == 'I':
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == 'R':
            L.append(nn.ReLU(inplace=True))
        elif t == 'r':
            L.append(nn.ReLU(inplace=False))
        elif t == 'L':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == 'l':
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        elif t == '2':
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == '3':
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == '4':
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == 'U':
            L.append(nn.Upsample(scale_factor=2, mode='nearest'))
        elif t == 'u':
            L.append(nn.Upsample(scale_factor=3, mode='nearest'))
        elif t == 'v':
            L.append(nn.Upsample(scale_factor=4, mode='nearest'))
        elif t == 'M':
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == 'A':
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError('Undefined type: '.format(t))
    return sequential(*L)

class ResBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC',
                 negative_slope=0.2):
        super(ResBlock, self).__init__()

        assert in_channels == out_channels, 'Only support in_channels==out_channels.'
        if mode[0] in ['R', 'L']:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(in_channels, out_channels, kernel_size, stride, padding, bias, mode, negative_slope)

    def forward(self, x):
        res = self.res(x)
        return x + res


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvTransBlock(nn.Module):
    def __init__(self, conv_dim, trans_dim, head_dim, window_size, drop_path, type='W', input_resolution=None):
        """ SwinTransformer and Conv Block
        """
        super(ConvTransBlock, self).__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        
        self.head_dim = head_dim
        self.window_size = window_size
        self.drop_path = drop_path
        self.type = type
        self.input_resolution = input_resolution
        
        assert self.type in ['W', 'SW']
        if self.input_resolution <= self.window_size:
            self.type = 'W'


        self.conv1_1 = nn.Conv2d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)

        self.mlp_block = Mlp(in_features=self.conv_dim, hidden_features=4*self.conv_dim, out_features=self.conv_dim, act_layer=nn.SiLU, drop=0.)

        self.trans_block = Block(self.trans_dim, self.trans_dim, self.head_dim, self.window_size, self.drop_path, self.type, self.input_resolution)

        self.mlp2trans = nn.Conv2d(self.conv_dim, self.trans_dim, kernel_size=3, padding=1, stride=1, bias=True)
        self.trans2mlp = nn.Conv2d(self.trans_dim, self.conv_dim, kernel_size=3, padding=1, stride=1, bias=True)

        self.layer_norm = nn.LayerNorm(self.conv_dim+self.trans_dim)
        self.conv2_1 = nn.Conv2d(self.conv_dim+self.trans_dim, 2 * (self.conv_dim+self.trans_dim), 1, 1, 0, bias=True)
        self.sg = SimpleGate()
        self.conv2_2 = nn.Conv2d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, 1, 1, 0, bias=True)


    def forward(self, x):
        mlp_x0, trans_x0 = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)

        mlp_x1 = self.mlp_block(mlp_x0)

        trans_x1 = Rearrange('b c h w -> b h w c')(trans_x0)
        trans_x1 = self.trans_block(trans_x1)
        trans_x1 = Rearrange('b h w c -> b c h w')(trans_x1)

        conv_trans_x = self.mlp2trans(mlp_x1)
        trans_conv_x = self.trans2mlp(trans_x1)

        mlp_x2 = mlp_x1 * trans_conv_x + mlp_x0
        trans_x2 = trans_x1 * conv_trans_x + trans_x0

        y = torch.cat((mlp_x2, trans_x2), dim=1)

        w = self.layer_norm(y.permute(0,2,3,1)).permute(0,3,1,2)
        w = self.conv2_1(w)
        w = self.sg(w)
        w = self.conv2_2(w)

        z = y + w
        return z


class ZeroConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
    def forward(self, x):
        return self.conv(x)


class Noise_T_Predictor(nn.Module):

    def __init__(self, in_nc=3, out_nc=3, nc_config=[2,2,2,2,2,2,2], t_config=[2,2,2,2], dim=64, drop_path_rate=0.0, input_resolution=128):
        super(Noise_T_Predictor, self).__init__()
        self.dim = dim
        self.head_dim = dim//2
        self.window_size = 8

        # -----------------------------------------
        # Noise Compensation
        # -----------------------------------------
        self.nc_head1 = [nn.Conv2d(in_nc+1, dim, 3, 1, 1, bias=False)]

        begin = 0
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(nc_config))]
        self.nc_down1 = [ConvTransBlock(dim//2, dim//2, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution) 
                      for i in range(nc_config[0])] + \
                      [nn.Conv2d(dim, 2*dim, 2, 2, 0, bias=False)]

        begin += nc_config[0]
        self.nc_down2 = [ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//2)
                      for i in range(nc_config[1])] + \
                      [nn.Conv2d(2*dim, 4*dim, 2, 2, 0, bias=False)]

        begin += nc_config[1]
        self.nc_down3 = [ConvTransBlock(2*dim, 2*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW',input_resolution//4)
                      for i in range(nc_config[2])] + \
                      [nn.Conv2d(4*dim, 8*dim, 2, 2, 0, bias=False)]

        begin += nc_config[2]
        self.nc_body = [ConvTransBlock(4*dim, 4*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//8)
            for i in range(nc_config[3])]
        self.m_body = [ConvTransBlock(4*dim, 4*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//8)
                    for i in range(nc_config[3])]

        begin += nc_config[3]
        self.nc_up3 = [nn.ConvTranspose2d(8*dim, 4*dim, 2, 2, 0, bias=False),] + \
                      [ConvTransBlock(2*dim, 2*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW',input_resolution//4)
                      for i in range(nc_config[4])]

        begin += nc_config[4]
        self.nc_up2 = [nn.ConvTranspose2d(4*dim, 2*dim, 2, 2, 0, bias=False),] + \
                      [ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//2)
                      for i in range(nc_config[5])]
                      
        begin += nc_config[5]
        self.nc_up1 = [nn.ConvTranspose2d(2*dim, dim, 2, 2, 0, bias=False),] + \
                    [ConvTransBlock(dim//2, dim//2, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution) 
                      for i in range(nc_config[6])]

        # -----------------------------------------
        # Timpstep Prediction
        # -----------------------------------------
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(t_config))]
        begin = 0
        self.m_head2 = [nn.Conv2d(in_nc, dim, 3, 1, 1, bias=False)]
        self.m_down1 = [ConvTransBlock(dim//2, dim//2, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution) 
                      for i in range(t_config[0])] + \
                      [nn.Conv2d(dim, 2*dim, 2, 2, 0, bias=False)]

        begin += t_config[0]
        self.m_down2 = [ConvTransBlock(dim, dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//2)
                      for i in range(t_config[1])] + \
                      [nn.Conv2d(2*dim, 4*dim, 2, 2, 0, bias=False)]

        begin += t_config[1]
        self.m_down3 = [ConvTransBlock(2*dim, 2*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW',input_resolution//4)
                      for i in range(t_config[2])] + \
                      [nn.Conv2d(4*dim, 8*dim, 2, 2, 0, bias=False)]

        begin += t_config[2]
        self.m_body = [ConvTransBlock(4*dim, 4*dim, self.head_dim, self.window_size, dpr[i+begin], 'W' if not i%2 else 'SW', input_resolution//8)
                    for i in range(t_config[3])]

        # Noise Compensation
        self.nc_head1 = nn.Sequential(*self.nc_head1)
        self.nc_down1 = nn.Sequential(*self.nc_down1)
        self.nc_down2 = nn.Sequential(*self.nc_down2)
        self.nc_down3 = nn.Sequential(*self.nc_down3)
        self.nc_body = nn.Sequential(*self.nc_body)
        self.nc_up3 = nn.Sequential(*self.nc_up3)
        self.nc_up2 = nn.Sequential(*self.nc_up2)
        self.nc_up1 = nn.Sequential(*self.nc_up1)
        self.zero_conv = ZeroConv(dim, 3)

        # Timpstep Prediction
        self.m_head2 = nn.Sequential(*self.m_head2)
        self.m_down1 = nn.Sequential(*self.m_down1)
        self.m_down2 = nn.Sequential(*self.m_down2)
        self.m_down3 = nn.Sequential(*self.m_down3)
        self.m_body = nn.Sequential(*self.m_body)
        self.t_pred = nn.Sequential(ResBlock(8*dim, 8*dim, bias=True, mode='C' + 'R' + 'C'),
                                  torch.nn.AdaptiveAvgPool2d((1, 1)),
                                  torch.nn.Flatten(),
                                  torch.nn.Linear(8*dim, 8*dim),
                                  nn.ReLU(),
                                  torch.nn.Linear(8*dim, 8*dim),
                                  nn.ReLU(),
                                  torch.nn.Linear(8*dim, 1)
                                  )




    def forward(self, x0, t_factor=None):

        h, w = x0.size()[-2:]
        paddingBottom = int(np.ceil(h/64)*64-h)
        paddingRight = int(np.ceil(w/64)*64-w)
        x0 = F.pad(x0, (0, paddingRight, 0, paddingBottom), mode='reflect')  # left, right, up, down
        
        # Timpstep Prediction
        x1 = self.m_head2(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        t_pred = self.t_pred(x)

        # Noise Compensation
        t_pred_expand = t_pred.unsqueeze(-1).unsqueeze(-1)
        t_pred_expand = t_pred_expand.expand(-1, 1, h + paddingBottom, w + paddingRight)

        nc_1 = self.nc_head1(torch.cat((x0, t_pred_expand), dim=1))
        
        nc_2 = self.nc_down1(nc_1)
        nc_3 = self.nc_down2(nc_2)
        nc_4 = self.nc_down3(nc_3)
        nc = self.nc_body(nc_4)
        nc = self.nc_up3(nc+nc_4)
        nc = self.nc_up2(nc+nc_3)
        nc = self.nc_up1(nc+nc_2)

        noise = self.zero_conv(nc+nc_1)

        x_s = x0 + noise

        return x_s[..., :h, :w], t_pred


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    

if __name__ == '__main__':
    # Optional profiling dependency; not required for training or evaluation.
    from ptflops import get_model_complexity_info

    net = Noise_T_Predictor(nc_config=[2,2,2,2,2,2,2],t_config=[2,2,2,1], dim=32)

    model_size_summary(net)

    input_x = torch.randn((1, 3, 256, 256))  # Batch size 1, C=3, H=256, W=256

    output, noise_maps = net(input_x)

    input_shape = (3, 256, 256)  # (C, H, W)

    # macs, parameters
    macs, params = get_model_complexity_info(net, input_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4]) 
    print(f"MACs (G): {macs}, Parameters (M): {params}")

