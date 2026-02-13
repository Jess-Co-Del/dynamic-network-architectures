import torch
import torch.nn as nn
from torch import Tensor
from timm.models.layers import trunc_normal_, DropPath
import torch.nn.functional as F


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(
        self,
        dim_resolution,
        normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.dim_resolution = dim_resolution
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
            if self.dim_resolution == 2:  # Add dimensions to match 2D or 3D.
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
            else:
                x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x


class ConvNextEncoderBlock(nn.Module):
    """
    Standard ConvNeXt encoder block taken from 
    "Liu, Zhuang, et al. "A convnet for the 2020s." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022."
    This implementation is based on Layer normalization. Refer to Fig. 2 of our paper for the definitive structure of this block

    Keyword arguments:
    argument -- 
    dim - dimensions of the input tensor
    drop_path - drop value of the DropPath() function. If its <=0 then an identity function is chosen instead
    layer_scale_init_values - scaling values 

    Return: 
     - None
    """
    def __init__(
        self,
        dim_resolution: int,
        dim: int,
        conv_op: nn.Module = nn.Conv2d,
        kernel: int = 7,
        drop_path: float = 0.,
        layer_scale_init_value: float = 1e-6
    ) -> None:
        super(ConvNextEncoderBlock, self).__init__()
        self.dim_resolution = dim_resolution
        self.dwconv = conv_op(dim, dim, kernel_size=kernel, padding=3, groups=dim)  # Depth-wise Convolution
        self.norm = LayerNorm(dim_resolution, dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # Pointwise or 1x1 Conv layer
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)),
            requires_grad = True) if layer_scale_init_value> 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        forward function

        Keyword arguments:
        argument -- x : Input torch.Tensor
        Return: x : Output torch.Tensor of the encoder block
        """

        input_x = x
        x = self.dwconv(x)
        if self.dim_resolution == 2:
            x = x.permute(0, 2, 3, 1)  # Permuting Dimensions --- (B, C, H, W) -> (B, H, W, C)
        else:
            x = x.permute(0, 2, 3, 4, 1)  # Permuting Dimensions --- (B, C, H, W, D) -> (B, H, W, D, C)
        x = self.norm(x)  # Layer Norm with channel dimensions are aligned at the last
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        if self.dim_resolution == 2:
            x = x.permute(0, 3, 1, 2)  # Permuting Dimensions --- (B, H, W, C) -> (B, C, H, W)
        else:
            x = x.permute(0, 4, 1, 2, 3)  # Permuting Dimensions --- (B, H, W, D, C) -> (B, C, H, W, D)

        x = input_x + self.drop_path(x)  # Add the residual connection
        return x


class ConvNextDecoderBlock(nn.Module):
    """
    A straightforward implementation of modified ConvNext blocks used for constructing the decoder network.
    For visualization, refer to Fig. 2. 

    Keyword arguments:
    arguments -- 
    input_dim : Input dimensions of the input torch.Tensor for the first conv 7x7 layer
    output_dim : Output dimensions of all the conv 1x1 layers. It also serves as the input dims of all the
                conv 1x1 layers
    

    Return: None
    """
    
    def __init__(
        self, 
        conv_op: nn.Conv2d,
        input_dim:int,
        output_dim:int,
        stride:int = 1,
        padding:int = 1) -> None:
        super(ConvNextDecoderBlock, self).__init__()
        
        if conv_op == nn.Conv2d:
            norm_op = nn.BatchNorm2d
        else:
            norm_op = nn.BatchNorm3d

        self.mod_convNext_block = nn.Sequential(
            conv_op(input_dim, output_dim, kernel_size = 7, stride = stride, padding = padding, groups = output_dim),
            norm_op(output_dim),
            conv_op(output_dim, output_dim, kernel_size = 1, stride = stride, padding = padding),
            nn.GELU(approximate = 'none'),
            conv_op(output_dim, output_dim, kernel_size = 1, stride = stride, padding = padding)
        )

        self.conv_skip_connection = nn.Sequential(
            conv_op(input_dim, output_dim, kernel_size = 3, stride = stride, padding = padding),
            norm_op(output_dim),
        )

        self.gelu = nn.GELU(approximate = 'none')

    def forward(self, x : Tensor) -> Tensor:
        """forward function
        
        Keyword arguments:
        argument -- x : Input torch.Tensor
        Return:  Output torch.Tensor of the encoder block
        """
        return self.gelu(self.mod_convNext_block(x) + self.conv_skip_connection(x))


class ConvNextEncoder(nn.Module):
    """
    Standard ConvNeXt model taken from 
    "Liu, Zhuang, et al. "A convnet for the 2020s." Proceedings of
    the IEEE/CVF conference on computer vision and pattern recognition. 2022."
    """
    def __init__(
        self,
        input_channels: int = 3, num_classes: int = 1000, dim_resolution: int = 2,
        conv_op: nn.Module = nn.Conv2d, norm_op: nn.Module = LayerNorm,
        depths= [3,3,9,3], dims = [96, 192, 384, 768], drop_path_rate: int = 0.,
        layer_scale_init_value:int  = 1e-6, head_init_scale: int = 1., embed_dim: int = None
        ) -> None:
        

        super(ConvNextEncoder, self).__init__()
        self.embed_dim = embed_dim
        #Stem operation and 3 downsampling layers
        self.downsample_layers = nn.ModuleList() 

        #Stem operation for initial spatial downsampling
        stem_op = nn.Sequential(
            conv_op(input_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm(
                dim_resolution,
                dims[0], eps=1e-6, data_format="channels_first")
        )

        self.downsample_layers.append(stem_op)

        # Staging downsampling layers
        for i in range(len(depths)-1):
            downsample = nn.Sequential(
                LayerNorm(
                    dim_resolution,
                    dims[i], eps = 1e-6, data_format="channels_first"),
                conv_op(dims[i], dims[i+1], kernel_size = 2, stride = 2)
            )
            self.downsample_layers.append(downsample)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        current = 0

        for i in range(len(depths)):
            # Construct each block one by one
            stage = nn.Sequential(
                *[ConvNextEncoderBlock(
                    dim_resolution=dim_resolution, conv_op=conv_op,
                    dim=dims[i], drop_path=dp_rates[current+j],
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )

            self.stages.append(stage)
            current += depths[i]

        # This is the standard layer normalization for the last layer
        self.norm = nn.LayerNorm(dims[-1], eps = 1e-6)
        if embed_dim is not None:
            self.head = nn.Linear(dims[-1], embed_dim)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)

        self.apply(self._init_weights)

    def _init_weights(self, m) -> None:
        if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []

        for dsl, stage in zip(self.downsample_layers, self.stages):
            x = dsl(x)
            x = stage(x)
            skips.append(x)

        if self.embed_dim is not None:
            x = self.head(x)

        return skips


weight_urls = {
    "convnext_tiny" : "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
}


def convnext_build(
    conv_op: nn.Conv3d, input_channels: int,
    model_size: str = 'convnext_tiny', pretrained: bool = False, **kwargs) -> ConvNextEncoder:
    """sumary_line
    
    Keyword arguments:
    argument -- description
    Return: return_description
    """
    dim_resolution = 3 if conv_op == nn.Conv3d else 2

    if model_size == 'convnext_tiny':
        model = ConvNextEncoder(
            conv_op=conv_op, dim_resolution=dim_resolution, input_channels=input_channels,
            depths = [3, 3, 9, 3], dims = [96,192,384,768], **kwargs)

        if pretrained:
            pretrained_weight_url = weight_urls['convnext_tiny']
            checkpoint = torch.hub.load_state_dict_from_url(pretrained_weight_url, map_location = "cpu")
            model.load_state_dict(checkpoint["model"])
        return model
    elif model_size == 'convnext_small':
        model = ConvNextEncoder(
            conv_op=conv_op, dim_resolution=dim_resolution, input_channels=input_channels,
            depths = [3, 3, 27, 3], dims = [96, 192, 384, 768], **kwargs)

        if pretrained:
            pretrained_weight_url = weight_urls['convnext_small']
            checkpoint = torch.hub.load_state_dict_from_url(pretrained_weight_url, map_location = "cpu")
            model.load_state_dict(checkpoint["model"])
        return model
    
    elif model_size == 'convnext_base':
        model = ConvNextEncoder(
            conv_op=conv_op, dim_resolution=dim_resolution, input_channels=input_channels,
            depths = [3, 3, 27, 3], dims = [128, 256, 512, 1024], **kwargs)

        if pretrained:
            pretrained_weight_url = weight_urls['convnext_base']
            checkpoint = torch.hub.load_state_dict_from_url(pretrained_weight_url, map_location = "cpu")
            model.load_state_dict(checkpoint["model"])
        return model
    elif model_size == 'convnext_large':
        model = ConvNextEncoder(
            conv_op=conv_op, dim_resolution=dim_resolution, input_channels=input_channels,
            depths = [3, 3, 27, 3], dims = [192, 384, 768, 1536], **kwargs)

        if pretrained:
            pretrained_weight_url = weight_urls['convnext_large']
            checkpoint = torch.hub.load_state_dict_from_url(pretrained_weight_url, map_location = "cpu")
            model.load_state_dict(checkpoint["model"])
        return model
