from typing import Union, Type
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from ..building_blocks.convnext import convnext_build, ConvNextDecoderBlock
from ..building_blocks.simple_conv_blocks import conv_relu
from ..building_blocks.attention import DualPathResponseFusionAttention
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0


class ResponseFusionAttentionUConvNextTiny(nn.Module):
    def __init__(
      self,
      strides: list,
      conv_op: nn.Conv2d,
      deep_supervision: bool = False,
      features: list = [384, 192, 96],
      pretrained_encoder_backbone: bool = True,
      num_classes: int = 2,
      **kwargs) -> None:
        super(ResponseFusionAttentionUConvNextTiny,self).__init__()
        # Down part of convunext
        self.encoder = convnext_build(
          conv_op=conv_op,
          model_size = 'convnext_tiny',
          pretrained = pretrained_encoder_backbone,
          **kwargs
          )
        self.downsampling_layers = list(self.encoder.downsample_layers.children())
        self.encoder_layers = list(self.encoder.stages.children())
        self.layer0 = nn.Sequential(self.downsampling_layers[0], self.encoder_layers[0])
        self.layer1 = nn.Sequential(self.downsampling_layers[1], self.encoder_layers[1])
        self.layer2 = nn.Sequential(self.downsampling_layers[2], self.encoder_layers[2])

        #bottleneck
        self.bottleneck = nn.Sequential(self.downsampling_layers[3])
        self.bott_1x1 = conv_relu(conv_op, 768,768,1,0)
        
        #Up part of convunext
        self.decoder = ConvNextVisionDecoder(
          conv_op=conv_op,
          features=features,
          out_channels=num_classes,
          deep_supervision=deep_supervision
          )

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        layer0 = self.layer0(input)
        #print(layer0.shape)
        layer1 = self.layer1(layer0)
        #print(layer1.shape)
        layer2 = self.layer2(layer1)
        #print(layer2.shape)

        bottleneck = self.bottleneck(layer2)
        bottleneck = self.bott_1x1(bottleneck)
        #print('bottleneck', bottleneck.shape)

        mask = self.decoder([bottleneck, layer2, layer1, layer0])

        return mask

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)


class _ResponseFusionAttentionUConvNextSmall(nn.Module):
  def __init__(
      self,
      conv_op: nn.Conv3d,
      strides: list,
      deep_supervision: bool = False,
      pretrained_encoder_backbone: bool = False, num_classes: int = 1, **kwargs
  ) -> None:
    super(_ResponseFusionAttentionUConvNextSmall, self).__init__()
    #Down part of convunext
    self.encoder = convnext_build(
      conv_op=conv_op,
      model_size = 'convnext_tiny', pretrained = pretrained_encoder_backbone, **kwargs
    )

    self.downsampling_layers = list(self.encoder.downsample_layers.children())
    self.encoder_layers = list(self.encoder.stages.children())
    self.layer0 = nn.Sequential(self.downsampling_layers[0], self.encoder_layers[0])
    self.layer1 = nn.Sequential(self.downsampling_layers[1], self.encoder_layers[1])
    self.layer2 = nn.Sequential(self.downsampling_layers[2], self.encoder_layers[2])

    self.layer0_1x1 = conv_relu(conv_op, 96,96,1,0)
    self.layer1_1x1 = conv_relu(conv_op, 192,192,1,0)
    self.layer2_1x1 = conv_relu(conv_op, 384,384,1,0)
    #bottleneck
    self.bottleneck = nn.Sequential(self.downsampling_layers[3])
    self.bott_1x1 = conv_relu(conv_op, 768,768,1,0)

    #Up part of convunext
    if conv_op == nn.Conv2d:
      transpconv_op = nn.ConvTranspose2d
    else:
      transpconv_op = nn.ConvTranspose3d
    ups = []
    features = [384,192, 96]
    for feature in features:
      ups.append(
          transpconv_op(
              feature*2, feature, kernel_size = 2, stride = 2,
          )
      )
      ups.append(DualPathResponseFusionAttention(conv_op=conv_op, F_g = feature, F_l = feature, F_int = feature))
      ups.append(ConvNextDecoderBlock(conv_op, feature*2, feature,1,1))

    self.ups = nn.ModuleList(ups)
    #Last conv layer
    conv_up = []
    for n in range(2):
      conv_up.append(
        transpconv_op(
          features[-1]//(n+1), features[-1]//(n+2), kernel_size=2, stride = 2,
      ))
      conv_up.append(
        conv_op(
          features[-1]//(n+2), features[-1]//(n+2), kernel_size=1, stride = 1,
      ))

    self.conv_up = nn.ModuleList(conv_up)

    #Last conv layer
    self.conv_last = conv_op(features[-1]//(n+2), num_classes, kernel_size = 1)

  def forward(self, input: torch.Tensor) -> torch.Tensor:

    layer0 = self.layer0(input)
    layer1 = self.layer1(layer0)
    layer2 = self.layer2(layer1)

    bottleneck = self.bottleneck(layer2)
    bottleneck = self.bott_1x1(bottleneck)

    x = self.ups[0](bottleneck) #upsample 2
    layer2 = self.layer2_1x1(layer2)
    layer2 = self.ups[1](u = x, f = layer2)
    x = torch.cat([x,layer2], dim = 1)
    x = self.ups[2](x) #Double Convolutions

    x = self.ups[3](x) #upsample1
    layer1 = self.layer1_1x1(layer1)
    layer1 = self.ups[4](u = x, f = layer1)
    x = torch.cat([x,layer1] , dim = 1)
    x = self.ups[5](x) #Double Convolutions

    x = self.ups[6](x)
    layer0 = self.layer0_1x1(layer0)
    layer0 = self.ups[7](u=x, f=layer0)
    x = torch.cat([x,layer0],dim = 1)
    x = self.ups[8](x)

    for c in range(len(self.conv_up)):
      x = self.conv_up[c](x)

    mask = self.conv_last(x)

    return mask


class ResponseFusionAttentionUConvNextSmall(nn.Module):
    def __init__(
        self,
        input_channels: int,
        strides: list,
        norm_op: Type[nn.Module] = None,
        norm_op_kwargs: dict = None,
        conv_op: Type[nn.Module] = nn.Conv3d,
        features: list = [384, 192, 96],
        deep_supervision: bool = False,
        pretrained_encoder_backbone: bool = True, num_classes: int = 3, **kwargs
    ) -> None:
      super(ResponseFusionAttentionUConvNextSmall,self).__init__()
      # self.stem = nn.Sequential(
      #     conv_op(input_channels, features[-1], 1, padding=0),
      #     norm_op(features[-1], **norm_op_kwargs),
      #     nn.ReLU(inplace=True),
      # )

      #Down part of convunext
      self.encoder = convnext_build(
        conv_op=conv_op, input_channels=input_channels,
        model_size = 'convnext_small', pretrained = pretrained_encoder_backbone, **kwargs
      )

      self.downsampling_layers = list(self.encoder.downsample_layers.children())
      self.encoder_layers = list(self.encoder.stages.children())
      self.layer0 = nn.Sequential(self.downsampling_layers[0], self.encoder_layers[0])
      self.layer1 = nn.Sequential(self.downsampling_layers[1], self.encoder_layers[1])
      self.layer2 = nn.Sequential(self.downsampling_layers[2], self.encoder_layers[2])
      #bottleneck
      self.bottleneck = nn.Sequential(self.downsampling_layers[3]) 
      self.bott_1x1 = conv_relu(conv_op, 768,768,1,0)

      #Up part of convunext
      self.decoder = ConvNextVisionDecoder(
        conv_op=conv_op,
        features=features,
        out_channels=num_classes,
        deep_supervision=deep_supervision
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
      # res = self.stem(input)
      layer0 = self.layer0(input)
      layer1 = self.layer1(layer0)
      layer2 = self.layer2(layer1)

      bottleneck = self.bottleneck(layer2)
      bottleneck = self.bott_1x1(bottleneck)

      mask = self.decoder([bottleneck, layer2, layer1, layer0])

      return mask

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)


class ResponseFusionAttentionUConvNextBase(nn.Module):
  def __init__(
      self,
      conv_op: nn.Conv2d,
      strides: list,
      pretrained_encoder_backbone: bool = True,
      deep_supervision: bool = False,
      **kwargs
  ) -> None:
    super(ResponseFusionAttentionUConvNextBase ,self).__init__()
    #Down part of convunext
    self.encoder = convnext_build( conv_op=conv_op,
      model_size = 'convnext_base', pretrained = pretrained_encoder_backbone, **kwargs
    )
    self.downsampling_layers = list(self.encoder.downsample_layers.children())
    self.encoder_layers = list(self.encoder.stages.children())
    self.layer0 = nn.Sequential(self.downsampling_layers[0], self.encoder_layers[0])
    self.layer1 = nn.Sequential(self.downsampling_layers[1], self.encoder_layers[1])
    self.layer2 = nn.Sequential(self.downsampling_layers[2], self.encoder_layers[2])

    #bottleneck
    self.bottleneck = nn.Sequential(self.downsampling_layers[3])
    self.bott_1x1 = conv_relu(conv_op, 1024,1024,1,0)

    self.decoder = ConvNextVisionDecoder(
      conv_op=conv_op,
      out_channels=3
      )

  def forward(self, input: torch.Tensor) -> torch.Tensor:

    layer0 = self.layer0(input)
    layer1 = self.layer1(layer0)
    layer2 = self.layer2(layer1)

    bottleneck = self.bottleneck(layer2)
    bottleneck = self.bott_1x1(bottleneck)

    mask = self.decoder([bottleneck, layer0, layer1, layer2])

    return mask

  # @staticmethod
  # def initialize(module):
  #     InitWeights_He(1e-2)(module)
  #     init_last_bn_before_add_to_0(module)


class ResponseFusionAttentionUConvNextLarge(nn.Module):
  def __init__(
      self, 
      strides: list,
      pretrained_encoder_backbone: bool = True, num_classes: int = 1, **kwargs
  ) -> None:
    super(ResponseFusionAttentionUConvNextLarge,self).__init__()
    #Down part of convunext
    self.encoder = convnext_build(
      model_size = 'convnext_large', pretrained = pretrained_encoder_backbone, **kwargs
    )
    self.downsampling_layers = list(self.encoder.downsample_layers.children())
    self.encoder_layers = list(self.encoder.stages.children())
    self.layer0 = nn.Sequential(self.downsampling_layers[0], self.encoder_layers[0])
    self.layer0_1x1 = conv_relu(192,192,1,0) 
    self.layer1 = nn.Sequential(self.downsampling_layers[1], self.encoder_layers[1])
    self.layer1_1x1 = conv_relu(384,384,1,0) 
    self.layer2 = nn.Sequential(self.downsampling_layers[2], self.encoder_layers[2])
    self.layer2_1x1 = conv_relu(768,768,1,0) 

    #bottleneck
    self.bottleneck = nn.Sequential(self.downsampling_layers[3])
    self.bott_1x1 = conv_relu(1536,1536,1,0)
    #Up part of convunext
    self.ups = nn.ModuleList()

    for feature in [768,384,192]:
      self.ups.append(
          nn.ConvTranspose2d(
              feature*2, feature, kernel_size = 2, stride = 2,
          )
      )
      self.ups.append(DualPathResponseFusionAttention(F_g = feature, F_l = feature, F_int = feature))
      self.ups.append(ConvNextDecoderBlock(feature*2, feature,1,1))

    #Last conv layer
    self.conv_last = nn.Conv2d(192, num_classes, kernel_size = 1)

  def forward(self, input : torch.Tensor) -> torch.Tensor:

    layer0 = self.layer0(input)
    layer1 = self.layer1(layer0)
    layer2 = self.layer2(layer1)

    bottleneck = self.bottleneck(layer2)
    bottleneck = self.bott_1x1(bottleneck)

    x = self.ups[0](bottleneck) #upsample 2
    layer2 = self.layer2_1x1(layer2)
    layer2 = self.ups[1](u = x, f = layer2)
    x = torch.cat([x,layer2], dim = 1)
    x = self.ups[2](x) #Double Convolutions

    x = self.ups[3](x) #upsample1
    layer1 = self.layer1_1x1(layer1)
    layer1 = self.ups[4](u = x, f = layer1)
    x = torch.cat([x,layer1] , dim = 1)
    x = self.ups[5](x) #Double Convolutions

    x = self.ups[6](x)
    layer0 = self.layer0_1x1(layer0)
    layer0 = self.ups[7](u=x, f=layer0)
    x = torch.cat([x,layer0],dim = 1)
    x = self.ups[8](x)

    mask = self.conv_last(x)

    return mask


class ConvNextVisionDecoder(nn.Module):
  def __init__(
    self,
    conv_op: nn.Conv2d,
    features: list = [512,256,128],
    out_channels: int = 3,
    deep_supervision: bool = False
    ):
    super().__init__()
    self.deep_supervision = deep_supervision
    self.features = features

    #Up part of convunext
    if conv_op == nn.Conv2d:
      transpconv_op = nn.ConvTranspose2d
    else:
      transpconv_op = nn.ConvTranspose3d

    pointwise = []
    upsample = []
    attention_fusion = []
    decoderblocks = []
    seg_layers = []
    for feature in features:
      pointwise.append(conv_relu(conv_op, feature,feature,1,0))
      upsample.append(
          transpconv_op(
              feature*2, feature, kernel_size = 2, stride = 2,
          )
      )
      attention_fusion.append(DualPathResponseFusionAttention(
        conv_op=conv_op,
        F_g = feature, F_l = feature, F_int = feature))
      decoderblocks.append(ConvNextDecoderBlock(
        conv_op=conv_op,
        input_dim=feature*2,
        output_dim=feature,
        stride=1, padding=1))
      seg_layers.append(conv_op(feature, out_channels, 1,1, bias=True))

    self.pwconv = nn.ModuleList(pointwise)
    self.ups = nn.ModuleList(upsample)
    self.atf = nn.ModuleList(attention_fusion)
    self.dblock = nn.ModuleList(decoderblocks)

    #Last conv layer
    resolution_up = []
    for n in range(2):
      resolution_up.append(
        transpconv_op(
          features[-1]//(n+1), features[-1]//(n+2), kernel_size=2, stride = 2,
      ))
      resolution_up.append(
        conv_op(
          features[-1]//(n+2), features[-1]//(n+2), kernel_size=1, stride = 1,
      ))
      seg_layers.append(conv_op(features[-1]//(n+2), out_channels, 1,1, bias=True))

    seg_layers.append(conv_op(features[-1]//(n+2), out_channels, 1,1, bias=True))

    self.conv_up = nn.ModuleList(resolution_up)
    self.seg_layers = nn.ModuleList(seg_layers)

  def forward(self, skips: list[torch.Tensor]):

    seg_outputs = []
    for s in range(len(self.features)):
      x = self.ups[s](skips[s]) # upsample
      # print('Upsample_1', x.shape)
      skip = self.pwconv[s](skips[(s+1)])
      # print('conv1 1x1', layer2.shape)
      skip = self.atf[s](u = x, f = skips[s+1])
      x = torch.cat([x,skip], dim = 1)
      x = self.dblock[s](x) # Decoder Convolutions

      if self.deep_supervision:
        seg_outputs.append(self.seg_layers[s](x))

    for c in range(0, len(self.conv_up)-1, 2):
      x = self.conv_up[c](x)
      x = self.conv_up[c+1](x)
      if self.deep_supervision and c != (len(self.conv_up)-2):
        seg_outputs.append(self.seg_layers[s + c + 1](x))
      elif c == (len(self.conv_up)-2):
        seg_outputs.append(self.seg_layers[-1](x))

    # invert seg outputs so that the largest segmentation prediction is returned first
    seg_outputs = seg_outputs[::-1]

    if not self.deep_supervision:
        r = seg_outputs[0]
    else:
        r = seg_outputs
    return r
