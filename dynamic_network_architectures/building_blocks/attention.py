import torch.nn as nn
from torch import Tensor
from dynamic_network_architectures.building_blocks.residual import BasicBlockD as ResidualBlock
import torch
from typing import Union, Type, List, Tuple
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.initialization.weight_init import InitWeights_XavierUniform


class DualPathResponseFusionAttention(nn.Module):
  """
    Official implementation of our proposed Dual Path Response Fusion Attention module.
  For better clarity, refer to the symbols given in the paper (Fig 3.)

  
  Keyword arguments:
  argument -- F_g, F_l and F_int are tunable dimensions for Conv 1x1 layers
  Return: returns a torch Tensor n_u2 + theta_fuse
  """
  

  def __init__(self, conv_op: nn.Conv2d, F_g: int , F_l: int , F_int: int) -> None:
        super(DualPathResponseFusionAttention,self).__init__()
        if conv_op == nn.Conv2d:
          norm_op = nn.BatchNorm2d
        else:
          norm_op = nn.BatchNorm3d

        self.W_f  = nn.Sequential(
            conv_op(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=False),
            norm_op(F_int)
            )
        
        self.W_u = nn.Sequential(
            conv_op(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            norm_op(F_int)
        )

        self.W = nn.Sequential(
            conv_op(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            norm_op(1),
            nn.Sigmoid()
        )
        
        self.gelu = nn.GELU(approximate="none")

  def forward(self, u: Tensor , f: Tensor) -> Tensor:
        n_u = self.W_u(u)
        n_u2 = self.gelu(n_u)

        n_f = self.W_f(f)

        psi = self.gelu(n_u + n_f)
        psi = self.W(psi)
        theta_fuse = n_f * psi

        return n_u2 + theta_fuse


class ResidualAttentionSoftMasking(nn.Module):
  def __init__(self,
        conv_op: nn.Conv3d,
        input_channels=None,
        output_channels: int = 3, depth: int = 2,
        kernel_size: Union[int, List[int], Tuple[int, ...]] = 3,
        stride: Union[int, List[int], Tuple[int, ...]] = 1,
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,):
      super(ResidualAttentionSoftMasking, self).__init__()
      self.depth = depth
      self.firstResBlock = ResidualBlock(
        conv_op, input_channels, input_channels,
        kernel_size=kernel_size,
                    stride=stride,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    dropout_op=dropout_op,
                    dropout_op_kwargs=dropout_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs)

      self.trunk_branches = nn.Sequential(*[
          ResidualBlock(conv_op, input_channels, input_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    dropout_op=dropout_op,
                    dropout_op_kwargs=dropout_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs),
          ResidualBlock(conv_op, input_channels, input_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    dropout_op=dropout_op,
                    dropout_op_kwargs=dropout_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs)]
        )

      upsampling_op = nn.ConvTranspose3d  # nn.Upsample(scale_factor=2, mode='bilinear') if conv_op == nn.Conv2d else 

      encoder_stages = []
      decoder_stages = []
      skip_residuals = []
      for s in range(depth):
        softmax_encoding_block = nn.Sequential(
          *[
            conv_op(input_channels, input_channels, kernel_size=kernel_size, stride=2, padding=1),
            ResidualBlock(conv_op, input_channels, input_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    dropout_op=dropout_op,
                    dropout_op_kwargs=dropout_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs)
          ]
        )
        encoder_stages.append(softmax_encoding_block)

        skip_residuals.append(  # Skip connection RES
            ResidualBlock(
                  conv_op,
                  input_channels, input_channels,
                  kernel_size=kernel_size,
                  stride=stride,
                  conv_bias=conv_bias,
                  norm_op=norm_op,
                  norm_op_kwargs=norm_op_kwargs,
                  dropout_op=dropout_op,
                  dropout_op_kwargs=dropout_op_kwargs,
                  nonlin=nonlin,
                  nonlin_kwargs=nonlin_kwargs)
        )

        softmax_decoding_block = nn.Sequential(
          *[
            ResidualBlock(conv_op, input_channels, input_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    dropout_op=dropout_op,
                    dropout_op_kwargs=dropout_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs),
            upsampling_op(
              input_channels,
              input_channels,  #output_channels if s == depth-1 else input_channels,
              kernel_size=2, stride=2)
          ]
        )
        decoder_stages.append(softmax_decoding_block)

      self.bridge_softmax_blocks = nn.Sequential(
        *[
          conv_op(input_channels, input_channels, kernel_size=kernel_size, stride=2, padding=1),
          ResidualBlock(conv_op, input_channels, input_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    dropout_op=dropout_op,
                    dropout_op_kwargs=dropout_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs),
        # ResidualBlock(conv_op, input_channels, input_channels,
        #             kernel_size=kernel_size,
        #             stride=stride,
        #             conv_bias=conv_bias,
        #             norm_op=norm_op,
        #             norm_op_kwargs=norm_op_kwargs,
        #             dropout_op=dropout_op,
        #             dropout_op_kwargs=dropout_op_kwargs,
        #             nonlin=nonlin,
        #             nonlin_kwargs=nonlin_kwargs),
        upsampling_op(input_channels, input_channels, kernel_size=2, stride=2)
        ]
      )

      decoder_stages.append(nn.Sequential(
        *[
          conv_op(input_channels, input_channels, 1, 1, 0, bias=True),
          conv_op(
            input_channels,
            input_channels,  # output_channels if s == depth-1 else input_channels,
            1, 1, 0, bias=True),
          nn.Sigmoid()
        ])
      )

      self.encoding_blocks = nn.ModuleList(encoder_stages)
      self.decoding_blocks = nn.ModuleList(decoder_stages)
      self.skip_residuals = nn.ModuleList(skip_residuals)

  def forward(self, x):
    residual = x
    x = self.firstResBlock(x)
    # Trunk branch
    trunk_branch_x = self.trunk_branches(x)

    # Soft Mask Branch
    skips_b = []
    for s in range(self.depth):
      x = self.encoding_blocks[s](x)
      skips_b.append(self.skip_residuals[s](x))

    x = self.bridge_softmax_blocks(x)

    for s in range(self.depth):
      x = x + skips_b[-(1 + s)]
      x = self.decoding_blocks[s](x)

    x = self.decoding_blocks[-1](x)
    x = x + 1
    x = x * trunk_branch_x # Attention: (1 + output_soft_mask) * output_trunk
    return x + residual
  
  def initialize(module):
      InitWeights_XavierUniform(1e-2)(module)


if __name__ == '__main__':
  filters = 32
  model = ResidualAttentionSoftMasking(
    conv_op=nn.Conv3d, input_channels=filters,
    output_channels=filters)
  import torch
  x = torch.rand([1,filters,96,96,96])
  output = model(x)
  print('OUTPUT', output[0].shape, output[1].shape)
  #print('OUTPUT', output.shape)

  if True:
      import hiddenlayer as hl
      print('Building network diagram')

      g = hl.build_graph(model, x,
                          transforms=None)
      g.save("network_architecture.pdf")
      del g
