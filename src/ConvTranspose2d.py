import math
from typing import Union, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from src import utils


class ConvTranspose2d(nn.Module):
    """
    Step-by-step implementation of a 2D transposed convolutional layer.

    Arguments:
    in_channels (int): Number of input channels.
    out_channels (int): Number of output channels produced by the convolution.
    kernel_size (int or tuple): Size of the convolutional kernel.
    stride (int or tuple, optional): Stride of the convolution. Default: 1
    padding (int or tuple, optional): Zero-padding added to the input. Default: 0
    output_padding (int or tuple, optional): Additional padding added to output. Default: 0
    dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
    bias (bool, optional): If True, adds a learnable bias to the output. Default: True
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        output_padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = True,
    ):
        super(ConvTranspose2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = utils.pair(kernel_size)
        self.stride = utils.pair(stride)
        self.padding = utils.pair(padding)
        self.output_padding = utils.pair(output_padding)
        self.dilation = utils.pair(dilation)

        self.weight = nn.Parameter(
            torch.empty(
                (
                    self.in_channels,
                    self.out_channels * self.kernel_size[0] * self.kernel_size[1],
                )
            )
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.zeros((self.out_channels)))

    def get_output_size(self, input_size: int, idx: int) -> int:
        """
        Calculates the output size (i.e. feature map size) of the transposed convolutional layer.

        Arguments:
        input_size: Height or width of input tensor.
        idx: Index to choose between height and width values (0 = height, 1 = width).

        Returns:
        Output size of the tensor after performing transposed convolution given the input tensor.
        """
        return (
            (input_size - 1) * self.stride[idx]
            - 2 * self.padding[idx]
            + self.dilation[idx] * (self.kernel_size[idx] - 1)
            + self.output_padding[idx]
            + 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines forward pass of the transposed convolutional layer.

        Arguments:
        x: Input tensor.

        Returns:
        The tensor obtained after applying the convolutional layer.
        """
        height, width = x.shape[-2:]

        # Rearrange x to (b, c, h*w).
        x = rearrange(x, "b c h w -> b c (h w)")

        # Apply weight matrix.
        output = torch.einsum("b i p, i o -> b o p", x, self.weight)

        # "Depatchify": Combine patches by summing overlapping values.
        output_size = (
            self.get_output_size(height, 0),
            self.get_output_size(width, 1),
        )
        output = torch.nn.functional.fold(
            output,
            output_size,
            stride=self.stride,
            kernel_size=self.kernel_size,
            padding=self.padding,
            dilation=self.dilation,
        )

        if hasattr(self, "bias"):
            return output + self.bias[None, :, None, None]

        return output
