import pytest
import torch
import torch.nn as nn
from einops import rearrange

from src import Conv2d


@pytest.fixture
def conv_params():
    """
    Fixture for creating predefined set of parameters for a convolutional layer.

    Returns:
    A dictionary of parameters for a convolutional layer.
    """
    return {
        "batch_size": 8,
        "in_channels": 32,
        "height": 16,
        "width": 16,
        "out_channels": 64,
        "kernel_size": (3, 2),
        "stride": (2, 3),
        "padding": (1, 0),
        "dilation": (3, 1),
    }


def test_Conv2d(conv_params):
    """
    Test to ensure that the output of the custom Conv2d class is equal to the 
    output of the PyTorch Conv2d implementation.

    Arguments:
    conv_params: Parameters for the convolutional layers.
    """
    x = torch.rand(
        conv_params["batch_size"],
        conv_params["in_channels"],
        conv_params["height"],
        conv_params["width"],
    )

    pytorch_conv = torch.nn.Conv2d(
        conv_params["in_channels"],
        conv_params["out_channels"],
        conv_params["kernel_size"],
        conv_params["stride"],
        conv_params["padding"],
        conv_params["dilation"],
    )
    output_pytorch = pytorch_conv(x)

    custom_conv = Conv2d.Conv2d(
        conv_params["in_channels"],
        conv_params["out_channels"],
        conv_params["kernel_size"],
        conv_params["stride"],
        conv_params["padding"],
        conv_params["dilation"],
    )

    # Set the weight parameter of custom Conv2D to match with PyTorch Conv2d.
    custom_conv.weight = nn.Parameter(
        rearrange(pytorch_conv.weight, "o i h w -> o (i h w)")
    )
    custom_conv.bias = nn.Parameter(pytorch_conv.bias)

    output_custom = custom_conv(x)

    # Assert both the outputs are similar (allowing some tolerance).
    torch.testing.assert_close(output_custom, output_pytorch, rtol=1e-4, atol=1e-4)


def test_Conv2d_gradients(conv_params):
    """
    Test the gradients calculated by the custom Conv2d layer against the native
    PyTorch implementation.

    Arguments:
    conv_params: Parameters for the convolutional layers.
    """
    x = torch.rand(
        conv_params["batch_size"],
        conv_params["in_channels"],
        conv_params["height"],
        conv_params["width"],
        dtype=torch.float,
        requires_grad=True,
    )

    pytorch_conv = torch.nn.Conv2d(
        conv_params["in_channels"],
        conv_params["out_channels"],
        conv_params["kernel_size"],
        conv_params["stride"],
        conv_params["padding"],
        conv_params["dilation"],
    )
    output_pytorch = pytorch_conv(x)

    target = torch.rand_like(output_pytorch)
    loss_pytorch = (output_pytorch - target).sum()
    loss_pytorch.backward()

    custom_conv = Conv2d.Conv2d(
        conv_params["in_channels"],
        conv_params["out_channels"],
        conv_params["kernel_size"],
        conv_params["stride"],
        conv_params["padding"],
        conv_params["dilation"],
    )

    # Set the weight parameter of custom Conv2D to match with PyTorch Conv2d.
    custom_conv.weight = nn.Parameter(
        rearrange(pytorch_conv.weight, "o i h w -> o (i h w)")
    )
    custom_conv.bias = nn.Parameter(pytorch_conv.bias)

    output_custom = custom_conv(x)
    loss_custom = (output_custom - target).sum()
    loss_custom.backward()

    # Compare gradients.
    torch.testing.assert_close(
        pytorch_conv.weight.grad.reshape(conv_params["out_channels"], -1),
        custom_conv.weight.grad,
    )
    torch.testing.assert_close(pytorch_conv.bias.grad, custom_conv.bias.grad)
