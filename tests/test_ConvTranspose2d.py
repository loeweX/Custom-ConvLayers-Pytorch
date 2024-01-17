import pytest
import torch
import torch.nn as nn
from einops import rearrange

from src import ConvTranspose2d


@pytest.fixture
def conv_params():
    """
    Fixture for creating predefined set of parameters for transposed convolutional layers.

    Returns:
    A dictionary of parameters for transposed convolutional layers.
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
        "output_padding": (1, 2),
    }


def test_ConvTranspose2d(conv_params):
    """
    Test to ensure that the output of the custom ConvTranspose2d class is equal to the 
    output of the PyTorch ConvTranspose2d implementation.

    Arguments:
    conv_params: Parameters for transposed convolutional layers.
    """
    x = torch.rand(
        conv_params["batch_size"],
        conv_params["in_channels"],
        conv_params["height"],
        conv_params["width"],
    )

    pytorch_conv_tran = torch.nn.ConvTranspose2d(
        conv_params["in_channels"],
        conv_params["out_channels"],
        conv_params["kernel_size"],
        conv_params["stride"],
        conv_params["padding"],
        conv_params["output_padding"],
        dilation=conv_params["dilation"],
    )
    output_pytorch = pytorch_conv_tran(x)

    custom_conv_tran = ConvTranspose2d.ConvTranspose2d(
        conv_params["in_channels"],
        conv_params["out_channels"],
        conv_params["kernel_size"],
        conv_params["stride"],
        conv_params["padding"],
        conv_params["output_padding"],
        dilation=conv_params["dilation"],
    )

    # Set the weight parameter of custom ConvTranspose2D to match with PyTorch ConvTranspose2d.
    custom_conv_tran.weight = nn.Parameter(
        rearrange(pytorch_conv_tran.weight, "i o h w -> i (o h w)")
    )
    custom_conv_tran.bias = nn.Parameter(pytorch_conv_tran.bias)

    output_custom = custom_conv_tran(x)

    torch.testing.assert_close(output_custom, output_pytorch, rtol=1e-4, atol=1e-4)


def test_ConvTranspose2d_gradoemts(conv_params):
    """
    Test the gradients calculated by the custom ConvTranspose2d layer against the native
    PyTorch implementation.

    Arguments:
    conv_params: Parameters for transposed convolutional layers.
    """
    x = torch.rand(
        conv_params["batch_size"],
        conv_params["in_channels"],
        conv_params["height"],
        conv_params["width"],
        dtype=torch.float,
        requires_grad=True,
    )

    pytorch_conv_tran = torch.nn.ConvTranspose2d(
        conv_params["in_channels"],
        conv_params["out_channels"],
        conv_params["kernel_size"],
        conv_params["stride"],
        conv_params["padding"],
        conv_params["output_padding"],
        dilation=conv_params["dilation"],
    )
    output_pytorch = pytorch_conv_tran(x)

    target = torch.rand_like(output_pytorch)
    loss_pytorch = (output_pytorch - target).sum()
    loss_pytorch.backward()

    custom_conv_tran = ConvTranspose2d.ConvTranspose2d(
        conv_params["in_channels"],
        conv_params["out_channels"],
        conv_params["kernel_size"],
        conv_params["stride"],
        conv_params["padding"],
        conv_params["output_padding"],
        dilation=conv_params["dilation"],
    )

    # Set the weight parameter of custom ConvTranspose2D to match with PyTorch ConvTranspose2d.
    custom_conv_tran.weight = nn.Parameter(
        rearrange(pytorch_conv_tran.weight, "i o h w -> i (o h w)")
    )
    custom_conv_tran.bias = nn.Parameter(pytorch_conv_tran.bias)

    output_custom = custom_conv_tran(x)
    loss_custom = (output_custom - target).sum()
    loss_custom.backward()

    torch.testing.assert_close(
        pytorch_conv_tran.weight.grad.reshape(conv_params["in_channels"], -1),
        custom_conv_tran.weight.grad,
    )
    torch.testing.assert_close(pytorch_conv_tran.bias.grad, custom_conv_tran.bias.grad)
