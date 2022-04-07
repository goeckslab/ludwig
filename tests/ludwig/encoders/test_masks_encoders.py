import pytest
import torch

from ludwig.encoders.masks_encoders import UnetEncoder


@pytest.mark.parametrize("height,width,num_channels", [(224, 224, 2)])
def test_unet_(height: int, width: int, num_channels: int):
    unet = UnetEncoder(
        height=height, width=width, num_channels=num_channels
    )
    inputs = torch.rand(2, num_channels, height, width)
    outputs = unet(inputs)
    assert outputs["encoder_output"].shape[1:] == unet.output_shape
