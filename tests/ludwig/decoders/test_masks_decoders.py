import pytest
import torch

from ludwig.constants import HIDDEN, LOGITS
from ludwig.decoders.masks_decoders import UnetDecoder


@pytest.mark.parametrize("num_classes", [2, 3])
@pytest.mark.parametrize("batch_size", [20, 1])
@pytest.mark.parametrize("weight, height", [(128, 256)])
def test_unet_decoder(num_classes, batch_size, weight, height):

    input = torch.rand(batch_size, 64, weight, height)
    unet_decoder = UnetDecoder(num_classes=num_classes)

    output = unet_decoder(input)

    assert output.shape == (batch_size, num_classes, weight, height), output.shape
