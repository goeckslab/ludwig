import pytest
import torch

from ludwig.constants import HIDDEN, LOGITS
from ludwig.decoders.masks_decoders import MasksDecoder


@pytest.mark.parametrize("num_classes", [2, 3])
@pytest.mark.parametrize("batch_size", [20, 1])
@pytest.mark.parametrize("weight, height", [(128, 256)])
def test_masks_decoder(num_classes, batch_size, weight, height):

    input = torch.rand(batch_size, 64, weight, height)
    masks_decoder = MasksDecoder(num_classes=num_classes)

    output = masks_decoder(input)

    assert output.shape == (batch_size, num_classes, weight, height), output.shape
