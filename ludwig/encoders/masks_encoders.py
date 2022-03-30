import logging
from typing import Dict

import torch
from ludwig.constants import IMAGE
from ludwig.encoders.base import Encoder
from ludwig.encoders.registry import register_encoder
from ludwig.utils.image_utils import get_img_output_shape
from ludwig.utils.torch_utils import get_activation
from ludwig.utils.unet_utils import *


logger = logging.getLogger(__name__)


@register_encoder("unet", IMAGE)
class UnetEncoder(Encoder):
    def __init__(
        self,
        height: int,
        width: int,
        num_channels: int = None,
        bilinear: bool = False,
        **kwargs,
    ):
        super().__init__()

        logger.debug(f" {self.name}")

        # map parameter input feature config names to internal names
        img_height = height
        img_width = width
        first_in_channels = num_channels

        self._input_shape = (first_in_channels, img_height, img_width)

        if first_in_channels is None:
            raise ValueError("first_in_channels must not be None.")

        logger.debug("  UnetEncoder")
        self.inc = DoubleConv(first_in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size(self._input_shape)

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        :param inputs: The inputs fed into the encoder.
                Shape: [batch x channels x height x width], type torch.uint8
        """

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        outputs = self.up4(x, x1)

        return {"encoder_output": outputs}
