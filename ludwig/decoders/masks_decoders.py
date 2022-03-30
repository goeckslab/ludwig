import logging

from ludwig.constants import LOSS, MASKS, TYPE
from ludwig.decoders.base import Decoder
from ludwig.decoders.registry import register_decoder
from ludwig.utils.unet_utils import OutConv


logger = logging.getLogger(__name__)


@register_decoder("masks", MASKS, default=True)
class Masks(Decoder):
    def __init__(
        self,
        num_classes: int,
        **kwargs,
    ):
        super().__init__()
        logger.debug(f" {self.name}")

        logger.debug("  Cov2D")
        self.num_classes = num_classes

        self.outc = OutConv(64, num_classes)

        self.sampled_loss = False
        if LOSS in kwargs and TYPE in kwargs[LOSS] and kwargs[LOSS][TYPE] is not None:
            self.sampled_loss = kwargs[LOSS][TYPE].startswith("sampled")

    @property
    def input_shape(self):
        return self.outc.input_shape

    def forward(self, inputs, **kwargs):
        return self.outc(inputs)
