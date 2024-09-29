from VAE.structure_constants import StructureConstants as C
from torch import nn


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = self._build()

    def forward(self, x):
        return self.blocks(x)

    @staticmethod
    def _build():
        filters = [
            C.FILTER0,
            C.FILTER1,
            C.FILTER2,
            C.FILTER3,
            C.FILTER4,
            C.FILTER5,
        ]
        def _build_single_block(i: int):
            in_channels = filters[i]
            out_channels = filters[i + 1]
            return nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          padding=1,
                          kernel_size=C.KERNEL,
                          stride=C.STRIDE, ),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels),
            )
        conv_blocks = (_build_single_block(i) for i in range(len(filters) - 1))

        return nn.Sequential(
            *conv_blocks,
            nn.Dropout(0.8),
            nn.Flatten()
        )
