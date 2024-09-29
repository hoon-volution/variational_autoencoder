from VAE.structure_constants import StructureConstants as C
from torch import nn


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = self._build()

    def forward(self, z):
        return self.blocks(z)
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
            out_channels = filters[i - 1]
            # no relu
            if i - 1 == 0:
                return nn.Sequential(
                    nn.ConvTranspose2d(in_channels,
                                       out_channels,
                                       padding=1,
                                       kernel_size=C.KERNEL,
                                       stride=C.STRIDE,
                                       output_padding=1),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                return nn.Sequential(
                    nn.ConvTranspose2d(in_channels,
                                       out_channels,
                                       padding=1,
                                       kernel_size=C.KERNEL,
                                       stride=C.STRIDE,
                                       output_padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(out_channels),
                )
        conv_tr_blocks = (_build_single_block(i) for i in reversed(range(1, len(filters))))

        h_dim = C.H_DIM
        z_dim = C.Z_DIM

        fc = nn.Linear(in_features=z_dim, out_features=h_dim)

        unflatten_size = C.SIZE_BEFORE_LATENT
        unflatten = nn.Unflatten(1, unflatten_size)

        activation = nn.Sigmoid()
        return nn.Sequential(
            fc,
            unflatten,
            nn.ReLU(),
            *conv_tr_blocks,
            activation,
        )
