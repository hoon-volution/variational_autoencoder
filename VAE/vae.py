from torch import nn
from VAE.encoder import Encoder
from VAE.bottleneck import Bottleneck
from VAE.decoder import Decoder

class VariationalAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.bottleneck = Bottleneck()
        self.decoder = Decoder()

    def forward(self, x):
        x_encoded = self.encoder(x)
        z, mu, logvar = self.bottleneck(x_encoded)
        z = self.decoder(z)
        return z, mu, logvar
