from VAE.structure_constants import StructureConstants as C
from torch import nn, randn


class Bottleneck(nn.Module):
    def __init__(self,):
        super().__init__()
        self._build()
        self.fc_mu: nn.Linear
        self.fc_logvar: nn.Linear

    def forward(self, x):
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        std = (logvar * 0.5 ).exp()

        # device
        if next(self.parameters(), None) is not None:
            device = next(self.parameters()).device
        elif next(self.buffers(), None) is not None:
            device = next(self.buffers()).device
        else:
            raise ValueError

        e = randn(*mu.size()).to(device)
        z = mu + std * e
        return z, mu, logvar

    def _build(self):
        h_dim = C.H_DIM
        z_dim = C.Z_DIM

        self.fc_mu = nn.Linear(in_features=h_dim, out_features=z_dim)
        self.fc_logvar = nn.Linear(in_features=h_dim, out_features=z_dim)

