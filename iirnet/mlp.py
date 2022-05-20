import torch
import pytorch_lightning as pl
import iirnet.loss as loss


class MLP(torch.nn.Module):
    """Multi-layer perceptron module."""

    def __init__(
        self,
        num_points=512,
        num_layers=2,
        hidden_dim=8192,
        max_order=2,
        normalization="none",
        output="sos",  # either "sos" or "zpk"
        eps=1e-8,
    ):
        super().__init__()

        self.num_points = num_points
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.max_order = max_order
        self.normalization = normalization
        self.output = output
        self.eps = eps

        self.layers = torch.nn.ModuleList()

        for n in range(self.num_layers):
            in_features = self.hidden_dim if n != 0 else self.num_points
            out_features = self.hidden_dim
            if n + 1 == self.num_layers:  # no activation at last layer
                linear_layer = torch.nn.Linear(in_features, out_features)
                # linear_layer.bias.data.fill_(0.5)
                self.layers.append(linear_layer)
            else:
                self.layers.append(
                    torch.nn.Sequential(
                        torch.nn.Linear(in_features, out_features),
                        torch.nn.LayerNorm(out_features),
                        torch.nn.LeakyReLU(0.2),
                    )
                )

        n_coef = (self.model_order // 2) * 6
        self.layers.append(torch.nn.Linear(out_features, n_coef))

        if self.normalization == "bn":
            self.bn = torch.nn.BatchNorm1d(self.num_points * 2)

        if self.output == "zpk":
            self.construct_filter = self.construct_filter_zpk
        elif self.output == "sos":
            self.construct_filter = self.construct_filter_sos
        else:
            raise RuntimeError(f"Invalid output: {self.output}")

    def construct_filter_zpk(self, x: torch.Tensor):
        # reshape into sos format (n_section, (b0, b1, b2, a0, a1, a2))
        n_sections = self.model_order // 2
        sos = x.view(-1, n_sections, 6)

        # extract gains, offset from 1
        g = 100 * torch.sigmoid(sos[:, :, 0])

        # all gains are held at 1 except first
        g[:, 1:] = 1.0

        # extract poles, and zeros
        pole_real = sos[:, :, 1]
        pole_imag = sos[:, :, 2]
        zero_real = sos[:, :, 4]
        zero_imag = sos[:, :, 5]

        # ensure stability
        pole = torch.complex(pole_real, pole_imag)
        pole = (
            (1 - self.eps)
            * pole
            * torch.tanh(pole.abs())
            / (pole.abs().clamp(self.eps))
        )

        # ensure zeros inside unit circle
        zero = torch.complex(zero_real, zero_imag)
        zero = (
            (1 - self.eps)
            * zero
            * torch.tanh(zero.abs())
            / (zero.abs().clamp(self.eps))
        )

        # Apply gain g to numerator by multiplying each coefficient by g
        b0 = g
        b1 = g * -2 * zero.real
        b2 = g * ((zero.real**2) + (zero.imag**2))
        a0 = torch.ones(g.shape, device=g.device)
        a1 = -2 * pole.real
        a2 = (pole.real**2) + (pole.imag**2)

        # reconstruct SOS
        out_sos = torch.stack([b0, b1, b2, a0, a1, a2], dim=-1)

        # store zeros poles and gains
        zpk = (zero, pole, g)

        return out_sos, zpk

    def construct_filter_sos(self, x: torch.Tensor):
        # reshape into sos format (n_section, (b0, b1, b2, a0, a1, a2))
        n_sections = self.model_order // 2
        sos = x.view(-1, n_sections, 6)

        # extract gains, offset from 1
        g = 100 * torch.sigmoid(sos[:, :, 0])

        # all gains are held at 1 except first
        g[:, 1:] = 1.0

        # extract numerator and denomenator coefficients
        b1 = sos[:, :, 1]
        b2 = sos[:, :, 2]
        a1 = sos[:, :, 4]
        a2 = sos[:, :, 5]

        # ensure stability
        a0 = torch.ones(g.shape, device=g.device)
        a1 = 2 * torch.tanh(a1)
        a2 = ((2 - a1.abs()) * torch.tanh(a2) + a1.abs()) / 2

        # ensure min phase
        b0 = g
        b1 = 2 * torch.tanh(b1)
        b2 = ((2 - b1.abs()) * torch.tanh(b2) + b1.abs()) / 2

        # reconstruct SOS
        out_sos = torch.stack([b0, b1, b2, a0, a1, a2], dim=-1)

        return out_sos, None

    def forward(self, mag, phs=None):
        x = mag

        for layer in self.layers:
            x = layer(x)

        out_sos, zpk = self.construct_filter(x)

        return out_sos, zpk
