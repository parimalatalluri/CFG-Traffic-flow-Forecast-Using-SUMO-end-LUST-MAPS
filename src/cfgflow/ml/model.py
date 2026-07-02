from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelMeta:
    version: int
    dt_s: float
    t_in: int
    t_out: int
    edge_ids: list[str]
    arcs: list[tuple[int, int]]
    config: dict


def require_torch():
    try:
        import torch  # noqa: F401
        import torch.nn as nn  # noqa: F401

        return
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Torch not available. Install ML extras first: `pip install -e .[ml]` "
            "or install PyTorch via conda."
        ) from e


def build_row_normalized_adjacency(*, n: int, arcs: list[tuple[int, int]]):
    require_torch()
    import torch

    a = torch.zeros((n, n), dtype=torch.float32)
    a.fill_diagonal_(1.0)
    for i, j in arcs:
        if 0 <= i < n and 0 <= j < n:
            a[i, j] = 1.0
    row_sum = a.sum(dim=1, keepdim=True).clamp_min(1.0)
    return a / row_sum


def build_model(*, n: int, t_out: int, hidden: int = 32, kernel: int = 3, alpha: float = 0.7):
    """
    A small spatiotemporal model:
      Temporal conv -> Graph mix -> Temporal conv -> Graph mix -> node-wise head.
    Input:  (B, T_in, N)
    Output: (B, T_out, N)
    """
    require_torch()
    import torch
    import torch.nn as nn

    class GraphTemporalCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden = hidden
            self.kernel = kernel
            self.alpha = float(alpha)
            pad = (kernel - 1, 0)
            self.conv1 = nn.Conv2d(1, hidden, kernel_size=(kernel, 1), padding=pad)
            self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=(kernel, 1), padding=pad)
            self.head = nn.Conv1d(hidden, t_out, kernel_size=1)
            self.act = nn.ReLU()

        def graph_mix(self, x: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
            # x: (B, C, T, N), a_norm: (N, N)
            return self.alpha * x + (1.0 - self.alpha) * torch.matmul(x, a_norm.T)

        def forward(self, x: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
            x = x.unsqueeze(1)  # (B, 1, T, N)
            x = self.act(self.conv1(x))
            x = self.graph_mix(x, a_norm)
            x = self.act(self.conv2(x))
            x = self.graph_mix(x, a_norm)
            h = x[:, :, -1, :]  # (B, hidden, N)
            y = self.head(h)  # (B, T_out, N)
            return y

    return GraphTemporalCNN()

