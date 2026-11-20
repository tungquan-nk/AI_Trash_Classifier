"""Shared test doubles for API and inference tests."""

import torch


class FixedLogitsModel(torch.nn.Module):
    """Return deterministic logits and optionally retain inference observations."""

    def __init__(self, logits: list[float] | None = None) -> None:
        super().__init__()
        self.logits = logits if logits is not None else [0.0, 2.0, 1.0]
        self.seen_shape: tuple[int, ...] | None = None
        self.seen_device: torch.device | None = None
        self.inference_mode: bool | None = None
        self.seen_input: torch.Tensor | None = None

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        self.seen_shape = tuple(image_tensor.shape)
        self.seen_device = image_tensor.device
        self.inference_mode = torch.is_inference_mode_enabled()
        self.seen_input = image_tensor.detach().clone()
        return torch.tensor(
            [self.logits], dtype=torch.float32, device=image_tensor.device
        ).expand(image_tensor.shape[0], -1)
