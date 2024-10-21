import torch
from torchmetrics import text

from minllm.evaluation.metrics.base import Metric

_IGNORE_INDEX = -100


class Perplexity(Metric):
    def __init__(self):
        super().__init__()

        self._metric = text.Perplexity(ignore_index=_IGNORE_INDEX)

    @property
    def name(self):
        return "perplexity"

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        B, L, D = preds.shape
        B, C = targets.shape

        # The model is a next token prediction task, so the sequence length
        # L is the context to the model. So for the preds, preds[0:L-1] is
        # the context for the model, and preds[-1] is the next token prediction,
        # corresponding to a target class of targets[-1]. So set the targets
        # to _IGNORE_INDEX for all of the context entries
        targets = targets.detach().clone()
        targets[:, :-1] = _IGNORE_INDEX
        self._metric.update(
            preds=preds.to(self._metric.device),
            target=targets.to(self._metric.device),
        )

    @torch.no_grad()
    def compute(self) -> torch.Tensor:
        return self._metric.compute()
