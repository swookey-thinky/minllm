"""Helper class for LLM evaluation."""

from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from minllm.datasets.utils import load_dataset
from minllm.utils import DotConfig, instantiate_from_config, cycle

_METRICS = {
    "perplexity": {
        "target": "minllm.evaluation.metrics.perplexity.Perplexity",
        "params": {},
    }
}


class Evaluator:
    def __init__(self, config: DotConfig):
        self._config = config

        # Create all of the metrics we are going to measure.
        self._metrics = [
            instantiate_from_config(_METRICS[metric]) for metric in config.metrics
        ]

        # The total number of samples to evaluate
        self._total_samples = config.total_samples

        # The number of samples to evaluate in each batch
        self._samples_per_batch = config.samples_per_batch

    @torch.inference_mode()
    def evaluate(
        self, model: torch.nn.Module, dataloader: DataLoader, accelerator: Accelerator
    ):
        results = {}
        with torch.inference_mode():
            step = 0
            with tqdm(
                initial=step,
                total=self._total_samples // self._samples_per_batch,
                leave=False,
                desc="Evaluating",
            ) as progress_bar:
                while step < (self._total_samples // self._samples_per_batch):
                    x, y = next(dataloader)
                    with accelerator.autocast():
                        logits, _ = model(x)

                        for metric in self._metrics:
                            metric.update(logits, y)
                    step += 1
                    progress_bar.update(1)

                # Compute all of the final results
                for metric in self._metrics:
                    results[metric.name] = metric.compute().detach().numpy()
        return results
