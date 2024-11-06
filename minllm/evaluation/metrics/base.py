from abc import abstractmethod
import torch


class Metric:
    @abstractmethod
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        raise NotImplementedError()

    @abstractmethod
    def compute(self) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def to(self, device) -> None:
        raise NotImplementedError()

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

