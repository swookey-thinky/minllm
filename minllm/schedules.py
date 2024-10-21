"""Learning Rate Schedules."""

import math
import torch

from minllm.utils import DotConfig


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer, config: DotConfig
):
    num_warmup_steps = config.training.warmup_steps

    print(
        f"Creating cosine learning rate schedule with {num_warmup_steps} warmup steps."
    )

    def get_learning_rate(step: int):
        # Factor in the number of gradient accumulation steps
        # when doing any calculations
        gradient_accumulation_steps = (
            config.training.gradient_accumulation_steps
            if "gradient_accumulation_steps" in config.training
            else 1
        )
        warmup_steps = config.training.warmup_steps // gradient_accumulation_steps
        learning_rate_decay_steps = (
            config.training.learning_rate_decay_steps // gradient_accumulation_steps
        )

        # 1) linear warmup for warmup_iters steps
        if step < warmup_steps:
            return step / warmup_steps

        # 2) if it > lr_decay_iters, return min learning rate
        if step > learning_rate_decay_steps:
            return 1.0

        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (step - warmup_steps) / (learning_rate_decay_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        expected_learning_rate = config.training.min_learning_rate + coeff * (
            config.training.learning_rate - config.training.min_learning_rate
        )
        lambda_lr = expected_learning_rate / config.training.learning_rate
        return lambda_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, get_learning_rate)


def get_constant_schedule_with_warmup(
    optimizer: torch.optim.Optimizer, config: DotConfig
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    num_warmup_steps = config.training.warmup_steps

    print(
        f"Creating constant learning rate schedule with {num_warmup_steps} warmup steps."
    )

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
