from accelerate import Accelerator, DataLoaderConfiguration
import argparse
import math
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from minllm.datasets.utils import load_dataset
from minllm.utils import (
    cycle,
    get_obj_from_str,
    instantiate_from_config,
    load_yaml,
    DotConfig,
)

OUTPUT_NAME = "output"


def train(
    config_path: str,
    dataset: str,
    batch_size: int,
    num_training_steps: int,
    save_and_sample_every_n: int = 1000,
):
    global OUTPUT_NAME
    OUTPUT_NAME = f"{OUTPUT_NAME}/{dataset}/{str(Path(config_path).stem)}"

    # Ensure the output directories exist
    os.makedirs(OUTPUT_NAME, exist_ok=True)

    # Open the model configuration
    config = load_yaml(config_path)
    dataset = load_dataset(dataset, context_length=config.model.params.context_length)

    if batch_size <= 0:
        batch_size = config.training.batch_size
    if num_training_steps <= 0:
        num_training_steps = config.training.training_steps

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    model = instantiate_from_config(config.model, use_config_struct=True)
    summary(
        model=model,
        input_size=(
            batch_size,
            config.model.params.context_length,
        ),
        dtypes=[torch.int64],
    )

    # The accelerate library will handle of the GPU device management for us.
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(split_batches=False),
        mixed_precision="no",
    )

    # Prepare the dataset with the accelerator. This makes sure all of the
    # dataset items are placed onto the correct device.
    dataloader = accelerator.prepare(dataloader)

    # We are going to train for a fixed number of steps, so set the dataloader
    # to repeat indefinitely over the entire dataset.
    dataloader = cycle(dataloader)

    # Now create the optimizer
    optimizer = configure_optimizers(
        model=model,
        config=config.training,
    )

    # Move the model and the optimizer to the accelerator as well.
    model = accelerator.prepare(model)

    # Step counter to keep track of training
    step = 0

    # Not mentioned in the DDPM paper, but the original implementation
    # used gradient clipping during training.
    max_grad_norm = 1.0
    average_loss = 0.0
    average_loss_cumulative = 0.0

    with tqdm(initial=step, total=num_training_steps) as progress_bar:
        # Perform gradient descent for the given number of training steps.
        while step < num_training_steps:
            # Set the learning rate for this iteration.
            lr = (
                get_learning_rate(step, config)
                if config.training.learning_rate_decay
                else config.training.learning_rate
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # That training data are the tokens at position i, and the targets
            # are the tokens at position i+1 (the next tokens)
            x, y = next(dataloader)

            # Calculate the loss on the batch of training data.
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1
            )

            # Calculate the gradients at each step in the network.
            accelerator.backward(loss)

            # On a multi-gpu machine or cluster, wait for all of the workers
            # to finish.
            accelerator.wait_for_everyone()

            # Clip the gradients.
            accelerator.clip_grad_norm_(
                model.parameters(),
                max_grad_norm,
            )

            # Perform the gradient descent step using the optimizer.
            optimizer.step()

            # Reset the gradients for the next step.
            optimizer.zero_grad()

            # Show the current loss in the progress bar.
            progress_bar.set_description(
                f"loss: {loss.item():.4f} avg_loss: {average_loss:.4f}"
            )
            average_loss_cumulative += loss.item()

            # To help visualize training, periodically sample from the
            # diffusion model to see how well its doing.
            if step % save_and_sample_every_n == 0:
                save(model, step, loss, optimizer, config)
                average_loss = average_loss_cumulative / float(save_and_sample_every_n)
                average_loss_cumulative = 0.0

            # Update the current step.
            step += 1

            # Update the training progress bar in the console.
            progress_bar.update(1)


def configure_optimizers(model: torch.nn.Module, config: DotConfig):
    weight_decay = config.weight_decay

    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(
        f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    print(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    )

    # Create AdamW optimizer and use the fused version if it is available
    optimizer = get_obj_from_str(config.optimizer.target)(
        optim_groups,
        lr=config.learning_rate,
        **config.optimizer.params.to_dict(),
    )
    return optimizer


def get_learning_rate(step: int, config: DotConfig):
    # 1) linear warmup for warmup_iters steps
    if step < config.training.warmup_steps:
        return config.training.learning_rate * step / config.training.warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if step > config.training.learning_rate_decay_steps:
        return config.training.min_learning_rate
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (step - config.training.warmup_steps) / (
        config.training.learning_rate_decay_steps - config.training.warmup_steps
    )
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return config.training.min_learning_rate + coeff * (
        config.training.learning_rate - config.training.min_learning_rate
    )


def save(
    model: torch.nn.Module,
    step,
    loss,
    optimizer: torch.optim.Optimizer,
    config: DotConfig,
):
    # Save a corresponding model checkpoint.
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "config": config.to_dict(),
        },
        f"{OUTPUT_NAME}/model-{step}.pt",
    )


def main(override=None):
    """
    Main entrypoint for the standalone version of this package.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="tinyshakespeare")
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--num_training_steps", type=int, default=-1)

    args = parser.parse_args()

    train(
        config_path=args.config_path,
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_training_steps=args.num_training_steps,
    )


if __name__ == "__main__":
    main()
