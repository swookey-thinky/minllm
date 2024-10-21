from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import GradientAccumulationPlugin
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm.auto import tqdm
from typing import Optional

from minllm.datasets.utils import load_dataset
from minllm.evaluation.evaluator import Evaluator
from minllm.schedules import get_cosine_schedule_with_warmup
from minllm.utils import (
    cycle,
    get_obj_from_str,
    instantiate_from_config,
    load_yaml,
    DotConfig,
)


def train(
    config_path: str,
    output_path: str,
    batch_size: int = -1,
    num_training_steps: int = -1,
    save_and_sample_every_n: int = -1,
    mixed_precision: str = "",
):
    # Open the model configuration
    config = load_yaml(config_path)

    # Use the dataset from the training config.
    dataset = config.training.dataset

    OUTPUT_NAME = f"{output_path}/{dataset}/{str(Path(config_path).stem)}"

    # Ensure the output directories exist
    os.makedirs(OUTPUT_NAME, exist_ok=True)

    # Load a tokenizer if we have one
    tokenizer = None
    if "tokenizer" in config:
        tokenizer = instantiate_from_config(config.tokenizer.to_dict())

    # Create the evaluator for measuring the training performance.
    evaluator = Evaluator(config=config.training.evaluation)

    # Load the training dataset
    dataset = load_dataset(
        dataset,
        context_length=config.model.params.context_length,
        tokenizer=tokenizer,
    )

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

    # Check to see if we are using gradient accumulation
    gradient_accumulation_steps = 1
    if "training" in config and "gradient_accumulation_steps" in config.training:
        gradient_accumulation_steps = config.training.gradient_accumulation_steps

    # The accelerate library will handle of the GPU device management for us.
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(split_batches=False),
        mixed_precision=(
            mixed_precision
            if mixed_precision != ""
            else config.training.mixed_precision
        ),
        gradient_accumulation_plugin=GradientAccumulationPlugin(
            num_steps=gradient_accumulation_steps,
            adjust_scheduler=True,
            sync_with_dataloader=False,
        ),
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

    # Create the learning rate schedule
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, config=config)

    # Move the model and the optimizer to the accelerator as well.
    model, lr_scheduler = accelerator.prepare(model, lr_scheduler)

    # Step counter to keep track of training
    step = 0

    # Not mentioned in the DDPM paper, but the original implementation
    # used gradient clipping during training.
    max_grad_norm = 1.0
    average_loss = 0.0
    average_loss_cumulative = 0.0
    save_and_sample_every_n = (
        save_and_sample_every_n
        if save_and_sample_every_n > 0
        else config.training.save_and_sample_every_n
    )

    current_evaluation_results = {}
    with tqdm(initial=step, total=num_training_steps) as progress_bar:
        # Perform gradient descent for the given number of training steps.
        while step < num_training_steps:
            with accelerator.accumulate(model):
                # That training data are the tokens at position i, and the targets
                # are the tokens at position i+1 (the next tokens)
                x, y = next(dataloader)

                # Calculate the loss on the batch of training data.
                with accelerator.autocast():
                    logits, _ = model(x)
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

            # Step the learning rate scheduler as well
            lr_scheduler.step()

            # Reset the gradients for the next step.
            optimizer.zero_grad()

            # Show the current loss in the progress bar.
            progress_bar.set_description(
                f"loss: {loss.item():.4f} avg_loss: {average_loss:.4f}"
                + "".join(
                    [f" {k}: {v:.4f}" for k, v in current_evaluation_results.items()]
                )
            )
            average_loss_cumulative += loss.item()

            # To help visualize training, periodically sample from the
            # diffusion model to see how well its doing.
            if step % save_and_sample_every_n == 0:
                current_evaluation_results = evaluator.evaluate(
                    model=model,
                    dataloader=dataloader,
                    accelerator=accelerator,
                )
                save(model, step, loss, optimizer, config, output_path=OUTPUT_NAME)
                average_loss = average_loss_cumulative / float(save_and_sample_every_n)
                average_loss_cumulative = 0.0

            # Update the current step.
            step += 1

            # Update the training progress bar in the console.
            progress_bar.update(1)

    # Final results.
    current_evaluation_results = evaluator.evaluate(
        model=model,
        dataloader=dataloader,
        accelerator=accelerator,
    )
    save(model, step, loss, optimizer, config, output_path=OUTPUT_NAME)


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


def save(
    model: torch.nn.Module,
    step,
    loss,
    optimizer: torch.optim.Optimizer,
    config: DotConfig,
    output_path: str,
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
        f"{output_path}/model-{step}.pt",
    )
