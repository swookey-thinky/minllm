from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import GradientAccumulationPlugin
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader, RandomSampler
from torchinfo import summary
from tqdm.auto import tqdm
import torch.distributed as dist
from typing import Optional

from minllm.datasets.utils import load_dataset
from minllm.evaluation.evaluator import Evaluator
from minllm.schedules import (
    get_cosine_schedule_with_warmup,
    get_trapezoidal_schedule_with_warmup,
)
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
    resume_from_checkpoint: str = "",
    compile: Optional[bool] = None,
    disable_evaluation: bool = False,
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
        step_scheduler_with_optimizer=False,
    )

    # Create the model to train
    model = instantiate_from_config(config.model, use_config_struct=True)

    if batch_size <= 0:
        batch_size = config.training.batch_size
    if num_training_steps <= 0:
        num_training_steps = config.training.training_steps

    accelerator.print(
        summary(
            model=model,
            input_size=(
                batch_size,
                config.model.params.context_length,
            ),
            dtypes=[torch.int64],
            verbose=0,
        )
    )

    # If we are resuming from a checkpoint, load the model and optimizer states,
    # and set the surrent step to the last training step.
    step = 0
    if resume_from_checkpoint:
        checkpoint = torch.load(resume_from_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        step = checkpoint["step"]
        accelerator.print(f"Resuming training from step {step}.")

    # Move everything to the accelerator. First we move the model,
    # so that the optimizer can see the on-device parameters.
    model = accelerator.prepare(model)

    # Create the evaluator for measuring the training performance.
    evaluator = Evaluator(config=config.training.evaluation)

    # Load the training dataset. Make sure to do this on the main process first
    # since we only want to download the data once if we can.
    with accelerator.main_process_first():
        dataset = load_dataset(
            dataset,
            context_length=config.model.params.context_length,
            tokenizer=tokenizer,
        )

    # Use a RandomSampler with replacement, otherwise for large datasets,
    # the non-replacement version will try to create an incredibly large list
    # of random indices and most likely blow out our memory. For example,
    # OpenWebText has ~9b tokens in it, so attempting to create a 9b list of random
    # indices of int64 dtype will take 9*8~72GB of RAM, and if doing this on a multi-gpu
    # node, well, boom.
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=RandomSampler(data_source=dataset, replacement=True),
        pin_memory=True,
        num_workers=4,
    )

    # Move the dataloader to the device
    dataloader = accelerator.prepare(dataloader)

    # Now create the optimizer
    optimizer = configure_optimizers(
        model=model,
        config=config.training,
        accelerator=accelerator,
    )

    if resume_from_checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Create the learning rate schedule
    if "learning_rate_schedule" in config.training:
        assert config.training.learning_rate_schedule in ["cosine", "trapezoidal"]
        lr_scheduler = (
            get_trapezoidal_schedule_with_warmup(optimizer, config=config)
            if config.training.learning_rate_schedule == "trapezoidal"
            else get_cosine_schedule_with_warmup(optimizer, config=config)
        )
    else:
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, config=config)

    # Finally move the optimizer and the learning rate schedule to the device.
    optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)

    # We are going to train for a fixed number of steps, so set the dataloader
    # to repeat indefinitely over the entire dataset.
    dataloader = cycle(dataloader)

    # Setup gradient clipping to help stabilize training
    max_grad_norm = 1.0
    average_loss = 0.0
    average_loss_cumulative = 0.0
    save_and_sample_every_n = (
        save_and_sample_every_n
        if save_and_sample_every_n > 0
        else config.training.save_and_sample_every_n
    )

    # Compile the model if we are asked to
    do_compile = False
    if compile is not None:
        do_compile = compile
    else:
        if "training" in config and "compile" in config.training:
            do_compile = config.training.compile
    accelerator.print(f"Model compilation setting: {do_compile}")
    if do_compile:
        model = torch.compile(model)

    current_evaluation_results = {}
    with tqdm(initial=step, total=num_training_steps, disable=not accelerator.is_main_process) as progress_bar:
        # Perform gradient descent for the given number of training steps.
        while step < num_training_steps:
            # All of the gradient accumulation steps count as one training step.
            for _ in range(gradient_accumulation_steps):
                with accelerator.accumulate(model):
                    # The training data are the tokens at position i, and the targets
                    # are the tokens at position i+1 (the next tokens)
                    x, y = next(dataloader)

                    # Calculate the loss on the batch of training data.
                    with accelerator.autocast():
                        _, _, loss = model(x, targets=y)

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
                            [
                                f" {k}: {v:.4f}"
                                for k, v in current_evaluation_results.items()
                            ]
                        )
                    )
                    average_loss_cumulative += loss.detach().cpu().item()

            # To help visualize training, periodically sample from the
            # diffusion model to see how well its doing.
            if step % save_and_sample_every_n == 0:
                # Only barrier if we are running distributed.
                if dist.is_initialized():
                    torch.distributed.barrier()

                if not disable_evaluation:
                    current_evaluation_results = evaluator.evaluate(
                        model=model,
                        dataloader=dataloader,
                        accelerator=accelerator,
                    )

                # Only save the model from the main process.
                if accelerator.is_main_process:
                    save(accelerator.unwrap_model(model), step, loss, optimizer, config, output_path=OUTPUT_NAME)
                average_loss = average_loss_cumulative / float(
                    save_and_sample_every_n * gradient_accumulation_steps
                )
                average_loss_cumulative = 0.0

            # Update the current step.
            step += 1

            # Update the training progress bar in the console.
            progress_bar.update(1)

    # Final results.
    if dist.is_initialized():
        torch.distributed.barrier()

    if not disable_evaluation:
        current_evaluation_results = evaluator.evaluate(
            model=model,
            dataloader=dataloader,
            accelerator=accelerator,
        )

    if accelerator.is_main_process:
        save(accelerator.unwrap_model(model), step, loss, optimizer, config, output_path=OUTPUT_NAME)

    accelerator.end_training()

def configure_optimizers(model: torch.nn.Module, config: DotConfig, accelerator: Accelerator):
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
    accelerator.print(
        f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    accelerator.print(
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
