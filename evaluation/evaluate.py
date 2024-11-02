import argparse
from accelerate import Accelerator, DataLoaderConfiguration
import torch
from torch.utils.data import DataLoader

from minllm.datasets.utils import load_dataset
from minllm.evaluation.evaluator import Evaluator
from minllm.utils import DotConfig, load_yaml, instantiate_from_config, cycle


def main(override=None):
    """
    Main entrypoint for the standalone version of this package.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--mixed_precision", type=str, default="")
    parser.add_argument("--model_weights_checkpoint", type=str, default="")

    args = parser.parse_args()

    # Load the config
    config = load_yaml(args.config_path)

    # Create the evaluation dataset
    dataset = config.training.evaluation.dataset

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

    dataloader = DataLoader(
        dataset,
        batch_size=config.training.evaluation.samples_per_batch,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )

    if args.model_weights_checkpoint:
        checkpoint = torch.load(args.model_weights_checkpoint, map_location="cpu")
        model = instantiate_from_config(
            DotConfig(checkpoint["config"]).model, use_config_struct=True
        )
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model = instantiate_from_config(config.model, use_config_struct=True)

    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(split_batches=False),
        mixed_precision=(
            args.mixed_precision
            if args.mixed_precision != ""
            else config.training.mixed_precision
        ),
    )

    model, dataloader = accelerator.prepare(model, dataloader)
    dataloader = cycle(dataloader)

    current_evaluation_results = evaluator.evaluate(
        model=model,
        dataloader=dataloader,
        accelerator=accelerator,
    )
    print(f"Evaluation results: {current_evaluation_results}")


if __name__ == "__main__":
    main()
