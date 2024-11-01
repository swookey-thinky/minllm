from accelerate import Accelerator, DataLoaderConfiguration
import argparse
import os
from pathlib import Path
import tiktoken
import torch
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm
from typing import List, Optional

from minllm.utils import cycle, instantiate_from_config, load_yaml, DotConfig
from minllm.datasets import tinyshakespeare


OUTPUT_PATH = "output/samples"


def sample(
    config_path: str,
    checkpoint_path: str,
    tokens_to_generate: int,
    initial_text: str,
    mixed_precision: str,
):
    num_samples = 1

    # Open the model configuration
    config = load_yaml(config_path)
    model = instantiate_from_config(config.model, use_config_struct=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    _, _ = model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    summary(
        model=model,
        input_size=(
            num_samples,
            config.model.params.context_length,
        ),
        dtypes=[torch.int64],
    )

    # The accelerate library will handle of the GPU device management for us.
    mixed_precision = (
        mixed_precision if mixed_precision != "" else config.training.mixed_precision
    )

    print(f"Sampling with mixed precision: {mixed_precision}")
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(split_batches=False),
        mixed_precision=mixed_precision,
    )

    # Move the model and the optimizer to the accelerator as well.
    model = accelerator.prepare(model)
    device = next(model.parameters()).device

    # Load a tokenizer if we have one
    tokenizer = None
    if "tokenizer" in config:
        tokenizer = instantiate_from_config(config.tokenizer.to_dict())
        encode = lambda s: tokenizer.encode(s)
        decode = lambda t: tokenizer.decode(t)
    else:
        token_encoder = tiktoken.get_encoding("gpt2")
        encode = lambda s: token_encoder.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: token_encoder.decode(l)

    # Encode the initial text
    start_ids = encode(initial_text)

    # Batch the input (batch size of 1)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    # run generation
    num_samples = 1
    with torch.no_grad():
        for k in range(num_samples):
            y = generate(
                model=model,
                idx=x,
                context_length=config.model.params.context_length,
                max_new_tokens=tokens_to_generate,
            )
            print(decode(y[0].tolist()).replace("</w>", " "))
            print("---------------")


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    context_length: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    model.eval()

    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= context_length else idx[:, -context_length:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")
        # apply softmax to convert logits to (normalized) probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def main(override=None):
    """
    Main entrypoint for the standalone version of this package.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokens_to_generate", type=int, default=1024)
    parser.add_argument("--initial_text", type=str, default="")
    parser.add_argument("--mixed_precision", type=str, default="")
    args = parser.parse_args()

    sample(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint,
        tokens_to_generate=args.tokens_to_generate,
        initial_text=args.initial_text,
        mixed_precision=args.mixed_precision,
    )


if __name__ == "__main__":
    main()
