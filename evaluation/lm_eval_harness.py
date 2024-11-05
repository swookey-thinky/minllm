"""Harness for evaluating local models with Eleuther LM Evaluation Harness

See https://github.com/EleutherAI/lm-evaluation-harness for details
and supported tasks.
"""

import argparse
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import MODEL_REGISTRY
import sys

from minllm.evaluation.harness import LMEvalLanguageModel


def main(override=None):
    """
    Main entrypoint for the standalone version of this package.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model_weights_checkpoint", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--task", type=str, default="hellaswag")

    args = parser.parse_args()

    # Register the model so the harness can instantiate it
    MODEL_NAME = "minllm"
    MODEL_REGISTRY[MODEL_NAME] = LMEvalLanguageModel

    sys.argv = [
        "cli_evaluate",
        "--model",
        "minllm",
        "--model_args",
        f"checkpoint_path={args.model_weights_checkpoint}",
        "--tasks",
        args.task,
        "--batch_size",
        str(args.batch_size),
        "--device",
        args.device,
    ]
    cli_evaluate()


if __name__ == "__main__":
    main()
