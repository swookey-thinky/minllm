import argparse

from minllm.training.train import train


def main(override=None):
    """
    Main entrypoint for the standalone version of this package.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--num_training_steps", type=int, default=-1)
    parser.add_argument("--mixed_precision", type=str, default="")
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument("--resume_from_checkpoint", type=str, default="")
    parser.add_argument("--no_eval", action="store_true")

    args = parser.parse_args()

    train(
        config_path=args.config_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        num_training_steps=args.num_training_steps,
        mixed_precision=args.mixed_precision,
        resume_from_checkpoint=args.resume_from_checkpoint,
        disable_evaluation=args.no_eval,
    )


if __name__ == "__main__":
    main()
