import argparse
import os

from minllm.tokenizer.bpe import simple

OUTPUT_PATH = "output/tokenizers/bpe"


def tokenize(dataset: str, vocab_size: int):
    assert dataset in ["wikipedia"], f"Dataset {dataset} not found."
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    if dataset == "wikipedia":
        _tokenize_wikipedia(vocab_size=vocab_size)


def _tokenize_wikipedia(vocab_size: int):
    from minllm.datasets.wikipedia import Wikipedia2020English

    dataset = Wikipedia2020English(".", split="0")
    data = [dataset[i] for i in range(len(dataset))]
    tokenizer = simple.BasicTokenizer()
    tokenizer.train(texts=data, vocab_size=vocab_size)
    tokenizer.save(file_prefix="bpe_basic", output_path=OUTPUT_PATH)


def main(override=None):
    """
    Main entrypoint for the standalone version of this package.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wikipedia")
    parser.add_argument("--vocab_size", type=int, default=50257)
    args = parser.parse_args()

    tokenize(dataset=args.dataset, vocab_size=args.vocab_size)


if __name__ == "__main__":
    main()
