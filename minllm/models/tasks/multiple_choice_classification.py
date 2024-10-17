"""Wrap a language model to perform a multiple choice classification task.

This classification task corresponds to the multiple choice task
defined in GPT-1 paper: "Improving Language Understanding by Generative Pre-Training"

The input token formulation for a single example looks like:

<start>context<delim>answer 1<classify>
<start>context<delim>answer 2<classify>
<start>context<delim>answer 3<classify>

For example, if there are 2 possible answers (A), the shape of the input
is: (B, A, L) where L is the context length and A is the number of possible answers.
"""

from einops import rearrange
import torch

from minllm.utils import DotConfig


class Classifier(torch.nn.Module):
    def __init__(self, config: DotConfig, base_language_model: torch.nn.Module):
        super().__init__()

        self._config = config
        self._base_language_model = base_language_model
        self._classifier_token = config.classifier_token
        self._num_choices = config.num_choices

        # Inner dimensionality of the base language model
        self._embedding_dim = config.embedding_dim

        # Classification head, for true/false of if the answer token
        # is correct for the question.
        self._classifier = torch.nn.Linear(config.embedding_dim, 1)

    def forward(self, idx: torch.Tensor):
        # idx is shape (B, A, L), so reshape to
        # (B*A, L) so that we can process through the language model
        _, A, _ = idx.shape
        h = rearrange(idx, "b a l -> (b a) l")

        # Send through the language model
        _, lm_h = self._base_language_model(h)

        B, L, D = lm_h.shape

        # For the language modeling task, we don't care about the next token
        # prediction, so calculate the logits based on the original tokens, by removing
        # the last token from the output
        # lm_logits will be shape (B, L-1, D)
        lm_logits = self._base_language_model.logits(lm_h[:, :-1])
        print(f"lm_h: {lm_h.shape}")
        print(f"lm_logits: {lm_logits.shape}")

        # Now calculate the classification task, starting with the language
        # model head. First flatten the language model head.
        clf_h = torch.reshape(lm_h, shape=(-1, D))

        # Find all of the token indices that correspond to the classifier token.
        # (B*A,)
        pool_idx = torch.argmax(
            torch.eq(h, torch.tensor(self._classifier_token, device=idx.device)).to(
                torch.float32
            ),
            dim=1,
        ).to(torch.int64)

        # Gather all of the classification tokens from the output
        # (B*A, D)
        batch_index = (
            torch.arange(B, dtype=torch.int64, device=idx.device) * L + pool_idx
        )

        clf_h = clf_h[batch_index]

        # Classification logits
        # (B*A, 1)
        clf_logits = self._classifier(clf_h)

        # Convert to (B, A)
        clf_logits = rearrange(clf_logits, "(b a) 1 -> b a", a=A)

        # Return the language model logits and the classification logits
        return lm_logits, clf_logits
