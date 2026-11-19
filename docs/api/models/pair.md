# Pair Prediction Models

Models that score the relationship between two items from their embeddings.
Both are registered with the factory (`@factory.nn_module`) and are
serializable via `to_dict` / `from_dict`.

## PairPredictor

Predicts a scalar score for a pair of items given a context. Each item is
embedded through a shared item module, the context through a context module, and
the three embeddings are concatenated and passed through a small MLP to a single
logit.

::: embkit.models.pair.PairPredictor
    options:
      show_source: false
      members:
        - __init__
        - forward
        - to_dict
        - from_dict
      merge_init_into_class: true

## PairProbabilityPredictor

Predicts the probability that a pair of items are related, using only the two
item embeddings (no context). The two embeddings are concatenated and passed
through an MLP with layer normalization to a single output.

::: embkit.models.pair.PairProbabilityPredictor
    options:
      show_source: false
      members:
        - __init__
        - forward
        - to_dict
        - from_dict
      merge_init_into_class: true
