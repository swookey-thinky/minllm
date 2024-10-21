from minllm.utils import (
    DotConfig,
    instantiate_from_config,
    instantiate_partial_from_config,
)


def estimate_model_size(config: DotConfig, **kwargs):
    from accelerate import init_empty_weights

    with init_empty_weights():
        model = instantiate_partial_from_config(
            config.multiple_choice_model, use_config_struct=True
        )(
            base_language_model=instantiate_from_config(
                config.base_language_model, use_config_struct=True
            ),
            **kwargs,
        )
    from accelerate.utils import calculate_maximum_sizes

    total_size, largest_layer = calculate_maximum_sizes(model)
    print(f"Total Size: {total_size} largest_layer: {largest_layer}")
