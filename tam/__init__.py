"""TAM: Token Activation Map for Qwen3-VL visual explainability."""

from .config import SPECIAL_IDS, SPATIAL_MERGE_SIZE


def __getattr__(name):
    """Lazy imports to avoid circular dependencies."""
    if name in ('TAM', 'rank_gaussian_filter', 'least_squares'):
        from . import tam_core
        return getattr(tam_core, name)
    if name in ('load_model', 'prepare_inputs', 'generate_with_logits', 'get_vision_shape'):
        from . import model_utils
        return getattr(model_utils, name)
    raise AttributeError(f"module 'tam' has no attribute {name!r}")
