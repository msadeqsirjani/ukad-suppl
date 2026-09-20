from .base_sampler import ISampler
from .source_sampler import CSVSampler

from src.utils.serialization_utils import create_object


def get(identifier, **kwargs):
    obj = create_object(
        identifier,
        module_objects={"CSVSampler": CSVSampler},
        **kwargs,
    )

    if isinstance(obj, ISampler):
        return obj
    raise ValueError(f"Could not interpret sampler instance: {obj}.")
