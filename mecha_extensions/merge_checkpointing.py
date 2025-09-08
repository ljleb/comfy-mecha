import torch
from typing import TypeVar
from sd_mecha import merge_method, Parameter, Return, StateDict


T = TypeVar("T")


@merge_method
def merge_checkpointing(
    a: Parameter(StateDict[T]),
    **kwargs,
) -> Return(T):
    key = kwargs["key"]
    cache = kwargs.get("cache")
    if cache is None:
        return a[key]

    if key not in cache:
        value = a[key]
        cache[key] = {}
        if isinstance(value, torch.Tensor):
            cache[key]["device"] = value.device
            if value.is_floating_point():
                cache[key]["dtype"] = value.dtype
                cache[key]["value"] = value.to(device="cpu", dtype=torch.float16)
            else:
                cache[key]["value"] = value.to(device="cpu")
        else:
            cache[key]["value"] = value
        return value

    key_cache = cache[key]
    res = key_cache["value"]
    if isinstance(res, torch.Tensor):
        to_kwargs = {}
        if "device" in key_cache:
            to_kwargs["device"] = key_cache["device"]
        if "dtype" in key_cache:
            to_kwargs["dtype"] = key_cache["dtype"]
        if to_kwargs:
            res = res.to(**to_kwargs)

    return res
