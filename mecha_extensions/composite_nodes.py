from sd_mecha.extensions.builtin.merge_methods import train_difference_mask, add_opposite_mask, add_strict_opposite_mask
from sd_mecha import merge_method, Parameter, Return
from torch import Tensor


@merge_method
def train_difference(
    a: Parameter(Tensor),
    b: Parameter(Tensor),
    c: Parameter(Tensor),
    alpha: Parameter(Tensor) = 1.0,
) -> Return(Tensor):
    return a + (b-c) * train_difference_mask.__wrapped__(a, b, c, alpha)


@merge_method
def add_opposite(
    a: Parameter(Tensor),
    b: Parameter(Tensor),
    c: Parameter(Tensor),
    alpha: Parameter(Tensor) = 1.0,
) -> Return(Tensor):
    return a + (b-c) * add_opposite_mask.__wrapped__(a, b, c, alpha)


@merge_method
def clamped_add_opposite(
    a: Parameter(Tensor),
    b: Parameter(Tensor),
    c: Parameter(Tensor),
    alpha: Parameter(Tensor) = 1.0,
) -> Return(Tensor):
    return a + (b-c) * add_strict_opposite_mask.__wrapped__(a, b, c, alpha)
