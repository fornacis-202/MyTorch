import numpy as np
from mytorch import Tensor, Dependency

def tanh(x: Tensor) -> Tensor:
    """
    TODO: (optional) implement tanh function
    hint: you can do it using function you've implemented (not directly define grad func)
    """

    exp = t.exp()
    neg_exp = t.__neg__().exp()
    return exp.__sub__(neg_exp).__mul__(exp.__add__(neg_exp).__pow__(-1))


