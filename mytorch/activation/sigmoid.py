import numpy as np
from mytorch import Tensor, Dependency

def sigmoid(x: Tensor) -> Tensor:
    """
    TODO: implement sigmoid function
    hint: you can do it using function you've implemented (not directly define grad func)
    """
    x_neg = x.__neg__()
    exp = x_neg.exp()
    exp = exp.__add__(Tensor(1.0))
    return exp.__pow__(-1)
    