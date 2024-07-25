import numpy as np
from mytorch import Tensor, Dependency

def leaky_relu(x: Tensor) -> Tensor:
    """
    TODO: implement leaky_relu function.
    fill 'data' and 'req_grad' and implement LeakyRelu grad_fn
    hint: use np.where like Relu method but for LeakyRelu
    """

    data = data = np.maximum(0.05 * x.data, x.data)
    req_grad = x.requires_grad

    if req_grad: 
        def grad_fn(grad: np.ndarray):
             return grad * np.where(data > 0, 1, 0.05)

        depends_on = [Dependency(x, grad_fn)]
    else:
        depends_on = []

    return Tensor(data=data, requires_grad=req_grad, depends_on=depends_on)
