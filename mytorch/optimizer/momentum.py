from mytorch.optimizer import Optimizer
import numpy as np

"TODO: (optional) implement Momentum optimizer"
class Momentum(Optimizer):
    def __init__(self, layers, learning_rate=0.01, momentum=0.9):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.momentum = momentum

        for layer in self.layers:
            layer.velocity_w = np.zeros_like(layer.weight)
            if layer.need_bias:
                layer.velocity_b = np.zeros_like(layer.bias)

    def step(self):
        for layer in self.layers:

            layer.velocity_w = self.momentum * layer.velocity_w - (1-self.momentum) * self.learning_rate * layer.weight.grad
            layer.weight = layer.weight + layer.velocity_w

            if layer.need_bias:
                layer.velocity_b = self.momentum * layer.velocity_b - (1-self.momentum) * self.learning_rate * layer.bias.grad
                layer.bias = layer.bias + layer.velocity_b