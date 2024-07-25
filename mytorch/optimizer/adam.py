from mytorch.optimizer import Optimizer
import numpy as np

"TODO: (optional) implement Adam optimizer"
class Adam(Optimizer):
    def __init__(self, layers, learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(layers)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon


        for layer in self.layers:
            layer.m_w = np.zeros_like(layer.weight)
            layer.v_w = np.zeros_like(layer.weight)
            if layer.need_bias:
                layer.m_b = np.zeros_like(layer.bias)
                layer.v_b = np.zeros_like(layer.bias)

    def step(self):
        for layer in self.layers:
            layer.m_w = self.beta1 * layer.m_w + (1 - self.beta1) * layer.weight.grad.data
            layer.v_w = self.beta2 * layer.v_w + (1 - self.beta2) * layer.weight.grad.data**2

            layer.weight.data = layer.weight.data - self.learning_rate * layer.m_w / (np.sqrt(layer.v_w) + self.epsilon)

            if layer.need_bias:
                layer.m_b = self.beta1 * layer.m_b + (1 - self.beta1) * layer.bias.grad.data
                layer.v_b = self.beta2 * layer.v_b + (1 - self.beta2) * layer.bias.grad.data**2

                layer.bias.data = layer.bias.data - self.learning_rate * layer.m_b / (np.sqrt(layer.v_b ) + self.epsilon)
