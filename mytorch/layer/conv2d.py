from mytorch import Tensor
from mytorch.layer import Layer
from mytorch.util import initializer

import numpy as np

class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), need_bias: bool = False, mode="xavier") -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.need_bias = need_bias
        self.weight: Tensor = None
        self.bias: Tensor = None
        self.initialize_mode = mode

        self.initialize()

    def forward(self, x: Tensor) -> Tensor:
        
        "TODO: implement forward pass"
        
        if (self.padding is not None):
            input= Tensor(
                data=initializer([x.data.shape[0], self.out_channels, x.data.shape[2] + 2*self.padding[0], x.data.shape[3] + 2*self.padding[1]], "zero"),
                requires_grad=x.requires_grad
            )
            input[:,:,self.padding[0]:-self.padding[0],self.padding[1]:-self.padding[1]] = x
            x = input

        ho = (x.data.shape[2] - (self.padding is None)*self.kernel_size[0]) // self.stride[0]
        wo = (x.data.shape[3] - (self.padding is None)*self.kernel_size[1]) // self.stride[1]
        output = Tensor(
            data=initializer([x.data.shape[0], self.out_channels, ho, wo], "zero"),
            requires_grad=x.requires_grad
        )
        # Perform convolution for each channel
        for i in range(self.out_channels):
            for h in range(0, x.data.shape[2] - (self.padding is None)*self.kernel_size[0], self.stride[0]):
                for w in range(0, x.data.shape[3] - (self.padding is None)*self.kernel_size[1], self.stride[1]):
                    o = (x[:, :, h:h+self.kernel_size[0], w:w+self.kernel_size[1]] * self.weight[i,:,:,:]).sum_axis((1,2,3))+ self.bias[i]
                    output[:, i, h // self.stride[0], w // self.stride[1]] = o
        return output
    

    
    def initialize(self):
        "TODO: initialize weights"
        self.weight = Tensor(
            data=initializer([self.out_channels, self.in_channels, *self.kernel_size], self.initialize_mode),
            requires_grad=True
        )
        if self.need_bias:
            self.bias = Tensor(
                data=initializer([self.out_channels], "zero"),
                requires_grad=True
            )
        else:
            self.bias = Tensor(
                data=initializer([self.out_channels], "zero"),
                requires_grad=False
            )



    def zero_grad(self):
        "TODO: implement zero grad"
        self.weight.zero_grad()
        if self.need_bias:
            self.bias.zero_grad()

    def parameters(self):
        "TODO: return weights and bias"
        return [self.weight, self.bias]
    
    def __str__(self) -> str:
        return "conv 2d - total params: {} - kernel: {}, stride: {}, padding: {}".format(
                                                                                    self.kernel_size[0] * self.kernel_size[1],
                                                                                    self.kernel_size,
                                                                                    self.stride, self.padding)
