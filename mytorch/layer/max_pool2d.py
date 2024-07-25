from mytorch import Tensor
from mytorch.layer import Layer
from mytorch.util import initializer

import numpy as np

class MaxPool2d(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1)) -> None:
        super()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        "TODO: implement forward pass"
        batch_size, channels, height, width = x.data.shape

        out_height = (height - self.kernel_size[0]) // self.stride[0] + 1
        out_width = (width - self.kernel_size[1]) // self.stride[1] + 1
        out = Tensor(
            data = initializer([batch_size, channels, out_height, out_width], "zero"),
            requires_grad=x.requires_grad
        )
        for c in range(channels):
            for l in range(x.shape[0]):
                for i in range(out_height):
                    for j in range(out_width):
                        region = x[l, c, i*self.stride[0]:i*self.stride[0]+self.kernel_size[0], j*self.stride[1]:j*self.stride[1]+self.kernel_size[1]]
                        max_i,max_j = np.unravel_index(region.data.argmax(), region.shape) # Find the maximum value in each channel for each data point
                        max_tensor = region[max_i,max_j]
                        out[l, c, i, j] = max_tensor
        return out
    
    def __str__(self) -> str:
        return "max pool 2d - kernel: {}, stride: {}, padding: {}".format(self.kernel_size, self.stride, self.padding)
