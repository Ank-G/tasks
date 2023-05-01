"""
Write a code using pytorch to replicate a grouped 2D convolution layer based on the original 2D convolution. 

The common way of using grouped 2D convolution layer in Pytorch is to use 
torch.nn.Conv2d(groups=n), where n is the number of groups.

However, it is possible to use a stack of n torch.nn.Conv2d(groups=1) to replicate the same
result. The wights must be copied and be split between the convs in the stack.

You can use:
    - use default values for anything unspecified  
    - all available functions in NumPy and Pytorch
    - the custom layer must be able to take all parameters of the original nn.Conv2d 
"""

import numpy as np
import torch
import torch.nn as nn


torch.manual_seed(8)    # DO NOT MODIFY!
np.random.seed(8)   # DO NOT MODIFY!

# random input (batch, channels, height, width)
x = torch.randn(2, 64, 100, 100)

# original 2d convolution
grouped_layer = nn.Conv2d(64, 128, 3, stride=1, padding=1, groups=16, bias=True)

# weights and bias
w_torch = grouped_layer.weight
b_torch = grouped_layer.bias

y = grouped_layer(x)

# now write your custom layer


class CustomGroupedConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, groups=1, bias=True):
        super(CustomGroupedConv2D, self).__init__()

        self.groups = groups
        self.conv_list = nn.ModuleList()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Split the input and output channels into groups
        self.in_channels_per_group = self.in_channels // self.groups
        self.out_channels_per_group = self.out_channels // self.groups

        # Stack the n convolutions (where n is the no. of groups)
        for i in range(groups):
            self.conv_list.append(nn.Conv2d(self.in_channels_per_group, self.out_channels_per_group, kernel_size,
                                            stride=stride, padding=padding, groups=1, bias=bias))

        # Copy and split the weights and biases among each convolution in the stack.
        iterable = np.arange(self.out_channels)
        for conv, i in zip(self.conv_list, iterable):
            conv.weight.data = w_torch.data[i * self.out_channels_per_group: (i + 1) * self.out_channels_per_group, :, :, :]
            conv.bias.data = b_torch.data[i * self.out_channels_per_group: (i + 1) * self.out_channels_per_group]

    def forward(self, x):
        # Split the input into groups
        x_split = torch.split(x, x.size(1) // self.groups, dim=1)

        # Perform grouped 2D convolution
        y_new = []
        for conv, x_i in zip(self.conv_list, x_split):
            y_new.append(conv(x_i))
        return torch.cat(y_new, dim=1)


# the output of CustomGroupedConv2D(x) must be equal to grouped_layer(x)
custom_layer = CustomGroupedConv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, groups=16, bias=True)
y_custom = custom_layer(x)

# check that the output of the custom layer is equal to the output of the original layer
print(torch.eq(y, y_custom))    # Returns True



        
