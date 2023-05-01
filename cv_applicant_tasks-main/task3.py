"""
develop a model based on the onnx file in model/model.onnx 

Note:
    - initialize the convolutions layer with uniform xavier
    - initialize the linear layer with a normal distribution (mean=0.0, std=1.0)
    - initialize all biases with zeros
    - use batch norm wherever is relevant
    - use random seed 8
    - use default values for anything unspecified
"""

import numpy as np
import torch
import torch.nn as nn
import onnxruntime
import onnx


torch.manual_seed(8)    # DO NOT MODIFY!
np.random.seed(8)   # DO NOT MODIFY!


# write your code here ...

# The model.onnx file was viewed in https://netron.app.
# Using the obtained graph visualisation, the following model was built.

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Sigmoid activation
        self.s = nn.Sigmoid()

        # Batch normalisation
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        # Max pool layer
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # Convolution Layers initialized with uniform xavier and all biases initialized with zeros
        self.conv1 = nn.Conv2d(3, 32, stride=1, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)

        self.conv2 = nn.Conv2d(32, 64, stride=2, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

        self.conv3 = nn.Conv2d(64, 64, stride=1, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

        self.conv4 = nn.Conv2d(64, 128, stride=2, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.conv4.weight)
        nn.init.zeros_(self.conv4.bias)

        self.conv5 = nn.Conv2d(128, 64, stride=1, kernel_size=1, padding=0)
        nn.init.xavier_uniform_(self.conv5.weight)
        nn.init.zeros_(self.conv5.bias)

        self.conv6 = nn.Conv2d(64, 64, stride=1, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.conv6.weight)
        nn.init.zeros_(self.conv6.bias)

        self.conv7 = nn.Conv2d(64, 64, stride=1, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.conv7.weight)
        nn.init.zeros_(self.conv7.bias)

        self.conv8 = nn.Conv2d(64, 64, stride=1, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.conv8.weight)
        nn.init.zeros_(self.conv8.bias)

        self.conv9 = nn.Conv2d(64, 64, stride=1, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.conv9.weight)
        nn.init.zeros_(self.conv9.bias)

        self.conv10 = nn.Conv2d(128, 64, stride=1, kernel_size=1, padding=0)
        nn.init.xavier_uniform_(self.conv10.weight)
        nn.init.zeros_(self.conv10.bias)

        self.conv11 = nn.Conv2d(256, 256, stride=1, kernel_size=1, padding=0)
        nn.init.xavier_uniform_(self.conv11.weight)
        nn.init.zeros_(self.conv11.bias)

        self.conv12 = nn.Conv2d(256, 128, stride=1, kernel_size=1, padding=0)
        nn.init.xavier_uniform_(self.conv12.weight)
        nn.init.zeros_(self.conv12.bias)

        self.conv13 = nn.Conv2d(128, 128, stride=2, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.conv13.weight)
        nn.init.zeros_(self.conv13.bias)

        self.conv14 = nn.Conv2d(256, 128, stride=1, kernel_size=1, padding=0)
        nn.init.xavier_uniform_(self.conv14.weight)
        nn.init.zeros_(self.conv14.bias)

        # Linear Layer initialized with a normal distribution (mean=0.0, std=1.0)
        self.fc1 = nn.Linear(256, 256)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=1.0)
        nn.init.zeros_(self.fc1.bias)

    def forward(self, x):
        x = torch.mul(self.conv1(x), self.s(self.bn1(self.conv1(x))))
        x = torch.mul(self.conv2(x), self.s(self.bn2(self.conv2(x))))
        x = torch.mul(self.conv3(x), self.s(self.bn2(self.conv3(x))))
        x = torch.mul(self.conv4(x), self.s(self.bn3(self.conv4(x))))
        l1 = torch.mul(self.conv10(x), self.s(self.bn2(self.conv10(x))))
        r1 = torch.mul(self.conv5(x), self.s(self.bn2(self.conv5(x))))
        x1 = torch.mul(self.conv6(r1), self.s(self.bn2(self.conv6(r1))))
        r2 = torch.mul(self.conv7(x1), self.s(self.bn2(self.conv7(x1))))
        x1 = torch.mul(self.conv8(r2), self.s(self.bn2(self.conv8(r2))))
        r3 = torch.mul(self.conv9(x1), self.s(self.bn2(self.conv9(x1))))
        x = torch.cat((r3, r2, r1, l1), 1)
        l2 = torch.mul(self.conv11(x), self.s(self.conv11(x)))
        r4 = torch.mul(self.conv12(l2), self.s(self.bn3(self.conv12(l2))))
        r4 = torch.mul(self.conv13(r4), self.s(self.bn3(self.conv13(r4))))
        l2 = self.bn4(self.maxpool(l2))
        l2 = torch.mul(self.conv14(l2), self.s(self.bn3(self.conv14(l2))))
        x = torch.cat((r4, l2), 1)
        x = torch.permute(x, (0, 2, 3, 1))
        x = torch.reshape(x, (-1, 256))
        x = self.fc1(x)
        x = torch.reshape(x, (1, 20, 40, 256))
        x = torch.permute(x, (0, 3, 1, 2))
        x = self.s(self.bn4(x))
        return x


# Load the custom model "MyModel()" which is based on the given model.onnx
model = MyModel()
x = torch.randn(1, 3, 160, 320)

# Output of the custom model for input x
custom_model_output = model(x)
print("Output of the custom model for input x: \n\n", custom_model_output)

# Load the model.onnx file
x = x.numpy()
model_onnx = onnx.load("model/model.onnx")
onnx.checker.check_model(model_onnx)
session = onnxruntime.InferenceSession('model/model.onnx')
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
res = session.run([output_name], {input_name: x})

# Output of model.onnx for input x
onnx_model_output = torch.from_numpy(np.array(res).reshape((1, 256, 20, 40)))
# print("Output of model.onnx for input x: \n\n", onnx_model_output)

# check if both outputs are equal
# print(torch.isclose(custom_model_output, onnx_model_output))


