import torch
import torch.nn as nn

class MyDenseLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        # Initialize weights and bias
        super(MyDenseLayer, self).__init__()
        self.W = nn.Parameter(torch.randn(input_dim, output_dim, requires_grad=True))
        self.b = nn.Parameter(torch.randn(1, output_dim, requires_grad=True))

    def forward(self, inputs):
        # Forward propagte the inputs (Activate Functions - Sigmoid)
        z = torch.matmul(inputs, self.W) + self.b
        output = torch.sigmoid(z)
        return output

# 사실 이건 스스로 칠 필요는 없음 다른 방식이 존재함