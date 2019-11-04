import torch

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.input_dim = input_dim
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        original_shape = x.shape
        outputs = self.linear(x.reshape(original_shape[0], original_shape[1]*original_shape[2]*original_shape[3]))
        return outputs
