import torch
import torch.nn as nn

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, std: float = 0.1):
        """
        Initialize a linear layer with random weights.

        The weights and biases are registered as parameters, allowing for 
        gradient computation and update during backpropagation.
        """
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        weight = torch.randn(in_features, out_features, requires_grad=True) * std
        bias = torch.zeros(out_features, requires_grad=True)
        
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        self.to(device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform linear transformation by multiplying the input tensor
        with the weights matrix, and adding the biases.
        """
        return x @ self.weight + self.bias

    def __repr__(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


class ReLU(nn.Module):
    """
    Rectified Linear Unit (ReLU) activation function.
    """

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return torch.clip(x, 0.)


class Dropout(nn.Module):
    """
    Applies the dropout regularization technique to the input tensor.

    During training, randomly sets a fraction of input units to 0 with probability `p`,
    scaling the remaining values by `1 / (1 - p)` to maintain the same expected output sum.
    During evaluation, no dropout is applied.
    """

    def __init__(self, p=0.2):
        super(Dropout, self).__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            mask = (torch.rand(x.shape) > self.p).float().to(x) / (1 - self.p)
            return x * mask
        return x


class Flatten(nn.Module):
    """
    Reshape the input tensor by flattening all dimensions except the first dimension.
    """

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        """
        x.view(x.size(0), -1) reshapes the x tensor to (x.size(0), N)
        where N is the product of the remaining dimensions.
        E.g. (batch_size, 28, 28) -> (batch_size, 784)
        """
        return x.view(x.size(0), -1)


class Sequential(nn.Module):
    """
    Sequential container for stacking multiple modules,
    passing the output of one module as input to the next.
    """

    def __init__(self, *layers):
        super(Sequential, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self) -> str:
        layer_str = '\n'.join([f' ({i}): {layer}' for i, layer in enumerate(self.layers)])
        return f'{self.__class__.__name__}(\n{layer_str}\n)'


class Classifier(nn.Module):
    """
    Classifier model consisting of a sequence of linear layers and ReLU activations,
    followed by a final linear layer that outputs logits (unnormalized scores)
    for each of the 10 garment classes.
    """

    def __init__(self):
        """
        The output logits of the last layer can be passed directly to
        a loss function like CrossEntropyLoss, which will apply the 
        softmax function internally to calculate a probability distribution.
        """
        super(Classifier, self).__init__()
        self.labels = ['T-shirt/Top', 'Trouser/Jeans', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle-Boot']
        
        self.main = Sequential(
            Flatten(),
            Linear(in_features=784, out_features=256),
            ReLU(),
            Dropout(0.2),
            Linear(in_features=256, out_features=64),
            ReLU(),
            Dropout(0.2),
            Linear(in_features=64, out_features=10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)
    
    def predictions(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.nn.functional.softmax(logits, dim=1)
            predictions = dict(zip(self.labels, probs.cpu().detach().numpy().flatten()))    
        return predictions