# Building a Neural Network Classifier from the Ground Up: A Step-by-Step Guide

[![GitHub license](https://img.shields.io/github/license/dcarpintero/nn-image-classifier.svg)](https://github.com/dcarpintero/nn-image-classifier/blob/master/LICENSE)
[![GitHub contributors](https://img.shields.io/github/contributors/dcarpintero/nn-image-classifier.svg)](https://GitHub.com/dcarpintero/nn-image-classifier/graphs/contributors/)
[![GitHub issues](https://img.shields.io/github/issues/dcarpintero/nn-image-classifier.svg)](https://GitHub.com/dcarpintero/nn-image-classifier/issues/)
[![GitHub pull-requests](https://img.shields.io/github/issues-pr/dcarpintero/nn-image-classifier.svg)](https://GitHub.com/dcarpintero/nn-image-classifier/pulls/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

[![GitHub watchers](https://img.shields.io/github/watchers/dcarpintero/nn-image-classifier.svg?style=social&label=Watch)](https://GitHub.com/dcarpintero/nn-image-classifier/watchers/)
[![GitHub forks](https://img.shields.io/github/forks/dcarpintero/nn-image-classifier.svg?style=social&label=Fork)](https://GitHub.com/dcarpintero/nn-image-classifier/network/)
[![GitHub stars](https://img.shields.io/github/stars/dcarpintero/nn-image-classifier.svg?style=social&label=Star)](https://GitHub.com/dcarpintero/nn-image-classifier/stargazers/)


Classification is one of the fundamental deep learning tasks. While modern frameworks like PyTorch, JAX, Keras, and TensorFlow offer a convenient abstraction to build and train neural networks, crafting one from scratch provides a more comprehensive understanding of the nuances involved.

In this article, we will implement in Python the essential modules required to build and train a multilayer perceptron that classifies garment images. In particular, we will delve into the fundamentals of **approximation**, **non-linearity**, **regularization**, **gradients**, and **backpropagation**. Additionally, we explore the significance of **random parameter initialization** and the benefits of **training in mini-batches**.

By the end of this guide, you will be able to construct the building blocks of a neural network from scratch, understand how it learns, and deploy it to [HuggingFace Spaces](https://huggingface.co/spaces/dcarpintero/fashion-image-recognitionhttps://huggingface.co/spaces/dcarpintero/fashion-image-recognition) to classify real-world garment images.

<figure>
  <img style="margin: 0 auto; display: block;" src="https://cdn-uploads.huggingface.co/production/uploads/64a13b68b14ab77f9e3eb061/bvC2A3Cb2zn81h_neojtH.png">
  <figcaption style="text-align: center;">Garment Classifier deployed to HuggingFace Spaces</figcaption>
</figure>

##  Table of Contents

* 1. [The Intuition behind our Neural Network](#TheIntuitionbehindourNeuralNetwork)
* 2. [Architecture](#Architecture)
	* 2.1. [Linear Transformation](#LinearTransformation)
	* 2.2. [Introducing non-linearity](#Introducingnon-linearity)
	* 2.3. [Regularization](#Regularization)
	* 2.4. [Flatten Transformation](#FlattenTransformation)
	* 2.5. [Sequential Layer](#SequentialLayer)
	* 2.6. [Classifier Model](#ClassifierModel)
	* 2.7. [Gradient Descent Optimizer](#GradientDescentOptimizer)
	* 2.8. [Backpropagation](#Backpropagation)
* 3. [Training](#Training)
	* 3.1. [The Fashion Dataset](#TheFashionDataset)
	* 3.2. [Data Loaders for Mini-Batches](#DataLoadersforMini-Batches)
	* 3.3. [Fitting the Model](#FittingtheModel)
* 4. [Model Assessment](#ModelAssessment)
* 5. [Inference](#Inference)
* 6. [Resources](#Resources)
* 7. [References](#References)

##  1. <a name='TheIntuitionbehindourNeuralNetwork'></a>The Intuition behind our Neural Network

Our goal is to classify garment images by approximating a large mathematical function to a training dataset of such images. We will begin this process by randomly initializing the parameters of our function, and adjusting them to combine input pixel values, until we obtain favorable outputs in form of class predictions. This iterative method seeks to learn features in the training dataset that differentiate between classes.

The foundation for this approach lies in the [Universal Approximation Theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem), a fundamental concept that highlights the combination of linear transformations and non-linear functions to approximate complex patterns, such as those needed for computer vision.

The principle of teaching computers through examples, rather than explicit programming, dates back to Arthur Samuel in 1949 [1]. Samuel suggested the concept of using weights as function parameters that can be adjusted to influence a program’s behavior and outputs. And emphasized the idea of automating such a method that tests and optimizes these weights based on their performance in real tasks.

We will then implement a method to adjust weights automatically, applying [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) [2] in mini-batches. In practice, this involves the following steps:

1. Initialize weights and bias parameters.
2. Calculate predictions on a *mini-batch*.
3. Calculate the average loss between the predictions and the targets.
4. Calculate the *gradients* to get an indication of how the parameters need to change to minimize the loss.
5. Update the weights and bias parameters based on the gradients and a learning rate.
6. Repeat from step 2.
7. Stop the process once a condition is met, such as a time constraint or when the training/validation losses and metrics cease to improve.

***A mini-batch refers to a randomly selected subset of the training dataset** that is used to calculate the loss and update the weights in each iteration. The benefits of training in mini-batches are explained in the Training section*.

***Gradients** are a measure inferred from the derivative of a function that signals how the output of the function would change by modifying its parameters. Within the context of neural networks, they represent a vector that **indicates the direction and magnitude in which we need to change each weight to improve our model**.*

##  2. <a name='Architecture'></a>Architecture

In the following sections, we dive into the implementation details of the required components to build and train our multilayer perceptron. For simpler integration with advanced functionality such as gradient computation, these components will be defined as custom PyTorch modules.

###  2.1. <a name='LinearTransformation'></a>Linear Transformation

At the heart of our neural network are linear functions. These functions perform two key operations: (i) transformation of input values by their weights and bias parameters through matrix multiplication, and (ii) dimensionality reduction (or augmentation in some cases).

The transformation operation projects input values into a different space, which along the use of stacked linear layers, enables the network to progressively learn more abstract and complex patterns. Dimensionality reduction is achieved when the number of output units in a linear layer is smaller than the number of inputs. This compression forces the layer to capture the most salient features of the higher-dimensional input.

```python
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, std: float = 0.1):
        """
        Initialize a linear layer with random weights.

        Weights and biases are registered as parameters, allowing for 
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
        with the weight matrix, and adding the bias.
        """
        return x @ self.weight + self.bias
```

Note that **the weights are randomly initialized  according to a Gaussian distribution (`randn`) to break symmetry and enable effective learning**. If all parameters were first set to the same value, such as zeros, they will compute the same gradients during backpropagation, leading to identical weight updates and slower (or non)convergence.

Furthermore, **scaling weights is also a common practice in initialization**. This helps in controlling the variance, and can have a big impact on the training dynamics. We favour a relatively small scale value (`std=0.1`) since large values might lead to gradients increasing exponentially (and overflowing to NaN) during backpropagation, resulting in the *exploding gradients problem*.

###  2.2. <a name='Introducingnon-linearity'></a>Introducing non-linearity

Without non-linearity, no matter how many layers our neural network has, it would still behave like a single-layer perceptron. This is due to the fact that the composition of successive linear transformations is itself another linear transformation, which would prevent the model from approximating complex patterns.

To overcome this limitation, we adhere to the Universal Approximation Theorem and introduce non-linearity by implementing the Rectified Linear Unit (ReLU), a widely used and effective activation function that sets negative values to zero while preserving positive values. 

```python
class ReLU(nn.Module):
    """
    Rectified Linear Unit (ReLU) activation function.
    """

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return torch.clip(x, 0.)
```

The Rectified Linear Unit (ReLU) was proposed by Kunihiko Fukushima in 1969 within the context of visual feature extraction in hierarchical neural networks [3]. In 2011 [4], it was found to enable better training of deeper networks compared to the widely used activation functions *logistic sigmoid* and *hyperbolic tangent*.

###  2.3. <a name='Regularization'></a>Regularization

Regularization is a fundamental technique used to reduce *overfitting* in neural networks, which occurs when parameters become tuned to noise on invidual data points during training. A widely used and effective method of regularization is the *dropout* function, introduced by G. Hinton's research group in 2014 [5]. Dropout works by randomly deactivating a portion of the network's units during the training phase. This encourages each unit to contribute independently, preventing the model from becoming overly reliant on over-specialized single units and enhancing its ability to generalize to new data.

```python
class Dropout(nn.Module):
    """
    Applies the dropout regularization technique to the input tensor.

    During training, it randomly sets a fraction of input units to 0 with probability `p`,
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
```

###  2.4. <a name='FlattenTransformation'></a>Flatten Transformation

In deep learning, flattening images is necessary to convert multi-dimensional data into a one-dimensional (1D) array before feeding it into a classification model. Our training dataset, Fashion MNIST [6], is a collection of 60,000 grayscale images of size 28x28. We include a transformation to flatten these images in their width and height dimensions to reduce memory usage (multi-dimensional arrays take up additional memory overhead to manage their structure), and simplify the input for the model (each pixel becomes an individual unit).

```python
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
```

###  2.5. <a name='SequentialLayer'></a>Sequential Layer

To construct the full neural network architecture, we need a way to connect the individual linear operations and activation functions in a sequential manner, forming a feedforward path from the inputs to the outputs. This is achieved by using a sequential layer, which allows to define the specific order and composition of the various layers in our network.

```python
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
```

###  2.6. <a name='ClassifierModel'></a>Classifier Model

After flattening the input images, we stack linear operations with non-linear functions, enabling the network to learn hierarchical representations and patterns in the data. This is essential for our image classification task, where the network needs to capture visual features to distinguish between various classes.

```python
class Classifier(nn.Module):
    """
    Classifier model consisting of a sequence of linear layers and ReLU activations,
    followed by a final linear layer that outputs logits (unnormalized scores)
    for each of the 10 garment classes.

    It encapsulates also a method to convert logits into a label/probability dict.
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
```

The research paper *Visualizing and Understanding Convolutional Networks* [7] offers insights into a concept akin to hierarchical progressive learning, specifically applied to convolutional layers. This provides a comparable intuition to understand how stacked layers are capable of automatically learning features within images:

<figure>
  <img style="margin: 0 auto; display: block;" src="https://cdn-uploads.huggingface.co/production/uploads/64a13b68b14ab77f9e3eb061/CMgQHSSzEaEmsBd0D2eKD.png">
  <figcaption style="text-align: center;">Visualization of features in a convolutional neural network - https://arxiv.org/pdf/1311.2901.pdf</figcaption>
</figure>

###  2.7. <a name='GradientDescentOptimizer'></a>Gradient Descent Optimizer

We implement a basic optimizer to automatically adjust the neural network’s parameters, weights and biases, based on gradients. Computed during backpropagation, gradients indicate how to update these parameters to minimize the loss function. Using these gradients, the optimizer updates the parameters in a stepwise manner, with the step size determined by the learning rate.

```python
class Optimizer:
    """
    Update model parameters during training.
    
    It performs a simple gradient descent step by updating the parameters
    based on their gradients and the specified learning rate (lr).
    """

    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    def step(self):
        for p in self.params:
            p.data -= p.grad.data * self.lr

    def zero_grad(self):
        """
        Reset the gradients of all parameters to zero.
        Since PyTorch accumulates gradients, this method ensures that
        the gradients from previous optimization steps do not interfere
        with the current step.
        """
        for p in self.params:
            p.grad = None
```

###  2.8. <a name='Backpropagation'></a>Backpropagation

Introduced by [Paul Werbos](https://ieeexplore.ieee.org/author/37344537300) in 1974 [8], the concept of backpropagation for neural networks was almost entirely ignored for decades. However, it is nowadays recognized as one of the most important AI foundations.

At its core, backpropagation serves to calculate the gradients of the loss function with respect to each parameter in the network. This is achieved by applying the [chain rule of calculus](https://en.wikipedia.org/wiki/Chain_rule), systematically calculating these gradients from the output layer back to the input layer — hence the term *backpropagation*.

Under the hood, this method involves computing partial derivatives of a complex function, and maintaining a directed acyclic graph (DAG) that tracks the sequence of operations on the input data. To simplify this task, modern frameworks like PyTorch provide an automatic differentiation tool known as [Autograd](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html). In practice, as in the implementation of the *Linear transformation*, setting `requires_grad = True` is the main way to control which parts of the model are to be tracked and included in the gradient computation.

##  3. <a name='Training'></a>Training

###  3.1 <a name='TheFashionDataset'></a>The Fashion Dataset

Fashion-MNIST is a dataset of garment images curated by [Zalando Research](https://github.com/zalandoresearch/) — consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes (T-shirt/Top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot).

Why this dataset? [As explained by the Zalando Research Team](https://github.com/zalandoresearch/fashion-mnist?tab=readme-ov-file#to-serious-machine-learning-researchers): *“MNIST is too easy. Convolutional nets can achieve 99.7%, and Classic ML algorithms can also achieve 97% easily […] We intend Fashion-MNIST to serve as a direct drop-in replacement for the original [MNIST dataset](http://yann.lecun.com/exdb/mnist/). It shares the same image size and structure of training and testing splits”*.

<figure>
  <img style="margin: 0 auto; display: block;" src="https://cdn-uploads.huggingface.co/production/uploads/64a13b68b14ab77f9e3eb061/iTxHeSK5Cen8QB-BBwqqS.png">
  <figcaption style="text-align: center;">Fashion-MNIST Dataset</figcaption>
</figure>

###  3.2 <a name='DataLoadersforMini-Batches'></a>Data Loaders for Mini-Batches

In the training process, we need to efficiently handle the loading and preprocessing of the dataset. For this purpose, we will use `torch.utils.data.DataLoader`, a utility class provided by PyTorch that helps with batching, shuffling, and loading data in parallel.

Using mini-batches instead of the entire dataset results in:
- (i) **computational efficiency** as GPUs tend to perform better when they have a larger amount of work to process in parallel;
- (ii) **better generalization** by randomly shuffling the mini-batches on every epoch, which introduces variance and prevents the model from overfitting; and,
- (iii) **reduced memory usage** as it is a practical choice to not overload the GPU’s memory with the entire dataset at once.

```python
from torchvision.transforms import ToTensor
from torchvision import datasets
from torch.utils.data import DataLoader

train_data = datasets.FashionMNIST(root = 'data', train = True, transform = ToTensor(), download = True)
test_data = datasets.FashionMNIST(root = 'data', train = False, transform = ToTensor())

loaders = {'train' : DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=2),
           'test'  : DataLoader(test_data, batch_size=config.batch_size, shuffle=False, num_workers=2)}
```

By setting `shuffle=True` in the train loader we reshuffle this data at every epoch. This is an important consideration since there might be correlations in the raw train data arising from the way the data was collected such as alphabetically or timely ordered.

###  3.3 <a name='FittingtheModel'></a>Fitting the Model

With the neural network architecture and data loaders in place, we can now focus on the process of training the model, also known as *fitting* the model to the data. The training process can be divided into two main components: the training loop and the validation loop.

**The training loop** is responsible for feeding the mini-batches of data to the model, computing the predictions and loss, and updating the model’s parameters using backpropagation and an optimization algorithm. This loop is typically run for a fixed number of epochs or until a certain stopping criterion is met.

On the other hand, **the validation loop** is used to evaluate the model’s performance on a separate validation dataset, which is not used for training. This helps monitor the model’s generalization performance and prevents overfitting to the training data.

In the following code, we implement a `Learner` class that encapsulates this logic and provides a convenient interface for fitting the model and monitoring its performance.

```python
class Learner:
    """
    Learner class for training and evaluating a model.
    """

    def __init__(self, config, loaders):
        """
        Initialize the Learner with custom configuration and data loaders.
        """
        self.model = config.model
        self.loaders = loaders
        self.optimizer = Optimizer(self.model.parameters(), config.lr)
        [...]

    def train_epoch(self, epoch):
        epoch_loss = 0.0
        for x, y in self.loaders["train"]:
            x, y = x.to(self.device), y.to(self.device)
            batch_size = x.size(0)

            # Zero out the gradients - otherwise, they will accumulate.
            self.optimizer.zero_grad()
   
            # Forward pass, loss calculation, and backpropagation
            output = self.model(x)
            loss = self.criterion(output, y)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item() * batch_size

        train_loss = epoch_loss / len(self.loaders['train'].dataset)
        return train_loss
    
    def valid_loss(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in self.loaders["test"]:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                val_loss += self.criterion(output, y).item() * y.size(0)
        val_loss /= len(self.loaders["test"].dataset)
        return val_loss

    def batch_accuracy(self, x, y):    
        _, preds = torch.max(x.data, 1)
        return (preds == y).sum().item() / x.size(0)

    def validate_epoch(self):      
        accs = [self.batch_accuracy(self.model(x.to(self.device)), y.to(self.device))
                for x, y in self.loaders["test"]]
        return sum(accs) / len(accs)
            
    def fit(self):
        """
        Train the model for the specified number of epochs.
        """
        print('epoch\ttrain_loss\tval_loss\ttest_accuracy')
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(epoch)
            valid_loss = self.valid_loss()
            batch_accuracy = self.validate_epoch()
            print(f'{epoch+1}\t{train_loss:.6f}\t{valid_loss:.6f}\t{batch_accuracy:.6f}')

        metrics = self.evaluate()
        return metrics
```

##  4. <a name='ModelAssessment'></a>Model Assessment

After 25 epochs, our model achieves 0.868 accuracy, which fairly approximates [benchmark results](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/#) (0.874 for an MLP Classifier using ReLU as the activation function).

<figure>
  <img style="margin: 0 auto; display: block;" src="https://cdn-uploads.huggingface.co/production/uploads/64a13b68b14ab77f9e3eb061/BMTJGJabSWi3Jcoc_m5sH.png">
  <figcaption style="text-align: center;">Model Assessment (epochs=25, lr=0.005, batch_size=32, SGD, CrossEntropyLoss) w/ self-implemented modules</figcaption>
</figure>

We observe comparable accuracy levels between our self-implemented modules and a standard PyTorch implementation with the same hyperparameters (`epochs=25, lr=0.005, batch_size=32`). Notably, the PyTorch model demonstrates a slighly smaller gap between validation and training losses, suggesting better generalization capabilities:

<figure>
  <img style="margin: 0 auto; display: block;" src="https://cdn-uploads.huggingface.co/production/uploads/64a13b68b14ab77f9e3eb061/mCLHZnCYArxC8YcSdEjYA.png">
  <figcaption style="text-align: center;">Model Assessment (epochs=25, lr=0.005, batch_size=32, SGD, CrossEntropyLoss) w/ PyTorch modules</figcaption>
</figure>

Furthermore, a basic analysis of precision (accuracy of the positive predictions for a specific class), recall (ability to detect all relevant instances of a specific class), and f1-score (mean of precision and recall) reveals that our model excels in categories with distinctive features such as Trouser/Jeans, Sandal, Bag, and Ankle-Boot. However, it performs below average with Shirts, Pullovers, and Coats.

<figure>
  <img style="margin: 0 auto; display: block;" src="https://cdn-uploads.huggingface.co/production/uploads/64a13b68b14ab77f9e3eb061/amnSh3QHhkgWtRKzQUpVu.png">
  <figcaption style="text-align: center;">Precision, Recall and F1-Scores across Categories</figcaption>
</figure>

The confussion matrix confirms that the Shirt category is frequently confused with the T-Shirt/Top, Pullover, and Coat classes; whereas Coat is confused with Shirt and Pullover. This suggests that working at 28x28 pixels resolution might cast upper body garment categories as visually challenging.

<figure>
  <img style="margin: 0 auto; display: block;" src="https://cdn-uploads.huggingface.co/production/uploads/64a13b68b14ab77f9e3eb061/KqbEVgVv8gz5X8JkZzX4H.png">
  <figcaption style="text-align: center;">Confussion Matrix</figcaption>
</figure>

##  5. <a name='Inference'></a>Inference

After training the model, we can use it for inference, which involves making predictions on new data. The inference process is relatively straightforward but requires to transform real-world garment images to the format of the training dataset. To achieve this, we implement a PyTorch transformation.

```python
import torch
import torchvision.transforms as transforms

# Images need to be transformed to the `fashion MNIST` dataset format
transform = transforms.Compose(
    [
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)), # Normalization
        transforms.Lambda(lambda x: 1.0 - x), # Invert colors
        transforms.Lambda(lambda x: x[0]),
        transforms.Lambda(lambda x: x.unsqueeze(0)),
    ]
)
```

This can be easily integrated into a Gradio App, and then deployed to [HuggingFace Spaces](https://huggingface.co/spaces/dcarpintero/fashion-image-recognition):

<figure>
  <img style="margin: 0 auto; display: block;" src="https://cdn-uploads.huggingface.co/production/uploads/64a13b68b14ab77f9e3eb061/HwZwQbtOuuJx-VrhnV-YD.png">
  <figcaption style="text-align: center;">Garment Classifier deployed to HuggingFace Spaces</figcaption>
</figure>

##  6. <a name='Resources'></a>Resources

- [GitHub Repo](https://github.com/dcarpintero/nn-image-classifier/)
- [Model Card](https://huggingface.co/dcarpintero/fashion-mnist-base)
- [HuggingFace Space](https://huggingface.co/spaces/dcarpintero/fashion-image-recognition)

##  7. <a name='References'></a>References

- [1] A. L. Samuel. 1959. *Some Studies in Machine Learning Using the Game of Checkers*. IBM Journal of Research and Development, Vol. 3, No. 3, pp. 210-229. [doi: 10.1147/rd.33.0210](https://ieeexplore.ieee.org/document/5392560/).

- [2] Herbert Robbins, Sutton Monro. 1951. *A Stochastic Approximation Method*. The annals of mathematical statistics, Vol. 22, No. 3, pp. 400-407. [JSTOR 2236626](https://www.jstor.org/stable/2236626)

- [3] Kunihiko Fukushima. 1969. *Visual Feature Extraction by a Multilayered Network of Analog Threshold Elements*.[doi:10.1109/TSSC.1969.300225](https://doi.org/10.1109%2FTSSC.1969.300225).

- [4] Xavier Glorot, Antoine Bordes, Yoshua Bengio. 2011. *Deep Sparse Rectifier Neural Networks*. Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics. [PMLR 15:315-323](https://proceedings.mlr.press/v15/glorot11a).

- [5] Nitish Srivastava, et al. 2014. *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. Journal of Machine Learning Research 14. [JMLR 14:1929-1958](https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)

- [6] Han Xiao, Kashif Rasul, Roland Vollgraf. 2017. *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*. [arXiv:1708.07747](http://arxiv.org/abs/1708.07747).

- [7] Matthew D Zeiler, Rob Fergus. 2013. *Visualizing and Understanding Convolutional Networks*. [arxiv:1311.2901](https://arxiv.org/abs/1311.2901).

- [8] Paul Werbos. 1974. *Beyond Regression: New Tools for Prediction and Analysis in the Behavioral Sciences*. 