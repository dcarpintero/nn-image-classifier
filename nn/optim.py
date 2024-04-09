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