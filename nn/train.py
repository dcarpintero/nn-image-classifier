class Learner:
    """
    Learner class for training and evaluating a model.

    This class encapsulates the training and validation loops, as well as
    utility methods for prediction, exporting the model, and calculating
    accuracy.
    """

    def __init__(self, config, loaders):
        """
        Initialize the Learner.

        Args:
            config (LearnerConfig): Configuration for the Learner.
            loaders (dict): Dictionary of data loaders for training and testing.
        """
        self.model = config.model
        self.loaders = loaders
        self.optimizer = Optimizer(self.model.parameters(), config.lr)
        self.criterion = config.criterion
        self.epochs = config.epochs
        self.device = config.device
        self.labels = ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle-Boot']
        self.model.to(self.device)

    def train_epoch(self, epoch):
        """
        Train the model for one epoch.
        """
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
        """
        Calculate the validation loss.
        """
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
        """
        Calculate the accuracy for a batch of inputs (x) and targets (y).
        """        
        _, preds = torch.max(x.data, 1)
        return (preds == y).sum().item() / x.size(0)

    def validate_epoch(self):
        """
        Evaluate the model on the test dataset after an epoch.
        """        
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
            
    def predict(self, x):
        with torch.no_grad():
            outputs = self.model(x.to(self.device))
            _, preds = torch.max(outputs.data, 1)
        return preds
    
    def export(self, path):
        torch.save(self.model, path)
                
    def evaluate(self):
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x, y in self.loaders["test"]:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())

        class_precision = precision_score(all_targets, all_preds, average=None)
        class_recall = recall_score(all_targets, all_preds, average=None)
        class_f1 = f1_score(all_targets, all_preds, average=None)

        metrics = {label: {"precision": prec, "recall": rec, "f1": f1}
                   for label, prec, rec, f1 in zip(self.labels, class_precision, class_recall, class_f1)}

        return metrics