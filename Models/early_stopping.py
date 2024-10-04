import numpy as np

# Early Stopping class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = np.inf
        self.epochs_without_improvement = 0
        self.should_stop = False

    def check(self, val_loss):
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0  # Reset the counter
        else:
            self.epochs_without_improvement += 1  # Increment the counter

        if self.epochs_without_improvement >= self.patience:
            self.should_stop = True