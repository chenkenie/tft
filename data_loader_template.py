import numpy as np

def randomize(x, y):
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y

def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]    
    return x_batch, y_batch

class TensorFlowDataLoaderTemplate:
    def __init__(self):
        self.x_train = []
        self.y_train = []
        self.x_valid = []
        self.y_valid = []
        self.start = 0
        self.end = 0

    def epoch_init(self):
        self.x_train, self.y_train = randomize(self.x_train, self.y_train)
        self.start = 0
        self.end = 0
        
    def next_batch(self, batch_size):
        self.start = self.end
        self.end = self.start + batch_size
        x_batch, y_batch = get_next_batch(self.x_train, self.y_train, self.start, self.end)
        return x_batch, y_batch

