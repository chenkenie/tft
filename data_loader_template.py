import numpy as np
from tensorflow.keras.utils import to_categorical

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
        self.input_train = []
        self.target_train = []
        self.input_valid = []
        self.target_valid = []
        self.start = 0
        self.end = 0

    def epoch_init(self):
        self.input_train, self.target_train = randomize(self.input_train, self.target_train)
        self.start = 0
        self.end = 0
        
    def next_batch(self, batch_size):
        self.start = self.end
        self.end = self.start + batch_size
        input_batch, target_batch = get_next_batch(self.input_train, self.target_train, self.start, self.end)
        return input_batch, target_batch


class DataMNIST(TensorFlowDataLoaderTemplate):
    def __init__(self):
        from tensorflow.keras.datasets import mnist
        img_h = img_w = 28
        img_size_flat = img_h * img_w
        
        (input_train, target_train), (input_test, target_test) = mnist.load_data()

        self.input_train = input_train
        self.target_train = to_categorical(target_train)
        self.input_valid = input_test
        self.target_valid = to_categorical(target_test)

        self.input_train = self.input_train.reshape((-1, img_size_flat))
        self.input_valid = self.input_valid.reshape((-1, img_size_flat))

class DataMNISTAE(TensorFlowDataLoaderTemplate):
    def __init__(self):
        from tensorflow.keras.datasets import mnist
        img_h = img_w = 28
        img_size_flat = img_h * img_w
        noise_level = 0.9
        
        (input_train, target_train), (input_test, target_test) = mnist.load_data()

        self.input_train = input_train
        self.input_valid = input_test

        self.input_train = input_train.reshape((-1, img_size_flat))*1.0
        self.input_train += noise_level * np.random.normal(loc=0.0, scale=255.0, size=self.input_train.shape)
        self.target_train = input_train.reshape((-1, img_size_flat))*1.0
        self.input_valid = input_test.reshape((-1, img_size_flat))*1.0
        self.input_valid += noise_level * np.random.normal(loc=0.0, scale=255.0, size=self.input_valid.shape)
        self.target_valid = input_test.reshape((-1, img_size_flat))*1.0


class DataMNISTAE_old(TensorFlowDataLoaderTemplate):
    def __init__(self):
        from tensorflow.examples.tutorials.mnist import input_data
        img_h = img_w = 28
        img_size_flat = img_h * img_w
        noise_level = 0.6
        
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        input_train = mnist.train.images
        input_valid = mnist.validation.images        

        #self.input_train = input_train + noise_level * np.random.normal(loc=0.0, scale=1.0, size=input_train.shape)
        self.input_train = input_train
        self.target_train = input_train

        #self.input_valid = input_valid + noise_level * np.random.normal(loc=0.0, scale=1.0, size=input_valid.shape)
        self.input_valid = input_valid
        self.target_valid = input_valid
        
