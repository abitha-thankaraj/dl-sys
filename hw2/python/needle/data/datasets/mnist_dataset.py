from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)    
        self.images, self.labels = parse_mnist(image_filename, label_filename)
        # Images to B x H x W x C
        # self.images = self.images.reshape(-1, 28, 28, 1)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        if self.transforms is not None:        
            return self.apply_transforms(self.images[index].reshape((28, 28, 1))), self.labels[index]
        else:
            return self.images[index], self.labels[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.labels)
        ### END YOUR SOLUTION


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    #local folder
    if "./data" in image_filename:
        image_filename = image_filename.replace("./data", "/home/abitha/projects/dl-sys/hw2/data")
        label_filename = label_filename.replace("./data", "/home/abitha/projects/dl-sys/hw2/data")
    elif "data/" in image_filename:
        image_filename = image_filename.replace("data/", "/home/abitha/projects/dl-sys/hw2/data/")
        label_filename = label_filename.replace("data/", "/home/abitha/projects/dl-sys/hw2/data/")
    with gzip.open(label_filename, 'rb') as lbpath:
        lbpath.read(8)  # skip the magic number and number of items
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)
    
    with gzip.open(image_filename, 'rb') as imgpath:
        imgpath.read(16)  # skip the magic number, number of items, rows, and columns
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), 784)
        images = images.astype(np.float32) / 255.0  # normalize to range [0.0, 1.0]
    
    return images, labels