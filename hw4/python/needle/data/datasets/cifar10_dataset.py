import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

def load_cifar10(base_folder, train):
  if train:
    data_files = [f"data_batch_{i}" for i in [1, 2, 3, 4, 5]]
  else:
    data_files = ['test_batch']

  X, y = [], []

  for data_file in data_files:
    file_absolute_path = os.path.join(base_folder, data_file)
    with open(file_absolute_path, 'rb') as f:
      data_dict = pickle.load(f, encoding='bytes')
      X.append(data_dict[b'data'])
      y.append(data_dict[b'labels'])

  X = np.concatenate(X, axis=0).reshape((-1, 3, 32, 32)).astype(np.float32)
  X /= 255.0

  y = np.concatenate(y, axis=None)
  return X, y


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)

        self.X, self.y = load_cifar10(
          base_folder=base_folder,
          train=train,
        )

        assert self.X.shape[0] == self.y.shape[0]
        assert self.X[0].shape == (3, 32, 32)

        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        return self.X[index], self.y[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return self.y.shape[0]
        ### END YOUR SOLUTION
