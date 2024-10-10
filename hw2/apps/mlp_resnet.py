import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    modules = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim),
    )
    residual = nn.Residual(modules)
    return nn.Sequential(residual, nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    layers = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *[ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob) for _ in range(num_blocks)],
        nn.Linear(hidden_dim, num_classes),
    )
    return layers
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    avg_error = 0.0
    total_loss = []
    loss_fn = nn.SoftmaxLoss()

    if opt is not None:
        model.train()
        for X, y in dataloader:
            opt.reset_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            avg_error += np.sum(logits.numpy().argmax(axis=1) != y.numpy())
            total_loss.append(loss.numpy())
            loss.backward()
            opt.step()
    else:
        model.eval()
        for X, y in dataloader:
            logits = model(X)
            loss = loss_fn(logits, y)
            avg_error += np.sum(logits.numpy().argmax(axis=1) != y.numpy())
            total_loss.append(loss.numpy())

    avg_error /= len(dataloader.dataset)
    avg_loss = np.mean(total_loss)
    return avg_error, avg_loss
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    
    net = MLPResNet(28*28, hidden_dim=hidden_dim)
    opt = optimizer(net.parameters(), lr=lr, weight_decay=weight_decay)
    train_data = ndl.data.MNISTDataset(
        f"{data_dir}/train-images-idx3-ubyte.gz",
        f"{data_dir}/train-labels-idx1-ubyte.gz",
    )
    train_loader = ndl.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data = ndl.data.MNISTDataset(
        f"{data_dir}/t10k-images-idx3-ubyte.gz", f"{data_dir}/t10k-labels-idx1-ubyte.gz"
    )
    test_loader = ndl.data.DataLoader(test_data, batch_size=batch_size)

    for _ in range(epochs):
        train_error, train_loss = epoch(train_loader, net, opt)
    test_error, test_loss = epoch(test_loader, net)

    return train_error, train_loss, test_error, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
