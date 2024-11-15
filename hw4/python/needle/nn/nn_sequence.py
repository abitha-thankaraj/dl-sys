"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module, ReLU


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return (1.0 + ops.exp(-x)) ** (-1)
        ### END YOUR SOLUTION
class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return ops.tanh(x)
class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype
        self.bias = bias

        # make modules
        higher_bound = 1.0 / np.sqrt(hidden_size)
        lower_bound = -higher_bound

        self.W_ih = Parameter(
          init.rand(
            self.input_size,
            self.hidden_size,
            low=lower_bound,
            high=higher_bound,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
          )
        )

        self.W_hh = Parameter(
          init.rand(
            self.hidden_size,
            self.hidden_size,
            low=lower_bound,
            high=higher_bound,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
          )
        )

        # make bias
        if self.bias:
          self.bias_ih = Parameter(
            init.rand(
              hidden_size,
              low=lower_bound,
              high=higher_bound,
              device=self.device,
              dtype=self.dtype,
              requires_grad=True,
            )
          )

          self.bias_hh = Parameter(
            init.rand(
              hidden_size,
              low=lower_bound,
              high=higher_bound,
              device=self.device,
              dtype=self.dtype,
              requires_grad=True,
            )
          )

        else:
          self.bias_ih = None
          self.bias_hh = None

        # make activations
        if nonlinearity == "relu":
          self.nonlinearity = ReLU()
        elif nonlinearity == "tanh":
          self.nonlinearity = Tanh()
        else:
          raise Exception(f" {nonlinearity} is not supported")
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        B = X.shape[0]

        # handles the case of h being None
        if h is None:
          h = init.zeros(
            B,
            self.hidden_size,
            device=self.device,
            dtype=self.dtype,
          )

        out = X @ self.W_ih + h @ self.W_hh
        if self.bias:
          # reshape bias to be same size as X @ self.W_ih
          reshape_shape = (1, self.hidden_size)
          broadcast_shape = (B, self.hidden_size)

          reshaped_bias_ih = self.bias_ih.reshape(reshape_shape).broadcast_to(broadcast_shape)
          reshaped_bias_hh = self.bias_hh.reshape(reshape_shape).broadcast_to(broadcast_shape)

          out += reshaped_bias_hh + reshaped_bias_ih

        out = self.nonlinearity(out)
        return out
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.dtype = dtype
        self.device = device

        self.rnn_cells = [
          RNNCell(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            bias=self.bias,
            nonlinearity=self.nonlinearity,
            device=self.device,
            dtype=self.dtype,
          )
        ]

        # add additional layers
        for _ in range(self.num_layers - 1):
          self.rnn_cells.append(
            RNNCell(
              input_size=self.hidden_size,
              hidden_size=self.hidden_size,
              bias=self.bias,
              nonlinearity=self.nonlinearity,
              device=self.device,
              dtype=self.dtype,
            )
          )
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        assert len(X.shape) == 3
        batch_size = X.shape[1]

        if h0 is None:
          h0 = []
          for index in range(self.num_layers):
            h0.append(
              init.zeros(
                batch_size,
                self.hidden_size,
                dtype=self.dtype,
                device=self.device,
              )
            )
          h0 = tuple(h0)
        else:
          h0 = ops.split(h0, 0)

        h_all = []
        out = list(ops.split(X, 0))

        # do the forward pass
        for i in range(self.num_layers):
          h_i = h0[i]

          for j in range(len(out)):
            h_i = self.rnn_cells[i](X=out[j], h=h_i)
            out[j] = h_i

          h_all.append(h_i)

        # stack the output
        out = ops.stack(out, 0)
        h_n = ops.stack(h_all, 0)

        return out, h_n
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype
        self.bias = bias

        # setup weights
        higher_bound = 1.0 / np.sqrt(self.hidden_size)
        lower_bound = -higher_bound

        self.W_ih = Parameter(
          init.rand(
            self.input_size,
            4 * self.hidden_size,
            low=lower_bound,
            high=higher_bound,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
          )
        )

        self.W_hh = Parameter(
          init.rand(
            self.hidden_size,
            4 * self.hidden_size,
            low=lower_bound,
            high=higher_bound,
            device=self.device,
            dtype=self.dtype,
            requires_grad=True,
          )
        )

        # setup biases
        if self.bias:
          self.bias_ih = Parameter(
            init.rand(
              4 * hidden_size,
              low=lower_bound,
              high=higher_bound,
              device=self.device,
              dtype=self.dtype,
              requires_grad=True,
            )
          )

          self.bias_hh = Parameter(
            init.rand(
              4 * hidden_size,
              low=lower_bound,
              high=higher_bound,
              device=self.device,
              dtype=self.dtype,
              requires_grad=True,
            )
          )

        else:
          self.bias_ih = None
          self.bias_hh = None

        self.tanh_layer = Tanh()
        self.sigmoid_layer = Sigmoid()
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        batch_size = X.shape[0]

        # initialize h0 and c0 if first time
        if h is None:
          h0 = init.zeros(
            batch_size,
            self.hidden_size,
            device=self.device,
            dtype=self.dtype,
          )

          c0 = init.zeros(
            batch_size,
            self.hidden_size,
            device=self.device,
            dtype=self.dtype,
          )

        else:
          h0 = h[0]
          c0 = h[1]

        # calculate the gates value
        gates = X @ self.W_ih + h0 @ self.W_hh
        if self.bias:
          # reshape bias to be same size as X @ self.W_ih
          reshape_shape = (1, 4 * self.hidden_size)
          broadcast_shape = (batch_size, 4 * self.hidden_size)

          reshaped_bias_ih = self.bias_ih.reshape(reshape_shape).broadcast_to(broadcast_shape)
          reshaped_bias_hh = self.bias_hh.reshape(reshape_shape).broadcast_to(broadcast_shape)

          gates += reshaped_bias_ih + reshaped_bias_hh

        gates = list(ops.split(gates, axis=1))

        # calculate i, f, g, o from the description
        vectors = []
        for index in range(4):
          vec = ops.stack(gates[index * self.hidden_size : (index + 1) * self.hidden_size], axis=1)
          vectors.append(vec)

        i = self.sigmoid_layer(vectors[0])
        f = self.sigmoid_layer(vectors[1])
        g = self.tanh_layer(vectors[2])
        o = self.sigmoid_layer(vectors[3])

        # calculate c', h'
        c_prime = f * c0 + i * g
        h_prime = o * self.tanh_layer(c_prime)

        return h_prime, c_prime
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dtype = dtype
        self.device = device

        # initialize LSTM cells
        self.lstm_cells = [
          LSTMCell(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            bias=self.bias,
            device=self.device,
            dtype=self.dtype,
          )
        ]

        for _ in range(num_layers - 1):
          self.lstm_cells.append(
            LSTMCell(
              input_size=self.hidden_size,
              hidden_size=self.hidden_size,
              bias=self.bias,
              device=self.device,
              dtype=self.dtype,
            )
          )
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        B = X.shape[1]

        # Initialize h0, c0
        if h is None:
          h0 = []
          c0 = []

          for _ in range(self.num_layers):
            h0.append(
              init.zeros(
                B,
                self.hidden_size,
                device=self.device,
                dtype=self.dtype,
              )
            )

            c0.append(
              init.zeros(
                B,
                self.hidden_size,
                device=self.device,
                dtype=self.dtype,
              )
            )

        else:
          h0 = ops.split(h[0], axis=0)
          c0 = ops.split(h[1], axis=0)

        # fwd
        h_n = []
        c_n = []

        out = list(ops.split(X, axis=0))
        for i in range(self.num_layers):
          h = h0[i]
          c = c0[i]

          for j in range(len(out)):
            h, c = self.lstm_cells[i](
              X=out[j], 
              h=(h, c),
            )
            out[j] = h

          h_n.append(h)
          c_n.append(c)

        # Convert lists into tensors w stack 
        out = ops.stack(out, 0)
        h_n = ops.stack(h_n, 0)
        c_n = ops.stack(c_n, 0)

        return out, (h_n, c_n)
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.weight = Parameter(
          init.randn(
            self.num_embeddings,
            self.embedding_dim,
            mean=0.0,
            std=1.0,
            device=self.device,
            dtype=self.dtype,
          )
        )
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        one_hot_embedding = init.one_hot(
          self.num_embeddings,
          x,
          device=self.device,
          dtype=self.dtype,
        )

        sequence_length, batch_size = x.shape[0], x.shape[1]

        one_hot_embedding = one_hot_embedding.reshape((sequence_length * batch_size, self.num_embeddings))

        out = one_hot_embedding @ self.weight
        out = out.reshape((sequence_length, batch_size, self.embedding_dim))

        return out
        ### END YOUR SOLUTION