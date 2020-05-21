from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    num_sample = X.shape[0]
    num_class = W.shape[1]
    dW = np.zeros_like(W)
    for i in range(num_sample):
        score = X[i].dot(W)
        max_score = np.max(score)
        score -= max_score
        sum_score = np.sum(np.exp(score))
        loss = loss - score[y[i]] + np.log(sum_score)

        dW[:, y[i]] -= X[i]  # 对分子的derive
        for j in range(num_class):
            dW[:, j] += np.exp(score[j]) * X[i] / sum_score  # 分母derive

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dW /= num_sample
    loss /= num_sample

    dW += reg * W
    loss += 0.5 * reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    loss_arrray = np.zeros((X.shape[0], 1))
    dW = np.zeros_like(W)

    num_sample = X.shape[0]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    score = X.dot(W)  # n,c
    max_score = np.max(score, axis=1, keepdims=True)  # n,1
    score -= max_score
    sum_score = np.sum(np.exp(score), axis=1).reshape(-1, 1)
    loss_arrray = loss_arrray - \
        score[np.arange(num_sample), y].reshape(-1, 1) + \
        np.log(sum_score)
    loss += np.sum(loss_arrray)

    transform_matrix = np.zeros_like(score)
    transform_matrix += np.exp(score) / sum_score
    transform_matrix[np.arange(num_sample), y] -= 1

    dW += X.T.dot(transform_matrix)

    dW /= num_sample
    loss /= num_sample

    dW += reg * W
    loss += 0.5 * reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
