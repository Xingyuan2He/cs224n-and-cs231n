#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    # YOUR CODE HERE for part 1g
    def __init__(self, e_char, e_word, k_size=5):

        super(CNN, self).__init__()
        self.conv_layer = nn.Conv1d(
            in_channels=e_char,
            out_channels=e_word,
            kernel_size=k_size, padding=1)

    def forward(self, X_reshape: torch.Tensor) -> torch.Tensor:

        # senten-len * b_size ,e_word,(m_word-2) :50,ew,19
        X_conv = self.conv_layer(X_reshape)

        x_relu = F.relu(X_conv)
        # senten-len * b_size ,e_word
        X_conv_out = F.max_pool1d(x_relu, kernel_size=x_relu.size(2))
        # print(X_conv_out.shape)

        return X_conv_out
    # END YOUR CODE
