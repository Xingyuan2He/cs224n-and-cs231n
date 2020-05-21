#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    # YOUR CODE HERE for part 1f
    def __init__(self, e_word):
        '''
        @param X_conv_out (Tensor): (N,e_word)

        '''
        super(Highway, self).__init__()
        self.proj_layer = nn.Linear(e_word, e_word)
        self.gate_layer = nn.Linear(e_word, e_word)

    def forward(self, X_conv_out: torch.Tensor) -> torch.Tensor:

        x_proj = F.relu(self.proj_layer(X_conv_out))
        x_gate = torch.sigmoid(self.gate_layer(X_conv_out))

        x_highway = x_gate * x_proj + (1 - x_gate) * X_conv_out

        return x_highway
    # END YOUR CODE
