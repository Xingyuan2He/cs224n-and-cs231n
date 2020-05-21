#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn


# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway


# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, word_embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()

        # YOUR CODE HERE for part 1h
        self.word_embed_size = word_embed_size
        self.e_char = 50
        self.drop_rate = 0.3
        self.vocab = vocab
        self.size_vocab = len(vocab.char2id)
        self.pad_index = vocab.char2id['<pad>']
        self.char_embedding_layer = nn.Embedding(
            self.size_vocab, self.e_char, self.pad_index)
        self.CNN = CNN(
            e_char=self.e_char,
            e_word=word_embed_size,
            k_size=5)
        self.Highway = Highway(self.word_embed_size)
        self.dropout = nn.Dropout(self.drop_rate)
        # END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        # YOUR CODE HERE for part 1h
        sentence_length, batch_size, max_word_length = input.shape
        # sentence_length, batch_size, max_word_length,e_char
        X_emb = self.char_embedding_layer(input)
        tgt_shape = (sentence_length * batch_size,
                     max_word_length, self.e_char)

        # x_reshape : sentence_length * batch_size, e_char, max_word_length
        x_reshape = X_emb.view(tgt_shape).transpose(1, 2)
        x_conv_out = self.CNN(x_reshape).squeeze(2)
        x_highway = self.Highway(x_conv_out)
        x_word_emb = self.dropout(x_highway)
        x_word_emb = x_word_emb.view(
            (sentence_length, batch_size, self.word_embed_size))
        # (sentence_length, batch_size, word_embed_size)

        return x_word_emb
        # END YOUR CODE
