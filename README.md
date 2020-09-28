## Short Text Clustering via Convolutional Neural Networks

This repository contains a Keras implementation of the algorithm presented in the paper 
[Short Text Clustering via Convolutional Neural Networks](https://www.aclweb.org/anthology/W15-1509/) with some changes:

* Using Spectral Hashing algorithm instead of locality preserving constraint
* Using word vector embeddings to obtain the binary code

### Usage

Execute ```tracker/server.py``` to monitor all instances.
Execute ```learn_and_test.py``` to start training or testing.

### References

https://github.com/wanji/sh
