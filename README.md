# keras-mnist-nn
A 3-layer neural network for MNIST using keras

The jupyter notebook contains my submission for the [Digit Recognizer Challenge](https://www.kaggle.com/c/digit-recognizer/) on Kaggle the same neural network. The neural network got a score of 0.97071 on 1 and the submission was ranked 1722 out of 2459 on a rolling scoreboard.

## Layers
- Input layer
- Layer with `392` neurons [relu activation]
- Layer with `196` neurons [relu activation]
- Output layer with `10` neurons [softmax activation]

This model achieves 98.5% test accuracy in 40 epochs using `rmsprop` optimizer 

### To Use
Download [MNIST](http://yann.lecun.com/exdb/mnist/) dataset in a folder and name it 'mnist'
