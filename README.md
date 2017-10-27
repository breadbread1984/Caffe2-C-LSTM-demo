# Caffe2 C++ LSTM demo
### Introduction

This demo shows how to create a recurrent network with Caffe2.

### modify Makefile
change the value of CAFFE2_PREFIX in the makefile according to your customized installation location.

### create training set for training recurrent network
```Shell
make dataset
```

### train on dataset and save params into lmdb format
```Shell
rm -rf LSTM_params && make train
```

### sample a sentence with trained recurrent network
```Shell
make test
```
you have to interpret the sampled sentence through mapping the output of softmax to alphabets.

### make sentence sampler implement in C++
```Shell
make predictor
```
you can sample sentence containing 100 alphabets through the following command.
```Shell
./predictor -l 100
```
