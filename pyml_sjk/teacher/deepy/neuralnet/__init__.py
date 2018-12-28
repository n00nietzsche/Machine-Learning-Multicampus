#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 14:11:25 2018

@author: seongjoo
"""
import time
import numpy as np

def cross_entropy_error(y, y_pred):
    delta = 1e-7 # np.log(0) --> - inf
    return -np.sum(y*np.log(y_pred+delta))

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val
        
    return grad

def numerical_gradient_batch(f, X):
    if X.ndim == 1:
        return numerical_gradient(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = numerical_gradient(f, x)
        
        return grad
    
class LayerError(RuntimeError):
    def __init__(self, arg):
        self.arg = arg
    

class FeedForwardNet:    
    def __init__(self, loss_func):
        self.layers = []
        self.loss_func = loss_func
        
    def add_layer(self, layer):
        self.layers.append(layer)
        
    def predict(self, X):
        if not len(self.layers):
            raise LayerError('No layers added')
        layer_input = X
        for layer in self.layers:
            layer_input = layer.output(layer_input)
        y = layer_input
        
        return y
    
    def _compute_loss(self, X, y):
        y_pred = self.predict(X)
        return self.loss_func(y, y_pred)
    
    
    def _gradient_descent(self, X, y, learning_rate=0.01):
        for layer in self.layers:
            dW = numerical_gradient_batch(lambda params: self._compute_loss(X, y), layer.params)
            layer.params -= learning_rate * dW
    
    def fit(self, X, y, epochs, learning_rate=0.01):
        loss_history = []
        for i in range(epochs):
            print('\nEpoch', i+1)
            self._gradient_descent(X, y, learning_rate)
            
            loss = self._compute_loss(X, y)
            print('\nLoss: {:.3f}'.format(loss))      
            loss_history.append(loss)
        return loss_history
    
    def batch_fit(self, X, y, epochs, batch_size, learning_rate=0.01):
        loss_history = []
        for i in range(epochs):
            print('\nEpoch', i+1, end=' ')
            
            start = time.time()

            n_batch = int(len(X) / batch_size) if batch_size else 1
            batch_losses = []
            for j in range(n_batch):
                print('=', end='')
                                
                batch_mask = np.random.choice(len(X), batch_size)
                X_batch = X[batch_mask]
                y_batch = y[batch_mask]

                self._gradient_descent(X_batch, y_batch, learning_rate)
                batch_losses.append(self._compute_loss(X_batch, y_batch))
            
            end = time.time()
            
            loss = np.mean(batch_losses)
            print('\nLoss: {:.3f} ({:.2f} seconds)'.format(loss, end-start))
            loss_history.append(loss)
        return loss_history        
        
    
class Layer:
    def __init__(self, input_size, output_size, activation):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros(output_size)
        self.params = np.vstack([self.bias, self.weights])
        
    def _net_input(self, X):
        return np.dot(X, self.params[1:]) + self.params[0]
    
    def set_weights(self, W):
        if not self.params[1:].shape == W.shape:
            raise ValueError('weight should be {}'.format(self.params[1:].shape))
        self.params[1:] = W
        
    def set_bias(self, b):
        if not self.params[0].shape == b.shape:
            raise ValueError('weight should be {}'.format(self.params[0:].shape))
        self.params[0] = b
    
    def output(self, X):
        z = self._net_input(X)
        y = self.activation(z)
        return y