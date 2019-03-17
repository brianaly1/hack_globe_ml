import sys
import tensorflow as tf
import numpy as np

class network(object):

    def __init__(self, scope, units, reuse=False):
        '''
        Inputs:
            scope: name to use for tf scopes
            units: list containing number of hidden units per layer, units[0] = number of input features
            activations: list containing activations to be used, activations[0] = None 
            reuse: wether tf variables should be reused with multiple objects of this class
        '''

        self.scope = scope
        self.units = units
        self.reuse = reuse
        self.outputs=[]
    
    def inference(self,inputs):
        '''
        Builds the computation graph for the critic
        Inputs:
            states: tf placeholder inputs to network
        '''
        self.outputs = [inputs]
        with tf.variable_scope(self.scope, reuse=self.reuse):
            for i in range(1,len(self.units)-1):
                layer = tf.layers.dense(self.outputs[i-1], self.units[i], tf.nn.relu)
                #dropout = tf.nn.dropout(layer,0.5)
                self.outputs.append(layer)
            final_layer = tf.layers.dense(self.outputs[-1], self.units[-1])
            return final_layer


    def loss(self,logits,labels):
        loss = tf.losses.softmax_cross_entropy(labels, logits)
        loss_mean = tf.reduce_mean(loss, name='cross_entropy')
        return loss




