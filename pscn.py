import networkx as nx
import numpy as np
import tensorflow as tf
import util
from receptive_field import ReceptiveField
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Model, Sequential
from keras.layers import Conv1D, Dense, Dropout, Flatten

class PSCN():
    def __init__(self, w, num_attr, s=1, k=10, l='betweenness', # Patchy-San parameters
                epochs=150, batch_size=32, optimizer='rmsprop', # CNN Parameters
                attribute_name='node_attributes', num_classes=2):
        # Receptive field parameters
        self.w = w
        self.s = s
        self.k = k
        self.l = l
        self.attribute_name = attribute_name
        # CNN parameters
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.metrics = ['accuracy']
        self.optimizer = optimizer
        self.num_attr = num_attr
        self.model = KerasClassifier(build_fn=self.init_model,
                                    epochs=self.epochs,
                                    batch_size=self.batch_size)
    
    # Initializes the CNN
    # Convolutional layer channels depends on the number of attributes 
    def init_model(self):
        model = Sequential()
        
        # First convolutional layer
        first_conv_layer = Conv1D(filters=16, 
                            kernel_size=self.k, 
                            strides=self.k, 
                            input_shape=(self.w * self.k, self.num_attr))
        model.add(first_conv_layer)
        # Second convolutional layer
        second_conv_layer = Conv1D(filters=8, 
                            kernel_size=min(self.w, 10), # Handle small graphs with less than 10 neighbourhoods
                            strides=1)
        model.add(second_conv_layer)
        model.add(Flatten())
        # ReLU units
        model.add(Dense(128,
                        activation='relu'))
        # Regularizer
        model.add(Dropout(0.5))
        
        # Binary classification
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', 
                      optimizer=self.optimizer, # Maybe use ADAM?
                      metrics=self.metrics)
        
        return model
    
    # Train model
    # Input data X should be in graph form
    def fit(self, X, y):
        # Reshape input receptive fields
        X_tensors = self.process_data(X, self.num_attr)
        
        self.model.fit(X_tensors, y)
    
    # Perform evaluations on test data X, y
    # TODO: Implement evaluation for multiclass predictions
    def evaluate(self, X, y):
        predictions = self.predict(X)
        
        accuracy = np.sum(predictions==y)/len(y)
        print('Accuracy: ', accuracy)
    
    # Makes predictions on the given input graphs
    def predict(self, X):
        X_tensors = self.process_data(X, self.num_attr)
        
        return self.model.predict(X_tensors).ravel()
    
    # Reshapes the data to fit the convolutional layer input shape
    # X - input graphs
    def process_data(self, X, num_attr):
        rf_tensors = []
        for g in X:
            rf = ReceptiveField(g,
                                w=self.w,
                                k=self.k,
                                s=self.s,
                                l=self.l, 
                                attribute_name=self.attribute_name, 
                                num_attr=self.num_attr)
            
            receptive_fields = rf.make_all_receptive_fields()
            # Reshape to (w*k, a_v)
            rf_tensor = np.array(receptive_fields).flatten().reshape(self.w * self.k, num_attr)
            rf_tensors.append(rf_tensor)
            
        return np.array(rf_tensors)