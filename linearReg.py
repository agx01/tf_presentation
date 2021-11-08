# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 00:52:32 2021

@author: Arijit Ganguly
"""
#%% Section 1
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

tf.enable_eager_execution()
tf.disable_v2_behavior()


np.random.seed(101)
tf.set_random_seed(101)

#Generating random linear data
x = np.linspace(0, 50, 50)
y = np.linspace(0, 50, 50)

#Adding noise to the random linear data
x += np.random.uniform(-4, 4, 50)
y += np.random.uniform(-4, 4, 50)

#No of samples
n = len(x)

plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title("Training Data")
plt.show()

#%% Section 2
#training examples
X = tf.placeholder("float")
Y = tf.placeholder("float")

#trainable variables for weights and bias
W = tf.Variable(np.random.randn(), name = "W")
B = tf.Variable(np.random.randn(), name = "b")

#Define hyperparameters
learning_rate = 0.01
training_epochs = 1000

#Hypothesis
y_pred = tf.add(tf.multiply(X, W), B)

#Mean squared error is the cost function
cost = tf.reduce_sum(tf.pow(y_pred-Y, 2)) / (2 * n)

#Gradient descent Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#Global Variables Initializer
init = tf.global_variables_initializer()

with tf.Session() as sess:
    
    #Initializing variables
    sess.run(init)
    
    #Iterating through all epochs
    for epoch in range(training_epochs):
        
        #Feeding each data point into the optimizer using the Feed Dictionary
        for (_x, _y) in zip(x,y):
            sess.run(optimizer, feed_dict = {X: _x, Y: _y})
            
        #Displaying the result after every 50 epochs
        if (epoch+1) % 50 == 0:
            
            #Calculating the cost for each epoch
            c = sess.run(cost, feed_dict = {X: x, Y: y})
            print("Epoch ", (epoch+1), ": cost = ", c, "W = ", sess.run(W), "b = ", sess.run(B))
    
    #Storing necessary values to be used outside the Session
    training_cost = sess.run(cost, feed_dict={X: x, Y: y})
    weight = sess.run(W)
    bias = sess.run(B)

#Calculating predictions    
predictions = weight * x + bias
print("Training Cost = ", training_cost, "Weight = ", weight, "bias = ", bias)

#PLotting the results
plt.plot(x, y, 'ro', label= 'Original Data')
plt.plot(x, predictions, label= 'Fitted Line')
plt.title("Linear Regression Result")
plt.legend()
plt.show()

    