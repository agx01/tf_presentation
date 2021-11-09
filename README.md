# TensorFlow Presentation
Presentation explains the variations of use of TensorFlow

It also includes the below mentioned simple projects that help simplify understanding the
use of TensorFlow

## Linear Regression using TensorFlow

The code generates samples which are used for implementing Linear Regression
![Sample dataset](https://github.com/agx01/tf_presentation/blob/main/Figure_1.png?raw=true)

Then the code generates the required model using PlaceHolders for samples and labels,
and Variables for the weights and the bias

After that we setup the hypothesis, for the linear model and setup 
the mean square function as the  cost function.

We use TensorFlow's implemented code to implement the Gradient Descent optimizer
and minimize the cost.

Code displays weights, bias and the cost after every 50 epochs.

The Results for displaying the fitted line
![Results of Linear Regression](https://github.com/agx01/tf_presentation/blob/main/Figure_2.png?raw=true)

## Iris Data Classification using Neural Networks on TensorFlow

We build the a 3 layer neural network including the input and output layers.

The first input layer shape is provided as per the number of features we are using.
The final output layer shape is the number of classes that we are trying to predict
