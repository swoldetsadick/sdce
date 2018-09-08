# Lesson IX: Deep neural networks

### 1. Let's go deeper

[![Let's go deeper](http://img.youtube.com/vi/SzTpc_EWbDs/0.jpg)](https://youtu.be/SzTpc_EWbDs "Let's go deeper")

### 2. Introduction to deep neural networks

[![Introduction to deep neural networks](http://img.youtube.com/vi/SXtXg_BB4lI/0.jpg)](https://youtu.be/SXtXg_BB4lI "Introduction to deep neural networks")

### 3. Quiz: Number of parameters

[![Quiz: Number of parameters](http://img.youtube.com/vi/8cIlVoH5dhw/0.jpg)](https://youtu.be/8cIlVoH5dhw "Quiz: Number of parameters")

**Answer:**

[![Quiz answer](http://img.youtube.com/vi/TkaTTptnYdA/0.jpg)](https://youtu.be/TkaTTptnYdA "Quiz answer")

### 4. Linear models are limited

[![Linear models are limited](http://img.youtube.com/vi/12AYOYDrpfQ/0.jpg)](https://youtu.be/12AYOYDrpfQ "Linear models are limited")

### 5. Quiz: Rectified linear units

[![Quiz: Rectified linear units](http://img.youtube.com/vi/z9crz_gwGCM/0.jpg)](https://youtu.be/z9crz_gwGCM "Quiz: Rectified linear units")

**Answer:**

[![Quiz answer](http://img.youtube.com/vi/TkaTTptnYdA/0.jpg)](https://youtu.be/TkaTTptnYdA "Quiz answer")

### 6. Network of RELUs

[![Network of RELUs](http://img.youtube.com/vi/mJk1UvhDb1g/0.jpg)](https://youtu.be/mJk1UvhDb1g "Network of RELUs")

### 7. 2-Layer neural network

###### Multilayer Neural Networks

In this lesson, you'll learn how to build multilayer neural networks with TensorFlow. Adding a hidden layer to a network 
allows it to model more complex functions. Also, using a non-linear activation function on the hidden layer lets it 
model non-linear functions.

Next, you'll see how a ReLU hidden layer is implemented in TensorFlow.
    **Note**: Depicted above is a "2-layer" neural network:
    
    1. The first layer effectively consists of the set of weights and biases applied to X and passed through ReLUs. The output 
    of this layer is fed to the next one, but is not observable outside the network, hence it is known as a hidden layer.
    2. The second layer consists of the weights and biases applied to these intermediate outputs, followed by the softmax 
    function to generate probabilities.

### 8. 

A Rectified linear unit (ReLU) is type of [activation function](https://en.wikipedia.org/wiki/Activation_function) that 
is defined as ```f(x) = max(0, x)```. The function returns 0 if ```x``` is negative, otherwise it returns ```x```. 
TensorFlow provides the ReLU function as ```tf.nn.relu()```, as shown below.

```
# Hidden Layer with ReLU activation function
hidden_layer = tf.add(tf.matmul(features, hidden_weights), hidden_biases)
hidden_layer = tf.nn.relu(hidden_layer)

output = tf.add(tf.matmul(hidden_layer, output_weights), output_biases)
```

The above code applies the ```tf.nn.relu()``` function to the ```hidden_layer```, effectively turning off any negative 
weights and acting like an on/off switch. Adding additional layers, like the ```output``` layer, after an activation 
function turns the model into a nonlinear function. This nonlinearity allows the network to solve more complex problems.

###### Quiz

![alt text](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/May/5af36c2a_relu-network/relu-network.png)

In this quiz, you'll use TensorFlow's ReLU function to turn the linear model below into a nonlinear model.

![quiz](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/09_01.PNG)

[Here full code](https://github.com/swoldetsadick/sdce/blob/master/Notebooks/Lesson_9/01/Quiz.md)

### 9. No neurones

[![No neurones](http://img.youtube.com/vi/svA0HOjFFl0/0.jpg)](https://youtu.be/svA0HOjFFl0 "No neurones")

### 10. The chain rule

[![The chain rule](http://img.youtube.com/vi/DxOg_olir0k/0.jpg)](https://youtu.be/DxOg_olir0k "The chain rule")

### 11. Backpropagation

[![Backpropagation](http://img.youtube.com/vi/wSXcrBbY8oE/0.jpg)](https://youtu.be/wSXcrBbY8oE "Backpropagation")

### 12. Deep neural networks in TF

[![](http://img.youtube.com/vi//0.jpg)]( "")

### 13. Training deep learning network

[![Training deep learning network](http://img.youtube.com/vi/CsB7yUtMJyk/0.jpg)](https://youtu.be/CsB7yUtMJyk "Training deep learning network")

### 14. Save and restore TF models

[![](http://img.youtube.com/vi//0.jpg)]( "")

### 15. Finetuning

[![](http://img.youtube.com/vi//0.jpg)]( "")

### 1. 

[![](http://img.youtube.com/vi//0.jpg)]( "")

### 1. 

[![](http://img.youtube.com/vi//0.jpg)]( "")

### 1. 

[![](http://img.youtube.com/vi//0.jpg)]( "")

### 1. 

[![](http://img.youtube.com/vi//0.jpg)]( "")