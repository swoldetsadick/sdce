# Lesson VII: MiniFlow

### 1. Introduction to miniflow

[![Introduction to miniflow](http://img.youtube.com/vi/FxmB3Q308h0/0.jpg)](https://youtu.be/FxmB3Q308h0 "Introduction to miniflow")

### 2. Introduction

In this lab, you’ll build a library called MiniFlow which will be your own version of [TensorFlow](http://tensorflow.org/)! (link for [China](http://tensorfly.cn/))

TensorFlow is one of the most popular open source neural network libraries, built by the team at Google Brain over just 
the last few years.

Following this lab, you'll spend the remainder of this module actually working with open-source deep learning libraries 
like [TensorFlow](http://tensorflow.org/) and [Keras](https://keras.io/). So why bother building MiniFlow? Good question, glad you asked!

The goal of this lab is to demystify two concepts at the heart of neural networks - backpropagation and differentiable 
graphs.

Backpropagation is the process by which neural networks update the weights of the network over time. (You may have seen 
it in [this video](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/83a4e710-a69e-4ce9-9af9-939307c0711b/concepts/4cc13714-37d7-4705-a714-314ede5290b5#) earlier.)

Differentiable graphs are graphs where the nodes are [differentiable functions](https://en.wikipedia.org/wiki/Differentiable_function). 
They are also useful as visual aids for understanding and calculating complicated derivatives. This is the fundamental 
abstraction of TensorFlow - it's a framework for creating differentiable graphs.

With graphs and backpropagation, you will be able to create your own nodes and properly compute the derivatives. Even 
more importantly, you will be able to think and reason in terms of these graphs.

Now, let's take the first peek under the hood...

### 3. Graphs

###### What is a Neural Network?

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/58375218_example-neural-network/example-neural-network.png)

A neural network is a graph of mathematical functions such as [linear combinations](https://en.wikipedia.org/wiki/Linear_combination) 
and activation functions. The graph consists of **nodes**, and **edges**.

Nodes in each layer (except for nodes in the input layer) perform mathematical functions using inputs from nodes in the 
previous layers. For example, a node could represent f(x, y) = x + y, where x and y are input values from 
nodes in the previous layer.

Similarly, each node creates an output value which may be passed to nodes in the next layer. The output value from the 
output layer does not get passed to a future layer (because it is the final layer).

Layers between the input layer and the output layer are called **hidden layers**.

The edges in the graph describe the connections between the nodes, along which the values flow from one layer to the 
next. These edges can also apply operations to the values that flow along them, such as multiplying by weights and 
adding biases. MiniFlow won't use a separate class for edges - instead, its nodes will perform both their own 
calculations and those of their input edges. This will be more clear as you go through these lessons.

###### Forward Propagation

By propagating values from the first layer (the input layer) through all the mathematical functions represented by each 
node, the network outputs a value. This process is called a forward pass.

Here's an example of a simple **forward pass**.

[![forward pass](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/07_01.PNG)](https://s3.amazonaws.com/content.udacity-data.com/courses/carnd/videos/input-to-output-2.mp4 "forward pass")

Notice that the output layer performs a mathematical function, addition, on its inputs. There is no hidden layer.

###### Graphs

The nodes and edges create a graph structure. Though the example above is fairly simple, it isn't hard to imagine that 
increasingly complex graphs can calculate . . . well . . . almost anything.

There are generally two steps to create neural networks:

1. Define the graph of nodes and edges.
2. Propagate values through the graph.

```MiniFlow``` works the same way. You'll define the nodes and edges of your network with one method and then propagate values 
through the graph with another method. 

```MiniFlow``` comes with some starter code to help you out. We'll take a look on the 
next page, but first, let's test your intuition.

###### Graph Quiz

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/58375a5a_addition-graph/addition-graph.png)

![alt text](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/07_02.PNG)

### 4. 

[![Starting ML](http://img.youtube.com/vi/UIycORUrPww/0.jpg)](https://youtu.be/UIycORUrPww "Starting ML")

### 5. 

[![Neuronal Networks: Intuition](http://img.youtube.com/vi/UKEIHK5IifI/0.jpg)](https://youtu.be/UKEIHK5IifI "Neuronal Networks: Intuition")

### 6. 

[![Intro to DL](http://img.youtube.com/vi/uyLRFMI4HkA/0.jpg)](https://youtu.be/uyLRFMI4HkA "Intro to DL")

### 7. 

[![Starting ML](http://img.youtube.com/vi/UIycORUrPww/0.jpg)](https://youtu.be/UIycORUrPww "Starting ML")

### 8. 

[![Starting ML](http://img.youtube.com/vi/UIycORUrPww/0.jpg)](https://youtu.be/UIycORUrPww "Starting ML")

### 9. 

[![Neuronal Networks: Intuition](http://img.youtube.com/vi/UKEIHK5IifI/0.jpg)](https://youtu.be/UKEIHK5IifI "Neuronal Networks: Intuition")

### 10. 

[![Intro to DL](http://img.youtube.com/vi/uyLRFMI4HkA/0.jpg)](https://youtu.be/uyLRFMI4HkA "Intro to DL")

### 11. 

[![Starting ML](http://img.youtube.com/vi/UIycORUrPww/0.jpg)](https://youtu.be/UIycORUrPww "Starting ML")

### 12. 

[![Starting ML](http://img.youtube.com/vi/UIycORUrPww/0.jpg)](https://youtu.be/UIycORUrPww "Starting ML")

### 13. 

[![Neuronal Networks: Intuition](http://img.youtube.com/vi/UKEIHK5IifI/0.jpg)](https://youtu.be/UKEIHK5IifI "Neuronal Networks: Intuition")

### 14. 

[![Intro to DL](http://img.youtube.com/vi/uyLRFMI4HkA/0.jpg)](https://youtu.be/uyLRFMI4HkA "Intro to DL")

### 15. 

[![Intro to DL](http://img.youtube.com/vi/uyLRFMI4HkA/0.jpg)](https://youtu.be/uyLRFMI4HkA "Intro to DL")

### 16. 

[![Neuronal Networks: Intuition](http://img.youtube.com/vi/UKEIHK5IifI/0.jpg)](https://youtu.be/UKEIHK5IifI "Neuronal Networks: Intuition")

### 17. 

[![Intro to DL](http://img.youtube.com/vi/uyLRFMI4HkA/0.jpg)](https://youtu.be/uyLRFMI4HkA "Intro to DL")

### 18. 

[![Starting ML](http://img.youtube.com/vi/UIycORUrPww/0.jpg)](https://youtu.be/UIycORUrPww "Starting ML")
