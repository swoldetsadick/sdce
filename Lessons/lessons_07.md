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

### 4. MiniFlow architecture

Let's consider how to implement this graph structure in ```MiniFlow```. We'll use a Python class to represent a generic 
node.

````python
class Node(object):
    def __init__(self):
        # Properties will go here!
````

We know that each node might receive input from multiple other nodes. We also know that each node creates a single 
output, which will likely be passed to other nodes. Let's add two lists: one to store references to the inbound nodes, 
and the other to store references to the outbound nodes.

````python
class Node(object):
    def __init__(self, inbound_nodes=[]):
        # Node(s) from which this Node receives values
        self.inbound_nodes = inbound_nodes
        # Node(s) to which this Node passes values
        self.outbound_nodes = []
        # For each inbound_node, add the current Node as an outbound_node.
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)
````

Each node will eventually calculate a value that represents its output. Let's initialize the ```value``` to ```None``` 
to indicate that it exists but hasn't been set yet.

````python
class Node(object):
    def __init__(self, inbound_nodes=[]):
        # Node(s) from which this Node receives values
        self.inbound_nodes = inbound_nodes
        # Node(s) to which this Node passes values
        self.outbound_nodes = []
        # For each inbound_node, add the current Node as an outbound_node.
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)
        # A calculated value
        self.value = None
````

Each node will need to be able to pass values forward and perform backpropagation (more on that later). For now, let's 
add a placeholder method for forward propagation. We'll deal with backpropagation later on.

````python
class Node(object):
    def __init__(self, inbound_nodes=[]):
        # Node(s) from which this Node receives values
        self.inbound_nodes = inbound_nodes
        # Node(s) to which this Node passes values
        self.outbound_nodes = []
        # For each inbound_node, add the current Node as an outbound_node.
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)
        # A calculated value
        self.value = None

    def forward(self):
        """
        Forward propagation.

        Compute the output value based on `inbound_nodes` and
        store the result in self.value.
        """
        raise NotImplemented
````

###### Nodes that Calculate

While ```Node``` defines the base set of properties that every node holds, only specialized [subclasses]() of ```Node``` 
will end up in the graph. As part of this lab, you'll build the subclasses of ```Node``` that can perform calculations 
and hold values. For example, consider the ```Input``` subclass of ```Node```.

````python
class Input(Node):
    def __init__(self):
        # An Input node has no inbound nodes,
        # so no need to pass anything to the Node instantiator.
        Node.__init__(self)

    # NOTE: Input node is the only node where the value
    # may be passed as an argument to forward().
    #
    # All other node implementations should get the value
    # of the previous node from self.inbound_nodes
    #
    # Example:
    # val0 = self.inbound_nodes[0].value
    def forward(self, value=None):
        # Overwrite the value if one is passed in.
        if value is not None:
            self.value = value
````

Unlike the other subclasses of ```Node```, the ```Input``` subclass does not actually calculate anything. The ```Input``` 
subclass just holds a ```value```, such as a data feature or a model parameter (weight/bias).

You can set ```value``` either explicitly or with the ```forward()``` method. This value is then fed through the rest of 
the neural network.

###### The Add Subclass

```Add```, which is another subclass of ```Node```, actually can perform a calculation (addition).

````python
class Add(Node):
    def __init__(self, x, y):
        Node.__init__(self, [x, y])

    def forward(self):
        """
        You'll be writing code here in the next quiz!
        """
````

Notice the difference in the ```__init__``` method, ```Add.__init__(self, [x, y])```. Unlike the ```Input``` class, 
which has no inbound nodes, the ```Add``` class takes 2 inbound nodes, ```x``` and ```y```, and adds the values of those 
nodes.

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
