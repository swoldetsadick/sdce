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

### 5. Forward propagation

```MiniFlow``` has two methods to help you define and then run values through your graphs: ```topological_sort()``` and 
```forward_pass()```.

In order to define your network, you'll need to define the order of operations for your nodes. Given that the input to 
some node depends on the outputs of others, you need to flatten the graph in such a way where all the input dependencies 
for each node are resolved before trying to run its calculation. This is a technique called a [topological sort](https://en.wikipedia.org/wiki/Topological_sorting).

![alt text](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/07_04.jpeg)

The ```topological_sort()``` function implements topological sorting using [Kahn's Algorithm](https://en.wikipedia.org/wiki/Topological_sorting#Kahn.27s_algorithm). The details of this method are 
not important, the result is; ```topological_sort()``` returns a sorted list of nodes in which all of the calculations 
can run in series. ```topological_sort()``` takes in a ```feed_dict```, which is how we initially set a value for an 
```Input``` node. The ```feed_dict``` is represented by the Python dictionary data structure. Here's an example use 
case:

````python
# Define 2 `Input` nodes.
x, y = Input(), Input()

# Define an `Add` node, the two above `Input` nodes being the input.
add = Add(x, y)

# The value of `x` and `y` will be set to 10 and 20 respectively.
feed_dict = {x: 10, y: 20}

# Sort the nodes with topological sort.
sorted_nodes = topological_sort(feed_dict=feed_dict)
````

(You can find the source code for ```topological_sort()``` in miniflow.py in the programming quiz below.)

The other method at your disposal is ```forward_pass()```, which actually runs the network and outputs a value.

````python
def forward_pass(output_node, sorted_nodes):
    """
    Performs a forward pass through a list of sorted nodes.

    Arguments:

        `output_node`: The output node of the graph (no outgoing edges).
        `sorted_nodes`: a topologically sorted list of nodes.

    Returns the output node's value
    """

    for n in sorted_nodes:
        n.forward()

    return output_node.value
````

###### Quiz 1 - Passing Values Forward

Create and run this graph!

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/58375a5a_addition-graph/addition-graph.png)

###### Setup

Review ```nn.py``` and ```miniflow.py```.

The neural network architecture is already there for you in nn.py. It's your job to finish ```MiniFlow``` to make it work.

For this quiz, I want you to:

1. Open ```nn.py``` below. **You don't need to change anything**. I just want you to see how ```MiniFlow``` works.
2. Open ```miniflow.py```. **Finish the** ```forward``` **method on the** ```Add``` **class. All that's required to pass 
this quiz is a correct implementation of** ```forward```**.**
3. Test your network by hitting "Test Run!" When the output looks right, hit "Submit!"

(You'll find the solution on the next page.)

_nn.py_
````python
"""
This script builds and runs a graph with miniflow.

There is no need to change anything to solve this quiz!

However, feel free to play with the network! Can you also
build a network that solves the equation below?

(x + y) + y
"""

from miniflow import *

x, y = Input(), Input()

f = Add(x, y)

feed_dict = {x: 10, y: 5}

sorted_nodes = topological_sort(feed_dict)
output = forward_pass(f, sorted_nodes)

# NOTE: because topological_sort sets the values for the `Input` nodes we could also access
# the value for x with x.value (same goes for y).
print("{} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], output))
````

_miniflow.py_
````python
"""
You need to change the Add() class below.
"""

class Node(object):
    def __init__(self, inbound_nodes=[]):
        # Nodes from which this Node receives values
        self.inbound_nodes = inbound_nodes
        # Nodes to which this Node passes values
        self.outbound_nodes = []
        # A calculated value
        self.value = None
        # Add this node as an outbound node on its inputs.
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)

    # These will be implemented in a subclass.
    def forward(self):
        """
        Forward propagation.

        Compute the output value based on `inbound_nodes` and
        store the result in self.value.
        """
        raise NotImplemented


class Input(Node):
    def __init__(self):
        # an Input node has no inbound nodes,
        # so no need to pass anything to the Node instantiator
        Node.__init__(self)

    # NOTE: Input node is the only node that may
    # receive its value as an argument to forward().
    #
    # All other node implementations should calculate their
    # values from the value of previous nodes, using
    # self.inbound_nodes
    #
    # Example:
    # val0 = self.inbound_nodes[0].value
    def forward(self, value=None):
        if value is not None:
            self.value = value


class Add(Node):
    def __init__(self, x, y):
        # You could access `x` and `y` in forward with
        # self.inbound_nodes[0] (`x`) and self.inbound_nodes[1] (`y`)
        Node.__init__(self, [x, y])

    def forward(self):
        """
        Set the value of this node (`self.value`) to the sum of its inbound_nodes.
        Remember to grab the value of each inbound_node to sum!

        Your code here!
        """
        somme = 0
        for n in self.inbound_nodes:
            somme += n.value
        self.value = somme


def topological_sort(feed_dict):
    """
    Sort generic nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` node and the value is the respective value feed to that node.

    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_pass(output_node, sorted_nodes):
    """
    Performs a forward pass through a list of sorted nodes.

    Arguments:

        `output_node`: A node in the graph, should be the output node (have no outgoing edges).
        `sorted_nodes`: A topologically sorted list of nodes.

    Returns the output Node's value
    """

    for n in sorted_nodes:
        n.forward()

    return output_node.value
````

![alt text](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/07_03.PNG)

### 6. Forward propagation solution

Here's my solution (I'm just showing ```forward``` method of the ```Add``` class):

```
def forward(self):
    x_value = self.inbound_nodes[0].value
    y_value = self.inbound_nodes[1].value
    self.value = x_value + y_value
```

While this looks simple, I want to show you why I used ```x_value``` and ```y_value``` from the ```inbound_nodes``` 
array. Let's take a look at the start with ```Node```'s constructor:

```
class Node(object):
    def __init__(self, inbound_nodes=[]):
        # Node(s) from which this Node receives values.
        self.inbound_nodes = inbound_nodes
        # Removed everything else for brevity.
```

```inbound_nodes``` are set when the ```Node``` is instantiated.

Of course, you weren't using ```Node``` directly, rather you used ```Add```, which is a subclass of ```Node```. 
```Add```'s constructor is responsible for passing the ```inbound_nodes``` to ```Node```, which happens here:

```
class Add(Node):
    def __init__(self, x, y):
         Node.__init__(self, [x, y]) # calls Node's constructor
    ...
```

Lastly, there's the question of why ```node.value``` holds the value of the inputs. For each node of the ```Input()``` 
class, the nodes are set directly when you run ```topological_sort```:

```
def topological_sort(feed_dict):
    ...
    if isinstance(n, Input):
        n.value = feed_dict[n]
    ...
```

For other classes, the value of node.value is set in the forward pass:

```
def forward_pass(output_node, sorted_nodes):
    ...
    for n in sorted_nodes:
        n.forward()
    ...
```

And that's it for addition!

Keep going to make ```MiniFlow``` more capable.

###### Bonus Challenges!

These are **ungraded** challenges as they are more of a test of your Python skills than neural network skills.

1. Can you make ```Add``` accept any number of inputs? Eg. ```Add(x, y, z)```.
2. Can you make a ```Mul``` class that multiplies _n_ inputs?

_nn.py_
````python
"""
No need to change anything here!

If all goes well, this should work after you
modify the Add class in miniflow.py.
"""

from miniflow import *

x, y, z = Input(), Input(), Input()

f = Add(x, y, z)

feed_dict = {x: 4, y: 5, z: 10}

graph = topological_sort(feed_dict)
output = forward_pass(f, graph)

# should output 19
print("{} + {} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], output))
````

_miniflow.py_
````python
"""
Bonus Challenge!

Write your code in Add (scroll down).
"""

class Node(object):
    def __init__(self, inbound_nodes=[]):
        # Nodes from which this Node receives values
        self.inbound_nodes = inbound_nodes
        # Nodes to which this Node passes values
        self.outbound_nodes = []
        # A calculated value
        self.value = None
        # Add this node as an outbound node on its inputs.
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)

    # These will be implemented in a subclass.
    def forward(self):
        """
        Forward propagation.

        Compute the output value based on `inbound_nodes` and
        store the result in self.value.
        """
        raise NotImplemented


class Input(Node):
    def __init__(self):
        # An Input Node has no inbound nodes,
        # so no need to pass anything to the Node instantiator
        Node.__init__(self)

    # NOTE: Input Node is the only Node where the value
    # may be passed as an argument to forward().
    #
    # All other Node implementations should get the value
    # of the previous nodes from self.inbound_nodes
    #
    # Example:
    # val0 = self.inbound_nodes[0].value
    def forward(self, value=None):
        # Overwrite the value if one is passed in.
        if value is not None:
            self.value = value


"""
Can you augment the Add class so that it accepts
any number of nodes as input?

Hint: this may be useful:
https://docs.python.org/3/tutorial/controlflow.html#unpacking-argument-lists
"""
class Add(Node):
    # You may need to change this...
    def __init__(self, *inputs):
        Node.__init__(self, inputs)

    def forward(self):
        """
        For reference, here's the old way from the last
        quiz. You'll want to write code here.
        """
        self.value = sum([n.value for n in self.inbound_nodes])

def topological_sort(feed_dict):
    """
    Sort the nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` Node and the value is the respective value feed to that Node.

    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_pass(output_node, sorted_nodes):
    """
    Performs a forward pass through a list of sorted nodes.

    Arguments:

        `output_node`: A node in the graph, should be the output node (have no outgoing edges).
        `sorted_nodes`: A topologically sorted list of nodes.

    Returns the output Node's value
    """

    for n in sorted_nodes:
        n.forward()

    return output_node.value
````

![alt text](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/07_05.PNG)

### 7. Learning and Loss

Like ```MiniFlow``` in its current state, neural networks take inputs and produce outputs. But unlike ```MiniFlow``` in its current 
state, neural networks can improve the accuracy of their outputs over time (it's hard to imagine improving the accuracy 
of ```Add``` over time!). To explore why accuracy matters, I want you to first implement a trickier (and more useful!) node 
than ```Add```: the ```Linear``` node.

###### The Linear Function

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/October/58118748_vlcsnap-2016-10-26-21h47m38s028/vlcsnap-2016-10-26-21h47m38s028.png)

Think back to the Neural Networks lesson with Luis and Matt. A simple artificial neuron depends on three components:

* inputs, _x_ (vector)
* weights, _&#969;_ (vector)
* bias, _b_ (scalar)

The output, _o_, is just the weighted sum of the inputs plus the bias:

_o_ = &#8721;<sub>i</sub> _x_<sub>_i_</sub> &#8901; _&#969;_<sub>_i_</sub> + _b_

Equation (1)

Remember, by varying the weights, you can vary the amount of influence any given input has on the output. The learning 
aspect of neural networks takes place during a process known as backpropagation. In backpropogation, the network 
modifies the weights to improve the network's output accuracy. You'll be applying all of this shortly.

In this next quiz, you'll try to build a linear neuron that generates an output by applying a simplified version of 
Equation (1). ```Linear``` should take an list of inbound nodes of length n, a list of weights of length n, and a bias.

###### Instructions

1. Open nn.py below. Read through the neural network to see the expected output of ```Linear```.
2. Open miniflow.py below. Modify ```Linear```, which is a subclass of ```Node```, to generate an output with Equation (1).

(Hint: you could use ```numpy``` to solve this quiz if you'd like, but it's possible to solve this with vanilla Python.)

_nn.py_
````python
"""
NOTE: Here we're using an Input node for more than a scalar.
In the case of weights and inputs the value of the Input node is
actually a python list!

In general, there's no restriction on the values that can be passed to an Input node.
"""
from miniflow import *

inputs, weights, bias = Input(), Input(), Input()

f = Linear(inputs, weights, bias)

feed_dict = {
    inputs: [6, 14, 3],
    weights: [0.5, 0.25, 1.4],
    bias: 2
}

graph = topological_sort(feed_dict)
output = forward_pass(f, graph)

print(output) # should be 12.7 with this example
````

_miniflow.py_
````python
"""
Write the Linear#forward method below!
"""


class Node:
    def __init__(self, inbound_nodes=[]):
        # Nodes from which this Node receives values
        self.inbound_nodes = inbound_nodes
        # Nodes to which this Node passes values
        self.outbound_nodes = []
        # A calculated value
        self.value = None
        # Add this node as an outbound node on its inputs.
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)

    # These will be implemented in a subclass.
    def forward(self):
        """
        Forward propagation.

        Compute the output value based on `inbound_nodes` and
        store the result in self.value.
        """
        raise NotImplemented


class Input(Node):
    def __init__(self):
        # An Input Node has no inbound nodes,
        # so no need to pass anything to the Node instantiator
        Node.__init__(self)

        # NOTE: Input Node is the only Node where the value
        # may be passed as an argument to forward().
        #
        # All other Node implementations should get the value
        # of the previous nodes from self.inbound_nodes
        #
        # Example:
        # val0 = self.inbound_nodes[0].value
    def forward(self, value=None):
        # Overwrite the value if one is passed in.
        if value is not None:
            self.value = value


class Linear(Node):
    def __init__(self, inputs, weights, bias):
        Node.__init__(self, [inputs, weights, bias])

        # NOTE: The weights and bias properties here are not
        # numbers, but rather references to other nodes.
        # The weight and bias values are stored within the
        # respective nodes.

    def forward(self):
        """
        Set self.value to the value of the linear function output.

        Your code goes here!
        """
        from numpy import multiply as npmultiply
        self.value = sum(npmultiply(self.inbound_nodes[0].value, self.inbound_nodes[1].value)) \
                     + self.inbound_nodes[2].value



def topological_sort(feed_dict):
    """
    Sort the nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` Node and the value is the respective value feed to that Node.

    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_pass(output_node, sorted_nodes):
    """
    Performs a forward pass through a list of sorted nodes.

    Arguments:

        `output_node`: A node in the graph, should be the output node (have no outgoing edges).
        `sorted_nodes`: A topologically sorted list of nodes.

    Returns the output Node's value
    """

    for n in sorted_nodes:
        n.forward()

    return output_node.value
````

![alt text](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/07_06.PNG)

### 8. 

###### Solution to Linear Node

Here's my solution to the last quiz:

````python
class Linear(Node):
    def __init__(self, inputs, weights, bias):
        Node.__init__(self, [inputs, weights, bias])

    def forward(self):
        """
        Set self.value to the value of the linear function output.

        Your code goes here!
        """
        inputs = self.inbound_nodes[0].value
        weights = self.inbound_nodes[1].value
        bias = self.inbound_nodes[2].value
        self.value = bias
        for x, w in zip(inputs, weights):
            self.value += x * w
````

In the solution, I set ```self.value``` to the bias and then loop through the inputs and weights, adding each weighted 
input to ```self.value```. Notice calling ```.value``` on ```self.inbound_nodes[0]``` or ```self.inbound_nodes[1]``` 
gives us a list.

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/October/580a9a2d_screen-shot-2016-10-21-at-15.43.05/screen-shot-2016-10-21-at-15.43.05.png)

[Linear algebra](https://www.khanacademy.org/math/linear-algebra) nicely reflects the idea of transforming values 
between layers in a graph. In fact, the concept of a [transform](https://www.khanacademy.org/math/linear-algebra/matrix-transformations/linear-transformations/v/vector-transformations) 
does exactly what a layer should do - it converts inputs to outputs in many dimensions.

Let's go back to our equation for the output.

_o_ = &#8721;<sub>i</sub> _x_<sub>_i_</sub> &#8901; _&#969;_<sub>_i_</sub> + _b_

Equation (1)

For the rest of this section we'll denote _x_ as _X_ and _w_ as _W_ since they are now matrices, and bb is now a vector 
instead of a scalar.

Consider a ```Linear``` node with 1 input and k outputs (mapping 1 input to k outputs). In this context an input/output 
is synonymous with a feature.

In this case _X_ is a 1 by 1 matrix.

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581f9571_newx/newx.png)

_W_ becomes a 1 by k matrix (looks like a row).

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581f9571_neww/neww.png)

The result of the matrix multiplication of _X_ and _W_ is a 1 by k matrix. Since bb is also a 1 by k row matrix (1 bias 
per output), _b_ is added to the output of the _X_ and _W_ matrix multiplication.

What if we are mapping n inputs to k outputs?

Then _X_  is now a 1 by n matrix and _W_ is a n by k matrix. The result of the matrix multiplication is still a 1 by k 
matrix so the use of the biases remain the same.

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581f9570_newx-1n/newx-1n.png)

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5820be00_w-nk/w-nk.png)

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581a94e5_b-1byk/b-1byk.png)

Let's take a look at an example of n input features. Consider an 28px by 28px greyscale image, as is in the case of 
images in the MNIST dataset. We can reshape (flatten) the image such that it's a 1 by 784 matrix, where n = 784. Each 
pixel is an input/feature. Here's an animated example emphasizing a pixel is a feature.

[![zoom-in](http://img.youtube.com/vi/qE5YYXtPq9U/0.jpg)](https://youtu.be/qE5YYXtPq9U "zoom-in")

In practice, it's common to feed in multiple data examples in each forward pass rather than just 1. The reasoning for 
this is the examples can be processed in parallel, resulting in big performance gains. The number of examples is called 
the batch size. Common numbers for the batch size are 32, 64, 128, 256, 512. Generally, it's the most we can comfortably 
fit in memory.

What does this mean for _X_, _W_ and _b_?

_X_ becomes a m by n matrix (where m is the batch size, by the number of input features, n) and _W_ and _b_ remain the 
same. The result of the matrix multiplication is now m by k (batch size by number of outputs), so the addition of _b_ is 
broadcast over each row.

![alt txet](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5820bdff_x-mn/x-mn.png)

In the context of MNIST each row of _X_ is an image reshaped/flattened from 28 by 28 to 1 by 784.
Equation (1) turns into:

![alt txet](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5820f6bc_z/z.png)

Equation (2) can also be viewed as _Z = XW + B_ where _B_ is the biases vector, _b_, stacked m times. Due to 
broadcasting it's abbreviated to _Z = XW + b_.

I want you to rebuild ```Linear``` to handle matrices and vectors using the venerable Python math package ```numpy``` to 
make your life easier. ```numpy``` is often abbreviated as ```np```, so we'll refer to it as ```np``` when referring to 
code.

I used ```np.array``` ([documentation](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html)) to create the matrices and vectors. You'll want to use ```np.dot```, which functions 
as matrix multiplication for 2D arrays ([documentation](https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html)), to multiply the input and weights matrices from Equation (2). 
It's also worth noting that numpy actually overloads the ```__add__``` operator so you can use it directly with np.array 
```(eg. np.array() + np.array())```.

###### Instructions

1. Open nn.py. See how the neural network implements the ```Linear``` node.
2. Open miniflow.py. Implement Equation (2) within the forward pass for the ```Linear``` node.
3. Test your work!

_nn.py_
````python
"""
The setup is similar to the prevous `Linear` node you wrote
except you're now using NumPy arrays instead of python lists.

Update the Linear class in miniflow.py to work with
numpy vectors (arrays) and matrices.

Test your code here!
"""

import numpy as np
from miniflow import *

X, W, b = Input(), Input(), Input()

f = Linear(X, W, b)

X_ = np.array([[-1., -2.], [-1, -2]])
W_ = np.array([[2., -3], [2., -3]])
b_ = np.array([-3., -5])

feed_dict = {X: X_, W: W_, b: b_}

graph = topological_sort(feed_dict)
output = forward_pass(f, graph)

"""
Output should be:
[[-9., 4.],
[-9., 4.]]
"""
print(output)
````

_.py_
````python
"""
Modify Linear#forward so that it linearly transforms
input matrices, weights matrices and a bias vector to
an output.
"""

import numpy as np


class Node(object):
    def __init__(self, inbound_nodes=[]):
        self.inbound_nodes = inbound_nodes
        self.value = None
        self.outbound_nodes = []
        for node in inbound_nodes:
            node.outbound_nodes.append(self)

    def forward():
        raise NotImplementedError


class Input(Node):
    """
    While it may be strange to consider an input a node when
    an input is only an individual node in a node, for the sake
    of simpler code we'll still use Node as the base class.

    Think of Input as collating many individual input nodes into
    a Node.
    """
    def __init__(self):
        # An Input node has no inbound nodes,
        # so no need to pass anything to the Node instantiator
        Node.__init__(self)

    def forward(self):
        # Do nothing because nothing is calculated.
        pass


class Linear(Node):
    def __init__(self, X, W, b):
        # Notice the ordering of the input nodes passed to the
        # Node constructor.
        Node.__init__(self, [X, W, b])

    def forward(self):
        """
        Set the value of this node to the linear transform output.

        Your code goes here!
        """


def topological_sort(feed_dict):
    """
    Sort the nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` Node and the value is the respective value feed to that Node.

    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_pass(output_node, sorted_nodes):
    """
    Performs a forward pass through a list of sorted Nodes.

    Arguments:

        `output_node`: A Node in the graph, should be the output node (have no outgoing edges).
        `sorted_nodes`: a topologically sorted list of nodes.

    Returns the output node's value
    """

    for n in sorted_nodes:
        n.forward()

    return output_node.value
````

![alt text](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/07_07.PNG)

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
