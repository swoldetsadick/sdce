"""  """


class Node(object):
    """  This class represents a node in MiniFlow architecture. """
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
        """ Forward propagation.

        Compute the output value based on `inbound_nodes` and store the result in self.value.

        :return:
        """
        raise NotImplemented


class Input(Node):
    """  This class represents a Inputs node in MiniFlow architecture. """
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
        """ Forward propagation.

        Compute the output value based on `inbound_nodes` or `passed arguments` and store the result in self.value.

        :return:
        """
        # Overwrite the value if one is passed in.
        if value is not None:
            self.value = value


class Add(Node):
    """  This class represents a add node in MiniFlow architecture. """
    # def __init__(self, x, y):
    # You could access `x` and `y` in forward with
    # self.inbound_nodes[0] (`x`) and self.inbound_nodes[1] (`y`)
    # Node.__init__(self, [x, y])
    # You may need to change this...
    def __init__(self, *inputs):
        Node.__init__(self, inputs)

    def forward(self):
        """
        Set the value of this node (`self.value`) to the sum of its inbound_nodes.
        Remember to grab the value of each inbound_node to sum!

        Your code here!
        """
        self.value = sum([n.value for n in self.inbound_nodes])


class Mul(Node):
    """  This class represents a multiplication node in MiniFlow architecture. """
    # def __init__(self, x, y):
    # You could access `x` and `y` in forward with
    # self.inbound_nodes[0] (`x`) and self.inbound_nodes[1] (`y`)
    # Node.__init__(self, [x, y])
    # You may need to change this...
    def __init__(self, *inputs):
        Node.__init__(self, inputs)

    def forward(self):
        """
        Set the value of this node (`self.value`) to the sum of its inbound_nodes.
        Remember to grab the value of each inbound_node to sum!

        Your code here!
        """
        from numpy import prod as npprod
        self.value = npprod([n.value for n in self.inbound_nodes])


class Linear(Node):
    """ This class represents a linear node in MiniFlow architecture. """
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
        # from numpy import multiply as npmultiply
        # self.value = sum(npmultiply(self.inbound_nodes[0].value, self.inbound_nodes[1].value)) + self.inbound_nodes[2].value
        from numpy import dot as npdot
        self.value = npdot(self.inbound_nodes[0].value, self.inbound_nodes[1].value) + self.inbound_nodes[2].value


class Sigmoid(Node):
    """
    You need to fix the `_sigmoid` and `forward` methods.
    """
    def __init__(self, node):
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        """
        This method is separate from `forward` because it
        will be used later with `backward` as well.

        `x`: A numpy array-like object.

        Return the result of the sigmoid function.

        Your code here!
        """
        from numpy import exp as npexp
        return 1/(1 + npexp(-x))

    def forward(self):
        """
        Set the value of this node to the result of the
        sigmoid function, `_sigmoid`.

        Your code here!
        """
        # This is a dummy value to prevent numpy errors
        # if you test without changing this method.
        sigmoider = lambda t: self._sigmoid(t)
        self.value = [sigmoider(n) for n in self.inbound_nodes[0].value]


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


def main(which):
    """ This is main function. """
    if which == 1:
        x, y = Input(), Input()
        f = Add(x, y)
        feed_dict = {x: 10, y: 5}
        sorted_nodes = topological_sort(feed_dict)
        output = forward_pass(f, sorted_nodes)
        # NOTE: because topological_sort sets the values for the `Input` nodes we could also access
        # the value for x with x.value (same goes for y).
        print("{} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], output))
    elif which == 2:
        x, y, z = Input(), Input(), Input()
        f = Add(x, y, z)
        feed_dict = {x: 4, y: 5, z: 10}
        graph = topological_sort(feed_dict)
        output = forward_pass(f, graph)
        print("{} + {} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], output))
    elif which == 3:
        x, y, z = Input(), Input(), Input()
        f = Mul(x, y, z)
        feed_dict = {x: 4, y: 5, z: 10}
        graph = topological_sort(feed_dict)
        output = forward_pass(f, graph)
        print("{} * {} * {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], output))
    elif which == 4:
        inputs, weights, bias = Input(), Input(), Input()
        f = Linear(inputs, weights, bias)
        feed_dict = {
            inputs: [6, 14, 3],
            weights: [0.5, 0.25, 1.4],
            bias: 2
        }
        graph = topological_sort(feed_dict)
        output = forward_pass(f, graph)
        print(output)  # should be 12.7 with this example
    elif which == 5:
        import numpy as np
        X, W, b = Input(), Input(), Input()
        f = Linear(X, W, b)
        g = Sigmoid(f)
        X_ = np.array([[-1., -2.], [-1, -2]])
        W_ = np.array([[2., -3], [2., -3]])
        b_ = np.array([-3., -5])
        feed_dict = {X: X_, W: W_, b: b_}
        graph = topological_sort(feed_dict)
        output = forward_pass(g, graph)
        """
        Output should be:
        [[  1.23394576e-04   9.82013790e-01]
         [  1.23394576e-04   9.82013790e-01]]
        """
        print(output)
    return None


if __name__ == "__main__":
    main(which=1)
    main(which=2)
    main(which=3)
    main(which=4)
    main(which=5)
