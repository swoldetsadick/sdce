# Lesson X: Convolutional neural networks

### 1. CNNs have taken over

[![CNNs have taken over](http://img.youtube.com/vi/yNOHThuy2UU/0.jpg)](https://youtu.be/yNOHThuy2UU "CNNs have taken over")

### 2. Introduction to CNNs

[![Introduction to CNNs](http://img.youtube.com/vi/B61jxZ4rkMs/0.jpg)](https://youtu.be/B61jxZ4rkMs "Introduction to CNNs")

### 3. Color

[![Color](http://img.youtube.com/vi/BdQccpMwk80/0.jpg)](https://youtu.be/BdQccpMwk80 "Color")

![alt text](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/10_01.PNG)

**Answer:**

[![Color answer](http://img.youtube.com/vi/BdQccpMwk80/0.jpg)](https://youtu.be/xpyldyLlMFg "Color answer")

### 4. Statistical Invariance

[![Statistical Invariance](http://img.youtube.com/vi/0Hr5YwUUhr0/0.jpg)](https://youtu.be/0Hr5YwUUhr0 "Statistical Invariance")

### 5. Convolutional neural networks

[![Convolutional neural networks](http://img.youtube.com/vi/ISHGyvsT0QY/0.jpg)](https://youtu.be/ISHGyvsT0QY "Convolutional neural networks")

### 6. Intuition

Let's develop better intuition for how Convolutional Neural Networks (CNN) work. We'll examine how humans classify 
images, and then see how CNNs use similar approaches.

Let’s say we wanted to classify the following image of a dog as a Golden Retriever.

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/58377b77_dog-1210559-1280/dog-1210559-1280.jpg)

As humans, how do we do this?

One thing we do is that we identify certain parts of the dog, such as the nose, the eyes, and the fur. We essentially 
break up the image into smaller pieces, recognize the smaller pieces, and then combine those pieces to get an idea of 
the overall dog.

In this case, we might break down the image into a combination of the following:

* A nose
* Two eyes
* Golden fur

These pieces can be seen below:

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/58377bdb_screen-shot-2016-11-24-at-12.49.08-pm/screen-shot-2016-11-24-at-12.49.08-pm.png)

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/58377bed_screen-shot-2016-11-24-at-12.49.43-pm/screen-shot-2016-11-24-at-12.49.43-pm.png)

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/58377bff_screen-shot-2016-11-24-at-12.50.54-pm/screen-shot-2016-11-24-at-12.50.54-pm.png)

###### Going One Step Further

But let’s take this one step further. How do we determine what exactly a nose is? A Golden Retriever nose can be seen as
an oval with two black holes inside it. Thus, one way of classifying a Retriever’s nose is to to break it up into 
smaller pieces and look for black holes (nostrils) and curves that define an oval as shown below.

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/58377c52_screen-shot-2016-11-24-at-12.51.47-pm/screen-shot-2016-11-24-at-12.51.47-pm.png)

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/58377c68_screen-shot-2016-11-24-at-12.51.51-pm/screen-shot-2016-11-24-at-12.51.51-pm.png)

Broadly speaking, this is what a CNN learns to do. It learns to recognize basic lines and curves, then shapes and blobs, 
and then increasingly complex objects within the image. Finally, the CNN classifies the image by combining the larger, 
more complex objects.

In our case, the levels in the hierarchy are:

* Simple shapes, like ovals and dark circles
* Complex objects (combinations of simple shapes), like eyes, nose, and fur
* The dog as a whole (a combination of complex objects)

With deep learning, we don't actually program the CNN to recognize these specific features. Rather, the CNN learns on 
its own to recognize such objects through forward propagation and backpropagation!

It's amazing how well a CNN can learn to classify images, even though we never program the CNN with information about 
specific features to look for.

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/583cb19d_heirarchy-diagram/heirarchy-diagram.jpg)

A CNN might have several layers, and each layer might capture a different level in the hierarchy of objects. The first 
layer is the lowest level in the hierarchy, where the CNN generally classifies small parts of the image into simple 
shapes like horizontal and vertical lines and simple blobs of colors. The subsequent layers tend to be higher levels in 
the hierarchy and generally classify more complex ideas like shapes (combinations of lines), and eventually full objects 
like dogs.

Once again, the CNN learns all of this on its own. We don't ever have to tell the CNN to go looking for lines or curves 
or noses or fur. The CNN just learns from the training set and discovers which characteristics of a Golden Retriever are 
worth looking for.

That's a good start! Hopefully you've developed some intuition about how CNNs work.

Next, let’s look at some implementation details.

### 7. Filters

###### Breaking up an Image

The first step for a CNN is to break up the image into smaller pieces. We do this by selecting a width and height that 
defines a filter.

The filter looks at small pieces, or patches, of the image. These patches are the same size as the filter.

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/58377d67_vlcsnap-2016-11-24-15h52m47s438/vlcsnap-2016-11-24-15h52m47s438.png)

We then simply slide this filter horizontally or vertically to focus on a different piece of the image.

The amount by which the filter slides is referred to as the 'stride'. The stride is a hyperparameter which you, the 
engineer, can tune. Increasing the stride reduces the size of your model by reducing the number of total patches each 
layer observes. However, this usually comes with a reduction in accuracy.

Let’s look at an example. In this zoomed in image of the dog, we first start with the patch outlined in red. The width 
and height of our filter define the size of this square.

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/5840fdac_retriever-patch/retriever-patch.png)

We then move the square over to the right by a given stride (2 in this case) to get another patch.

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/5840fe04_retriever-patch-shifted/retriever-patch-shifted.png)

What's important here is that **we are grouping together adjacent pixels** and treating them as a collective.

In a normal, non-convolutional neural network, we would have ignored this adjacency. In a normal network, we would have 
connected every pixel in the input image to a neuron in the next layer. In doing so, we would not have taken advantage 
of the fact that pixels in an image are close together for a reason and have special meaning.

By taking advantage of this local structure, our CNN learns to classify local patterns, like shapes and objects, in an 
image.

###### Filter Depth

It's common to have more than one filter. Different filters pick up different qualities of a patch. For example, one 
filter might look for a particular color, while another might look for a kind of object of a specific shape. The amount 
of filters in a convolutional layer is called the filter depth.

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/58377e4f_neilsen-pic/neilsen-pic.png)

How many neurons does each patch connect to?

That’s dependent on our filter depth. If we have a depth of ```k```, we connect each patch of pixels to ```k``` neurons 
in the next layer. This gives us the height of ```k``` in the next layer, as shown below. In practice, ```k``` is a 
hyperparameter we tune, and most CNNs tend to pick the same starting values.

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/5840ffda_filter-depth/filter-depth.png)

But why connect a single patch to multiple neurons in the next layer? Isn’t one neuron good enough?

Multiple neurons can be useful because a patch can have multiple interesting characteristics that we want to capture.

For example, one patch might include some white teeth, some blonde whiskers, and part of a red tongue. In that case, we 
might want a filter depth of at least three - one for each of teeth, whiskers, and tongue.

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584104c8_teeth-whiskers-tongue/teeth-whiskers-tongue.png)

Having multiple neurons for a given patch ensures that our CNN can learn to capture whatever characteristics the CNN 
learns are important.

Remember that the CNN isn't "programmed" to look for certain characteristics. Rather, it learns **on its own which** 
characteristics to notice.

### 8. Features map sizes

[![Features map sizes](http://img.youtube.com/vi/lp1NrLZnCUM/0.jpg)](https://youtu.be/lp1NrLZnCUM "Features map sizes")

![alt text](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/10_02.PNG)

_n_ is pixel size, _p_ is padding, _f_ filter size and _s_ stride.

Same padding (_p_ = (_f_ - 1) / 2) and Valid padding (_p_ = 0):

new_n = &#8970; ((_n_ + 2_p_ - _f_ ) / (_s_)) + 1 &#8971;

[![Features map sizes answer](http://img.youtube.com/vi/W4xtf8LTz1c/0.jpg)](https://youtu.be/W4xtf8LTz1c "Features map sizes answer")

### 9. Convolutions continued

[![Convolutions continued](http://img.youtube.com/vi/utOv-BKI_vo/0.jpg)](https://youtu.be/utOv-BKI_vo "Convolutions continued")

### 10. Parameters

###### Parameter Sharing

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/58377f77_vlcsnap-2016-11-24-16h01m35s262/vlcsnap-2016-11-24-16h01m35s262.png)

When we are trying to classify a picture of a cat, we don’t care where in the image a cat is. If it’s in the top left or the bottom right, it’s still a cat in our eyes. We would like our CNNs to also possess this ability known as translation invariance. How can we achieve this?

As we saw earlier, the classification of a given patch in an image is determined by the weights and biases corresponding to that patch.

If we want a cat that’s in the top left patch to be classified in the same way as a cat in the bottom right patch, we 
need the weights and biases corresponding to those patches to be the same, so that they are classified the same way.

This is exactly what we do in CNNs. The weights and biases we learn for a given output layer are shared across all 
patches in a given input layer. Note that as we increase the depth of our filter, the number of weights and biases we 
have to learn still increases, as the weights aren't shared across the output channels.

There’s an additional benefit to sharing our parameters. If we did not reuse the same weights across all patches, we 
would have to learn new parameters for every single patch and hidden layer neuron pair. This does not scale well, 
especially for higher fidelity images. Thus, sharing parameters not only helps us with translation invariance, but also 
gives us a smaller, more scalable model.

###### Padding

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5837d4d5_screen-shot-2016-11-24-at-10.05.37-pm/screen-shot-2016-11-24-at-10.05.37-pm.png)

![alt text](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/10_03.PNG)

As we can see, the width and height of each subsequent layer decreases in the above scheme.

In an ideal world, we'd be able to maintain the same width and height across layers so that we can continue to add 
layers without worrying about the dimensionality shrinking and so that we have consistency. How might we achieve this? 
One way is to simply add a border of ```0```s to our original ```5x5``` image. You can see what this looks like in the 
below image.

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5837d4ee_screen-shot-2016-11-24-at-10.05.46-pm/screen-shot-2016-11-24-at-10.05.46-pm.png)

This would expand our original image to a ```7x7```. With this, we now see how our next layer's size is again a 
```5x5```, keeping our dimensionality consistent.

###### Dimensionality

From what we've learned so far, how can we calculate the number of neurons of each layer in our CNN?

Given:

* our input layer has a width of ```W``` and a height of ```H```
* our convolutional layer has a filter size ```F```
* we have a stride of ```S```
* a padding of ```P```
* and the number of filters ```K```,

the following formula gives us the width of the next layer: ```W_out =[ (W−F+2P)/S] + 1```.

The output height would be ```H_out = [(H-F+2P)/S] + 1```.

And the output depth would be equal to the number of filters ```D_out = K```.

The output volume would be ```W_out * H_out * D_out```.

Knowing the dimensionality of each additional layer helps us understand how large our model is and how our decisions 
around filter size and stride affect the size of our network.

### 11. Quiz: Convolution output shape

###### Introduction

For the next few quizzes we'll test your understanding of the dimensions in CNNs. Understanding dimensions will help you 
make accurate tradeoffs between model size and performance. As you'll see, some parameters have a much bigger impact on 
model size than others.

###### Setup

H = height, W = width, D = depth

* We have an input of shape 32x32x3 (HxWxD)
* 20 filters of shape 8x8x3 (HxWxD)
* A stride of 2 for both the height and width (S)
* With padding of size 1 (P)

Recall the formula for calculating the new height or width:

```
new_height = (input_height - filter_height + 2 * P)/S + 1
new_width = (input_width - filter_width + 2 * P)/S + 1
```

![alt text](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/10_04.PNG)

### 12. Solution: Convolution output shape

###### Solution

The answer is **14x14x20**.

We can get the new height and width with the formula resulting in:

```
(32 - 8 + 2 * 1)/2 + 1 = 14
(32 - 8 + 2 * 1)/2 + 1 = 14
```

The new depth is equal to the number of filters, which is 20.

This would correspond to the following code:

```
input = tf.placeholder(tf.float32, (None, 32, 32, 3))
filter_weights = tf.Variable(tf.truncated_normal((8, 8, 3, 20))) # (height, width, input_depth, output_depth)
filter_bias = tf.Variable(tf.zeros(20))
strides = [1, 2, 2, 1] # (batch, height, width, depth)
padding = 'SAME'
conv = tf.nn.conv2d(input, filter_weights, strides, padding) + filter_bias
```

Note the output shape of ```conv``` will be \[1, 16, 16, 20]. It's 4D to account for batch size, but more importantly, 
it's not \[1, 14, 14, 20]. This is because the padding algorithm TensorFlow uses is not exactly the same as the one above
. An alternative algorithm is to switch ```padding``` from ```'SAME'``` to ```'VALID'``` which would result in an output shape of 
\[1, 13, 13, 20]. If you're curious how padding works in TensorFlow, read [this document](https://www.tensorflow.org/api_guides/python/nn#Convolution).

In summary TensorFlow uses the following equation for 'SAME' vs 'VALID'

**SAME Padding**, the output height and width are computed as:

```out_height``` = ceil(float(in_height) / float(strides \[1]))

```out_width``` = ceil(float(in_width) / float(strides \[2]))

**VALID Padding**, the output height and width are computed as:

```out_height``` = ceil(float(in_height - filter_height + 1) / float(strides \[1]))

```out_width``` = ceil(float(in_width - filter_width + 1) / float(strides \[2]))

### 13. Quiz: Number of parameters

We're now going to calculate the number of parameters of the convolutional layer. The answer from the last quiz will 
come into play here!

Being able to calculate the number of parameters in a neural network is useful since we want to have control over how 
much memory a neural network uses.

###### Setup

H = height, W = width, D = depth

* We have an input of shape 32x32x3 (HxWxD)
* 20 filters of shape 8x8x3 (HxWxD)
* A stride of 2 for both the height and width (S)
* Zero padding of size 1 (P)

###### Output Layer

* 14x14x20 (HxWxD)

###### Hint

Without parameter sharing, each neuron in the output layer must connect to each neuron in the filter. In addition, each 
neuron in the output layer must also connect to a single bias neuron.

![alt text](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/10_05.PNG)

### 14. Solution: Number of parameters

###### Solution

There are ```756560``` total parameters. That's a HUGE amount! Here's how we calculate it:

```(8 * 8 * 3 + 1) * (14 * 14 * 20) = 756560```

```8 * 8 * 3``` is the number of weights, we add ```1``` for the bias. Remember, each weight is assigned to every single 
part of the output (```14 * 14 * 20```). So we multiply these two numbers together and we get the final answer.

### 15. Quiz: Parameter sharing

Now we'd like you to calculate the number of parameters in the convolutional layer, if every neuron in the output layer 
shares its parameters with every other neuron in its same channel.

This is the number of parameters actually used in a convolution layer (```tf.nn.conv2d()```).

###### Setup

H = height, W = width, D = depth

* We have an input of shape 32x32x3 (HxWxD)
* 20 filters of shape 8x8x3 (HxWxD)
* A stride of 2 for both the height and width (S)
* Zero padding of size 1 (P)

###### Output Layer

* 14x14x20 (HxWxD)

###### Hint

With parameter sharing, each neuron in an output channel shares its weights with every other neuron in that channel. So 
the number of parameters is equal to the number of neurons in the filter, plus a bias neuron, all multiplied by the 
number of channels in the output layer.

![alt text](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/10_06.PNG)

### 16. Solution: Parameter sharing

###### Solution

There are ```3860``` total parameters. That's 196 times fewer parameters! Here's how the answer is calculated:

```(8 * 8 * 3 + 1) * 20 = 3840 + 20 = 3860```

That's ```3840``` weights and ```20``` biases. This should look similar to the answer from the previous quiz. The 
difference being it's just ```20``` instead of (```14 * 14 * 20```). Remember, with weight sharing we use the same 
filter for an entire depth slice. Because of this we can get rid of ```14 * 14``` and be left with only ```20```.

### 17. Visualizing CNNs

Let’s look at an example CNN to see how it works in action.

The CNN we will look at is trained on ImageNet as described in [this paper](http://www.matthewzeiler.com/pubs/arxive2013/eccv2014.pdf) 
by Zeiler and Fergus. In the images below (from the same paper), we’ll see what each layer in this network detects and 
see how each layer detects more and more complex ideas.

###### Layer 1

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/583cbd42_layer-1-grid/layer-1-grid.png)

The images above are from Matthew Zeiler and Rob Fergus' [deep visualization toolbox](https://www.youtube.com/watch?v=ghEmQSxT6tw), 
which lets us visualize what each layer in a CNN focuses on.

Each image in the above grid represents a pattern that causes the neurons in the first layer to activate - in other words, 
they are patterns that the first layer recognizes. The top left image shows a -45 degree line, while the middle top 
square shows a +45 degree line. These squares are shown below again for reference.

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/583cbba2_diagonal-line-1/diagonal-line-1.png)

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/583cbc02_diagonal-line-2/diagonal-line-2.png)

Let's now see some example images that cause such activations. The below grid of images all activated the -45 degree 
line. Notice how they are all selected despite the fact that they have different colors, gradients, and patterns.

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/583cbace_grid-layer-1/grid-layer-1.png)

So, the first layer of our CNN clearly picks out very simple shapes and patterns like lines and blobs.

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/583780f3_screen-shot-2016-11-24-at-12.09.02-pm/screen-shot-2016-11-24-at-12.09.02-pm.png)

The second layer of the CNN captures complex ideas.

As you see in the image above, the second layer of the CNN recognizes circles (second row, second column), stripes 
(first row, second column), and rectangles (bottom right).

**The CNN learns to do this on its own**. There is no special instruction for the CNN to focus on more complex objects 
in deeper layers. That's just how it normally works out when you feed training data into a CNN.

###### Layer 3

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5837811f_screen-shot-2016-11-24-at-12.09.24-pm/screen-shot-2016-11-24-at-12.09.24-pm.png)

The third layer picks out complex combinations of features from the second layer. These include things like grids, and 
honeycombs (top left), wheels (second row, second column), and even faces (third row, third column).

We'll skip layer 4, which continues this progression, and jump right to the fifth and final layer of this CNN.

###### Layer 5

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/58378151_screen-shot-2016-11-24-at-12.08.11-pm/screen-shot-2016-11-24-at-12.08.11-pm.png)

The last layer picks out the highest order ideas that we care about for classification, like dog faces, bird faces, and 
bicycles.

###### On to TensorFlow

This concludes our high-level discussion of Convolutional Neural Networks.

Next you'll practice actually building these networks in TensorFlow.

### 18. TF convolution layer

Let's examine how to implement a CNN in TensorFlow.

TensorFlow provides the ```tf.nn.conv2d()``` and ```tf.nn.bias_add()``` functions to create your own convolutional layers.

```
# Output depth
k_output = 64

# Image Properties
image_width = 10
image_height = 10
color_channels = 3

# Convolution filter
filter_size_width = 5
filter_size_height = 5

# Input/Image
input = tf.placeholder(
    tf.float32,
    shape=[None, image_height, image_width, color_channels])

# Weight and bias
weight = tf.Variable(tf.truncated_normal(
    [filter_size_height, filter_size_width, color_channels, k_output]))
bias = tf.Variable(tf.zeros(k_output))

# Apply Convolution
conv_layer = tf.nn.conv2d(input, weight, strides=[1, 2, 2, 1], padding='SAME')
# Add bias
conv_layer = tf.nn.bias_add(conv_layer, bias)
# Apply activation function
conv_layer = tf.nn.relu(conv_layer)
```

The code above uses the ```tf.nn.conv2d()``` function to compute the convolution with ```weight``` as the filter and 
```[1, 2, 2, 1]``` for the strides. TensorFlow uses a stride for each ```input``` dimension, 
```[batch, input_height, input_width, input_channels]```. We are generally always going to set the stride for ```batch```
 and ```input_channels``` (i.e. the first and fourth element in the ```strides``` array) to be ```1```.

You'll focus on changing ```input_height``` and ```input_width``` while setting ```batch``` and input_channels to 1. The 
```input_height``` and ```input_width``` strides are for striding the filter over ```input```. This example code uses a 
stride of 2 with 5x5 filter over ```input```.

The ```tf.nn.bias_add()``` function adds a 1-d bias to the last dimension in a matrix.

### 19. Explore the design space

[![Explore the design space](http://img.youtube.com/vi/FG7M9tWH2nQ/0.jpg)](https://youtu.be/FG7M9tWH2nQ "Explore the design space")

### 20. TF max pooling

The image above is an example of [max pooling](https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer) 
with a 2x2 filter and stride of 2. The four 2x2 colors represent each time the filter was applied to find the maximum value.

For example, ```[[1, 0], [4, 6]]``` becomes ```6```, because ```6``` is the maximum value in this set. Similarly, 
```[[2, 3], [6, 8]]``` becomes ```8```.

Conceptually, the benefit of the max pooling operation is to reduce the size of the input, and allow the neural network to focus on only the most important elements. Max pooling does this by only retaining the maximum value for each filtered area, and removing the remaining values.

TensorFlow provides the ```tf.nn.max_pool()``` function to apply [max pooling](https://en.wikipedia.org/wiki/Convolutional_neural_network#Pooling_layer) 
to your convolutional layers.

```
...
conv_layer = tf.nn.conv2d(input, weight, strides=[1, 2, 2, 1], padding='SAME')
conv_layer = tf.nn.bias_add(conv_layer, bias)
conv_layer = tf.nn.relu(conv_layer)
# Apply Max Pooling
conv_layer = tf.nn.max_pool(
    conv_layer,
    ksize=[1, 2, 2, 1],
    strides=[1, 2, 2, 1],
    padding='SAME')
```

The ```tf.nn.max_pool()``` function performs max pooling with the ```ksize``` parameter as the size of the filter and 
the ```strides``` parameter as the length of the stride. 2x2 filters with a stride of 2x2 are common in practice.

The ```ksize``` and ```strides``` parameters are structured as 4-element lists, with each element corresponding to a 
dimension of the input tensor (```[batch, height, width, channels]```). For both ```ksize``` and ```strides```, the 
batch and channel dimensions are typically set to ```1```.

### 21. Quiz: Pooling intuition

The next few quizzes will test your understanding of pooling layers.

![alt text](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/10_07.PNG)

### 22. Solution: Pooling intuition

###### Solution
The correct answer is **decrease the size of the output** and prevent **overfitting**. Preventing overfitting is a 
consequence of reducing the output size, which in turn, reduces the number of parameters in future layers.

Recently, pooling layers have fallen out of favor. Some reasons are:

* Recent datasets are so big and complex we're more concerned about underfitting.
* Dropout is a much better regularizer.
* Pooling results in a loss of information. Think about the max pooling operation as an example. We only keep the 
largest of n numbers, thereby disregarding n-1 numbers completely.

### 23. Quiz: Polling mechanics

###### Setup

H = height, W = width, D = depth

* We have an input of shape 4x4x5 (HxWxD)
* Filter of shape 2x2 (HxW)
* A stride of 2 for both the height and width (S)

Recall the formula for calculating the new height or width:

```
new_height = (input_height - filter_height)/S + 1
new_width = (input_width - filter_width)/S + 1
```
NOTE: For a pooling layer the output depth is the same as the input depth. Additionally, the pooling operation is 
applied individually for each depth slice.


The image below gives an example of how a max pooling layer works. In this case, the max pooling filter has a shape of 
2x2. As the max pooling filter slides across the input layer, the filter will output the maximum value of the 2x2 square.

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58a5fe3e_convolutionalnetworksquiz/convolutionalnetworksquiz.png)

![alt text](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/10_08.PNG)

### 24. Solution: Polling mechanics

###### Solution

The answer is 2x2x5. Here's how it's calculated using the formula:

```
(4 - 2)/2 + 1 = 2
(4 - 2)/2 + 1 = 2
```

The depth stays the same.

Here's the corresponding code:

```
input = tf.placeholder(tf.float32, (None, 4, 4, 5))
filter_shape = [1, 2, 2, 1]
strides = [1, 2, 2, 1]
padding = 'VALID'
pool = tf.nn.max_pool(input, filter_shape, strides, padding)
```

The output shape of ```pool``` will be \[1, 2, 2, 5], even if ```padding``` is changed to ```'SAME'```.

### 25. Quiz: Pooling practice

Great, now let's practice doing some pooling operations manually.

![alt text](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/10_09.PNG)

### 26. Solution: Pooling practice

###### Solution

The correct answer is ```2.5,10,15,6```. We start with the four numbers in the top left corner. Then we work 
left-to-right and top-to-bottom, moving 2 units each time.

```
max(0, 1, 2, 2.5) = 2.5
max(0.5, 10, 1, -8) = 10
max(4, 0, 15, 1) = 15
max(5, 6, 2, 3) = 6
```

### 27. Quiz: Average pooling

![alt text](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/10_10.PNG)

### 28. Solution: Average pooling

###### Solution

The correct answer is ```1.375,0.875,5,4```. We start with the four numbers in the top left corner. Then we work 
left-to-right and top-to-bottom, moving 2 units each time.

```
mean(0, 1, 2, 2.5) = 1.375
mean(0.5, 10, 1, -8) = 0.875
mean(4, 0, 15, 1) = 5
mean(5, 6, 2, 3) = 4
```

### 29. 1 * 1 convolutions

[![1 * 1 convolutions](http://img.youtube.com/vi/Zmzgerm6SjA/0.jpg)](https://youtu.be/Zmzgerm6SjA "1 * 1 convolutions")

### 30. Inception module

[![Inception](http://img.youtube.com/vi/SlTm03bEOxA/0.jpg)](https://youtu.be/SlTm03bEOxA "Inception")

### 31. 

[![](http://img.youtube.com/vi//0.jpg)]( "")

### 32. 

[![](http://img.youtube.com/vi//0.jpg)]( "")

### 33. 

[![](http://img.youtube.com/vi//0.jpg)]( "")

### 34. 

[![](http://img.youtube.com/vi//0.jpg)]( "")

### 35. 

[![](http://img.youtube.com/vi//0.jpg)]( "")

### 36. 

[![](http://img.youtube.com/vi//0.jpg)]( "")

### 37. 

[![](http://img.youtube.com/vi//0.jpg)]( "")

### 38. CNNs - Additional Resources

###### Additional Resources
There are many wonderful free resources that allow you to go into more depth around Convolutional Neural Networks. In 
this course, our goal is to give you just enough intuition to start applying this concept on real world problems so you 
have enough of an exposure to explore more on your own. We strongly encourage you to explore some of these resources 
more to reinforce your intuition and explore different ideas.

These are the resources we recommend in particular:

* Andrej Karpathy's [CS231n Stanford course](http://cs231n.github.io/) on Convolutional Neural Networks.
* Michael Nielsen's [free book](http://neuralnetworksanddeeplearning.com/) on Deep Learning.
* Goodfellow, Bengio, and Courville's more advanced [free book](http://deeplearningbook.org/) on Deep Learning.