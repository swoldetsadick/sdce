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

[![](http://img.youtube.com/vi//0.jpg)]( "")

### 1. 

[![](http://img.youtube.com/vi//0.jpg)]( "")

### 1. 

[![](http://img.youtube.com/vi//0.jpg)]( "")

### 1. 

[![](http://img.youtube.com/vi//0.jpg)]( "")

### 1. 

[![](http://img.youtube.com/vi//0.jpg)]( "")

### 1. 

[![](http://img.youtube.com/vi//0.jpg)]( "")

### 1. 

[![](http://img.youtube.com/vi//0.jpg)]( "")

### 1. 

[![](http://img.youtube.com/vi//0.jpg)]( "")

### 1. 

[![](http://img.youtube.com/vi//0.jpg)]( "")

### 1. 

[![](http://img.youtube.com/vi//0.jpg)]( "")

### 2. 

[![](http://img.youtube.com/vi//0.jpg)]( "")

### 2. 

[![](http://img.youtube.com/vi//0.jpg)]( "")