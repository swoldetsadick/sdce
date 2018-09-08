
### 01. TensorFlow Input


```python
import tensorflow as tf
x = tf.placeholder(tf.string)

with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: 'Hello World'})
    print(output)
```

    Hello World



```python
x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)

with tf.Session() as sess:
    output = sess.run(x, feed_dict={x: 'Test String', y: 123, z: 45.67})
    print(output)
    output = sess.run(y, feed_dict={x: 'Test String', y: 123, z: 45.67})
    print(output)
    output = sess.run(z, feed_dict={x: 'Test String', y: 123, z: 45.67})
    print(output)
    output = sess.run(x, feed_dict={x: 123, y: 123, z: 45.67})
    print(output)
    # If the data passed to the feed_dict doesn’t match the tensor type and can’t be cast into the tensor type, 
    # you’ll get an error
```

    Test String
    123
    45.67



    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-39-5773c1b62768> in <module>()
         10     output = sess.run(z, feed_dict={x: 'Test String', y: 123, z: 45.67})
         11     print(output)
    ---> 12     output = sess.run(x, feed_dict={x: 123, y: 123, z: 45.67})
         13     print(output)
         14     # If the data passed to the feed_dict doesn’t match the tensor type and can’t be cast into the tensor type,


    /usr/local/lib/python2.7/dist-packages/tensorflow/python/client/session.pyc in run(self, fetches, feed_dict, options, run_metadata)
        875     try:
        876       result = self._run(None, fetches, feed_dict, options_ptr,
    --> 877                          run_metadata_ptr)
        878       if run_metadata:
        879         proto_data = tf_session.TF_GetBuffer(run_metadata_ptr)


    /usr/local/lib/python2.7/dist-packages/tensorflow/python/client/session.pyc in _run(self, handle, fetches, feed_dict, options, run_metadata)
       1058                     type(subfeed_val)) +
       1059                 ' is not compatible with Tensor type ' + str(subfeed_dtype) +
    -> 1060                 '. Try explicitly setting the type of the feed tensor'
       1061                 ' to a larger type (e.g. int64).')
       1062 


    TypeError: Type of feed value 123 with type <type 'int'> is not compatible with Tensor type <type 'object'>. Try explicitly setting the type of the feed tensor to a larger type (e.g. int64).


### 02. TensorFlow Math


```python
x_0 = tf.add(5, 2)  # 7

a = tf.placeholder(tf.int32)
b = tf.placeholder(tf.int32)
x_1 = tf.add(a, b)  # 7

y = tf.subtract(10, 4) # 6
z = tf.multiply(2, 5)  # 10

tf.placeholder(tf.int32)

with tf.Session() as sess:
    output_x_0 = sess.run(x_0)
    print(output_x_0)
    output_x_1 = sess.run(x_1, feed_dict={a: 5, b: 2})
    print(output_x_0 == output_x_1)
    output_y = sess.run(y)
    print(output_y)
    output_z = sess.run(z)
    print(output_z)
```

    7
    True
    6
    10



```python
A = tf.subtract(tf.constant(2.0), tf.constant(1))  
# Fails with ValueError: Tensor conversion requested dtype float32 for Tensor with dtype int32:
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-41-91f610169f42> in <module>()
    ----> 1 A = tf.subtract(tf.constant(2.0), tf.constant(1))
          2 # Fails with ValueError: Tensor conversion requested dtype float32 for Tensor with dtype int32:


    /usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/math_ops.pyc in subtract(x, y, name)
        221 @tf_export("subtract")
        222 def subtract(x, y, name=None):
    --> 223   return gen_math_ops.sub(x, y, name)
        224 
        225 


    /usr/local/lib/python2.7/dist-packages/tensorflow/python/ops/gen_math_ops.pyc in sub(x, y, name)
       8186   if _ctx is None or not _ctx._eager_context.is_eager:
       8187     _, _, _op = _op_def_lib._apply_op_helper(
    -> 8188         "Sub", x=x, y=y, name=name)
       8189     _result = _op.outputs[:]
       8190     _inputs_flat = _op.inputs


    /usr/local/lib/python2.7/dist-packages/tensorflow/python/framework/op_def_library.pyc in _apply_op_helper(self, op_type_name, name, **keywords)
        544                   "%s type %s of argument '%s'." %
        545                   (prefix, dtypes.as_dtype(attrs[input_arg.type_attr]).name,
    --> 546                    inferred_from[input_arg.type_attr]))
        547 
        548           types = [values.dtype]


    TypeError: Input 'y' of 'Sub' Op has type int32 that does not match type float32 of argument 'x'.



```python
B = tf.subtract(tf.cast(tf.constant(2.0), tf.int32), tf.constant(1))  # 1
with tf.Session() as sess:
    output_B = sess.run(B)
    print(output_B)
```

    1

