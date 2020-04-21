import tensorflow as tf

# Define the computational graph
x = tf.placeholder(dtype=tf.int32, shape=[1], name='x')
a = tf.constant([5, 6, 7, 8], name='a')
b = tf.constant([1, 1, 1, 1], name='b')

addition = tf.add(a, b, name='addition')
print(addition)

mul = tf.multiply(x, addition, name='mul')

# Initialize the Session
with tf.Session() as sess:
    writer = tf.summary.FileWriter('example_logs', sess.graph)

    # Run the graph
    result = sess.run(mul, feed_dict={x: [2]})
    print(result)
