import tensorflow as tf
a = tf.compat.v1.placeholder(tf.float32)
b = tf.compat.v1.placeholder(tf.float32)
add = tf.add(a, b)
sess = tf.Session()
binding = {a: 1, b: 2}
c = sess.run(add, feed_dict=binding)
print(c)