import tensorflow as tf

n_C0 = 3
n_C11 = 64
f11 = 3	# filter_size
s11 = 2	# stride
p11 = 0	# padding
W11 = tf.compat.v1.get_variable(dtype = tf.float32,
    shape = [f11, f11, n_C0, n_C11],
    initializer = tf.compat.v1.glorot_uniform_initializer(),
    name = "W11")

# variable_summaries(W11)
print(W11)