import tensorflow as tf
import numpy as np
#from __future__ import print_function



x_data = np.linspace(-1,1,300)[:,np.newaxis]
bias = np.random.normal(0,0.5,x_data.shape)
y_data = np.square(x_data) - 0.5 + bias

x_train = tf.placeholder(tf.float32, [None, 1])
y_train = tf.placeholder(tf.float32, [None, 1])


def add_Layer(input, input_size, output_size, relu):
    weights = tf.Variable(tf.random_normal([input_size, output_size]))
    bias = tf.Variable(tf.zeros([1, output_size]) + 0.1)
    output = tf.matmul(input, weights) + bias

    if(relu):
        output_actual = relu(output)
    else:
        output_actual = output

    return output_actual;


sess = tf.Session()

middel_out = add_Layer(x_train, 1, 10, tf.nn.relu)
print(middel_out)
final_out = add_Layer(middel_out, 10, 1, None)

cost = tf.reduce_mean(tf.reduce_sum(tf.square(y_train - final_out), reduction_indices=[1]))

trainer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
sess.run(tf.global_variables_initializer())

for i in range(1000):
    sess.run(trainer, feed_dict = {x_train:x_data, y_train:y_data})

    if i%50 == 0:
        print("-----cost-----")
        print(sess.run(cost, feed_dict = {x_train:x_data, y_train:y_data}))
        print("------cost-----")
