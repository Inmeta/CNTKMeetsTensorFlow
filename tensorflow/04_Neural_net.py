import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_h, w_o):
    with tf.name_scope("Layer_1"):
        h1 = tf.nn.sigmoid(tf.matmul(X, w_h)) # this is a basic mlp, think 2 stacked logistic regressions
    with tf.name_scope("Layer_2"):
        h2 = tf.matmul(h1, w_o)

    return h2 # note that we dont take the softmax at the end because our cost fn does that for us


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])

w_h = init_weights([784, 625]) # create symbolic variables
w_o = init_weights([625, 10])

py_x = model(X, w_h, w_o)

# Add histogram summaries for weights
tf.histogram_summary("w_h_summ", w_h)
tf.histogram_summary("w_o_summ", w_o)

with tf.name_scope("cost"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y)) # compute costs
    train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer
    predict_op = tf.argmax(py_x, 1)

with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(Y, 1), tf.argmax(py_x, 1))  # Count correct predictions
    acc_op = tf.reduce_mean(tf.cast(correct_pred, "float"))  # Cast boolean to float to average
    # Add scalar summary for accuracy
    tf.scalar_summary("accuracy", acc_op)

# Save / Restore
ckpt_dir = "./ckpt_dir"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

global_step = tf.Variable(0, name='global_step', trainable=False)

# Call this after declaring all tf.Variables.
saver = tf.train.Saver()


# Launch the graph in a session
with tf.Session() as sess:

    # create a log writer. run 'tensorboard --logdir=./logs/nn_logs'
    writer = tf.train.SummaryWriter("./logs", sess.graph) # for 0.8
    merged = tf.merge_all_summaries()


    # you need to initialize all variables
    tf.initialize_all_variables().run()

    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)  # restore all variables

    start = global_step.eval()  # get last global_step
    print("Start from:", start)

    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

        global_step.assign(i).eval() # set and update(eval) global_step with index, i
        saver.save(sess, ckpt_dir + "/model.ckpt", global_step=global_step)

        #print(i, np.mean(np.argmax(teY, axis=1) ==
        #                 sess.run(predict_op, feed_dict={X: teX, Y: teY})))

        summary, acc = sess.run([merged, acc_op], feed_dict={X: teX, Y: teY})
        writer.add_summary(summary, i)  # Write summary
        print(i, acc)  # Report the accuracy