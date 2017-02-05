import tensorflow as tf

class tfECGNetwork:
    def __init__(self, N, num_keys):
        with tf.Graph().as_default():
            self.prepare_model(N, num_keys)
            self.prepare_session()

    def prepare_model(self, N, num_keys):
        # Input
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, N], name='input')

        # Output layer
        with tf.name_scope('output'):
            w = tf.Variable(tf.zeros([N, num_keys]), name='weights')
            w0 = tf.Variable(tf.zeros([num_keys]), name='biases')
            f = tf.matmul(x, w) + w0
            p = tf.nn.softmax(f, name='softmax')

        # Optimizer
        with tf.name_scope('optimizer'):
            t = tf.placeholder(tf.float32, [None, num_keys])
            #loss = -tf.reduce_sum(t*tf.log(p))
            loss = tf.reduce_sum(tf.square(p-t))
            #train_step = tf.train.AdamOptimizer().minimize(loss)
            train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

        # Evaluator
        with tf.name_scope('evaluator'):
            correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.scalar_summary("loss", loss)
        tf.scalar_summary("accuracy", accuracy)
        tf.histogram_summary("weights", w)
        tf.histogram_summary("biases", w0)

        self.x, self.t, self.p = x, t, p
        self.train_step = train_step
        self.loss = loss
        self.accuracy = accuracy

    def prepare_session(self):
        sess = tf.InteractiveSession()
        sess.run(tf.initialize_all_variables())
        summary = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter("/tmp/ecg/tb", sess.graph_def)

        self.sess = sess
        self.summary = summary
        self.writer = writer
