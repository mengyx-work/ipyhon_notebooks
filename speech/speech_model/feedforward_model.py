import os, time
import tensorflow as tf

def _build_NN_model(wav_input, model_settings):
    feature_size = model_settings['feature_size']
    label_size = model_settings['label_size']
    hidden_layer_output_size = model_settings['hidden_layer_output_size']
    hidden_layer_weight = tf.Variable(tf.truncated_normal([feature_size, hidden_layer_output_size], stddev=0.001))
    hidden_layer_bias = tf.Variable(tf.zeros([hidden_layer_output_size]))
    hidden_layer_output = tf.add(tf.matmul(wav_input, hidden_layer_weight), hidden_layer_bias)
    output_weights = tf.Variable(tf.truncated_normal([hidden_layer_output_size, label_size]))
    output_bias = tf.Variable(tf.zeros([label_size]))
    logits = tf.add(tf.matmul(output_weights, hidden_layer_output), output_bias)
    return logits


class FeedforwardModel(object):

    def __init__(self, sess_config, mode, saving_steps, log_path, model_settings):
        self.saving_steps = saving_steps
        self.log_path = log_path
        self.model_settings = model_settings
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=sess_config)
        self.global_step = 0

        MODEL_MODES = ['train', 'restore_model', 'eval']
        if mode not in MODEL_MODES:
            raise ValueError('failed to recognize given mode {}'.format(mode))
        if mode == 'train':
            self._build_training_graph()

    def _build_training_graph(self):
        self._init_placeholders()
        logits = self._build_NN_model(self.wav_inputs, self.model_settings)
        loss = self._build_loss(logits)
        self._build_optimizer(loss)

    def _init_placeholders(self):
        with tf.name_scope('initial_inputs'):
            self.dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
            self.learning_rate_input = tf.placeholder(tf.float32, name='learning_rate_input')
            self.wav_inputs = tf.placeholder(shape=(None, self.model_settings['frame_num'], self.model_settings['frame_size']),
                                             dtype=tf.float32,
                                             name='wav_inputs')
            self.ground_truth_labels = tf.placeholder(shape=(None,), dtype=tf.int32, name='ground_truth_labels')
            self.global_saving_steps = tf.Variable(0, name='global_saving_steps', trainable=False, dtype=tf.int32)
            self.increment_saving_step_op = tf.assign(self.global_saving_steps,
                                                      self.global_saving_steps + self.saving_steps,
                                                      name="increment_saving_step_op")

    def _build_loss(self, logits):
        with tf.name_scope('cross_entropy'):
            cross_entropy_mean = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.ground_truth_labels,
                                                               logits=logits))
        tf.summary.scalar('cross_entropy', cross_entropy_mean)
        return cross_entropy_mean

    def _build_optimizer(self, loss):
        with tf.name_scope('train'):
            self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate_input).minimize(loss)

    def _saving_step_run(self):
        summary, _ = self.sess.run(self.merged_summary_op, self.increment_saving_step_op)
        self.writer.add_summary(summary, self.global_step)
        self.saver.save(self.sess, os.path.join(self.model_path, 'models'), global_step=self.global_step)

    def train(self, num_batches, batch_generator):
        with self.graph.as_default():
            self.writer = tf.summary.FileWriter(self.log_path, graph=self.graph)
            self.merged_summary_op = tf.summary.merge_all()
            start_time = time.time()
            while self.global_step < num_batches:
                feed_content = self._next_feed(batch_generator)
                _ = self.sess.run([self.train_op], feed_content)
                self.global_step += 1

                if self.global_step % self.saving_steps == 0:
                    self._saving_step_run()

                if self.global_step == 1 or self.global_step % self.display_steps == 0:
                    start_time = self._display_step_run(start_time, feed_content)
            self.saver.save(self.sess, os.path.join(self.model_path, 'complete_model'), global_step=self.global_step)
