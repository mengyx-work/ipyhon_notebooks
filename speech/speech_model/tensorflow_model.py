import os, time
import tensorflow as tf


class TensorFlowModel(object):
    def __init__(self, mode, log_path, model_path, model_settings,
                 model_building_fn=None, saving_steps=20, dipslay_steps=10):
        self.log_path = log_path
        self.model_path = model_path
        self.saving_steps = saving_steps
        self.display_steps = dipslay_steps
        self.model_settings = model_settings
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=self.model_settings['sess_config'])
        self.global_step = 0

        MODEL_MODES = ['train', 'restore_model', 'eval']
        if mode not in MODEL_MODES:
            raise ValueError('failed to recognize given mode {}'.format(mode))
        if mode == 'train':
            if model_building_fn is None:
                raise ValueError('the model building function is missing')
            self._build_training_graph(model_building_fn)

    def _build_training_graph(self, build_model):
        with self.graph.as_default():
            self._init_placeholders()
            logits = build_model(self.wav_inputs, self.model_settings)
            self.loss = self._build_loss(logits)
            self._build_optimizer()
            self.saver = tf.train.Saver(max_to_keep=2, keep_checkpoint_every_n_hours=1)
            init = tf.global_variables_initializer()
            self.sess.run(init)

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
        with tf.name_scope('loss'):
            cross_entropy_mean = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.ground_truth_labels,
                                                               logits=logits))
        tf.summary.scalar('cross_entropy', cross_entropy_mean)
        return cross_entropy_mean

    def _build_optimizer(self):
        with tf.name_scope('train'):
            self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate_input).minimize(self.loss)

    def _next_feed(self, batches, learning_rate):
        data_batch, target_batch = next(batches)
        return {self.wav_inputs: data_batch,
                self.ground_truth_labels: target_batch,
                self.dropout_prob: 0.5,
                self.learning_rate_input: learning_rate
                }

    def _saving_step_run(self):
        _ = self.sess.run([self.increment_saving_step_op])
        self.saver.save(self.sess, os.path.join(self.model_path, 'models'), global_step=self.global_step)

    def _display_step_run(self, start_time, feed_content):
        summary, loss_value = self.sess.run([self.merged_summary_op, self.loss], feed_content)
        self.writer.add_summary(summary, self.global_step)
        print 'step {}, batch loss: {} and using {:.2f} seconds'.format(self.global_step, loss_value, time.time()-start_time)

    def _reach_display_step(self):
        return self.global_step == 1 or self.global_step % self.display_steps == 0

    def _reach_saving_step(self):
        return self.global_step % self.saving_steps == 0

    def train(self, batch_generator, num_batches):
        with self.graph.as_default():
            self.merged_summary_op = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(self.log_path, graph=self.graph)

            cur_time = time.time()
            while self.global_step < num_batches:
                feed_content = self._next_feed(batch_generator, 0.0001)
                _ = self.sess.run([self.train_step], feed_content)
                self.global_step += 1

                if self._reach_saving_step():
                    self._saving_step_run()

                if self._reach_display_step():
                    self._display_step_run(cur_time, feed_content)
                    cur_time = time.time()
            self.saver.save(self.sess, os.path.join(self.model_path, 'complete_model'), global_step=self.global_step)