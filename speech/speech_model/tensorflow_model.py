import os, time
import tensorflow as tf
from utils import clear_folder, model_meta_file
from create_tensorboard_start_script import generate_tensorboard_script


class TensorFlowModel(object):
    def __init__(self, mode, log_path, model_path, model_settings,
                 model_building_fn=None, saving_steps=7500, dipslay_steps=5000):
        self.log_path = log_path
        self.model_path = model_path
        self.saving_steps = saving_steps
        self.display_steps = dipslay_steps
        self.model_settings = model_settings
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=self.model_settings['sess_config'])
        self.global_step = 0

        MODEL_MODES = ['train', 'restore_train', 'eval']
        if mode not in MODEL_MODES:
            raise ValueError('failed to recognize given mode {}'.format(mode))

        if mode == 'train':
            if model_building_fn is None:
                raise ValueError('the model building function is missing')
            self._build_training_graph(model_building_fn)

        if mode == 'restore_train':
            self._restore_model()

    def _build_training_graph(self, build_model):
        clear_folder(self.log_path)
        clear_folder(self.model_path)
        generate_tensorboard_script(self.log_path)  # create the script to start a Tensorboard

        with self.graph.as_default():
            self._init_placeholders()
            with tf.name_scope('model'):
                self.logits = build_model(self.wav_inputs, self.model_settings, self.dropout_prob, 'logit_output')
            self.loss = self._build_loss(self.logits)
            self._build_optimizer()
            self._build_predictor()
            self.saver = tf.train.Saver(max_to_keep=2, keep_checkpoint_every_n_hours=1)
            init = tf.global_variables_initializer()
            self.sess.run(init)

    def _build_predictor(self):
        with tf.name_scope('predictor'):
            predict_index = tf.argmax(self.logits, axis=1, output_type=tf.int32)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(predict_index, self.ground_truth_labels), tf.float32),
                                           name='accuracy')
        tf.summary.scalar('accuracy', self.accuracy)

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

    def _restore_model(self):
        with self.graph.as_default():
            self.saver = tf.train.import_meta_graph(model_meta_file(self.model_path))
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_path))
            self._restore_placeholders()
            self._restore_variable()

    def _restore_placeholders(self):
        self.wav_inputs = self.sess.graph.get_tensor_by_name("initial_inputs/wav_inputs:0")
        self.dropout_prob = self.sess.graph.get_tensor_by_name("initial_inputs/dropout_prob:0")
        self.learning_rate_input = self.sess.graph.get_tensor_by_name("initial_inputs/learning_rate_input:0")
        self.ground_truth_labels = self.sess.graph.get_tensor_by_name("initial_inputs/ground_truth_labels:0")
        self.global_saving_steps = self.sess.graph.get_tensor_by_name("initial_inputs/global_saving_steps:0")
        self.global_step = self.sess.run(self.global_saving_steps)

    def _restore_variable(self):
        self.logits = self.sess.graph.get_operation_by_name("model/logit_output")
        self.train_op = self.sess.graph.get_operation_by_name("optimizer/train_op")
        self.increment_saving_step_op = self.sess.graph.get_operation_by_name("initial_inputs/increment_saving_step_op")
        self.loss = self.sess.graph.get_tensor_by_name("loss/reduce_mean_loss:0")
        self.accuracy = self.sess.graph.get_operation_by_name("predictor/accuracy")

    def _build_loss(self, logits):
        with tf.name_scope('loss'):
            cross_entropy_mean = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.ground_truth_labels,
                                                               logits=logits), name='reduce_mean_loss')
        tf.summary.scalar('cross_entropy', cross_entropy_mean)
        return cross_entropy_mean

    def _build_optimizer(self):
        with tf.name_scope('optimizer'):
            #self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate_input).minimize(self.loss, name='train_op')
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate_input).minimize(self.loss, name='train_op')

    def _run_predictor(self, test_batches):
        feed_content = self._next_feed(test_batches, 'eval')
        accuracy = self.sess.run(self.accuracy, feed_content)
        return accuracy

    def _next_feed(self, batches, mode='train', learning_rate=None):
        if mode not in ['train', 'eval']:
            raise ValueError('failed to recognize the mode {} in `_next_feed`'.format(mode))
        data_batch, target_batch = next(batches)

        if mode == 'train':
            return {self.wav_inputs: data_batch,
                    self.ground_truth_labels: target_batch,
                    self.dropout_prob: 0.5,
                    self.learning_rate_input: learning_rate
                    }

        return {self.wav_inputs: data_batch,
                self.ground_truth_labels: target_batch,
                self.dropout_prob: 1.0,
                }

    def _saving_step_run(self):
        _ = self.sess.run([self.increment_saving_step_op])
        self.saver.save(self.sess, os.path.join(self.model_path, 'models'), global_step=self.global_step)

    def _display_step_run(self, start_time, feed_content, test_batches=None):
        _start_time = time.time()
        if test_batches is not None:
            accuracy_ = self._run_predictor(test_batches)
            print('using {:.2f} seconds, accuracy from test data is {}'.format((time.time() - _start_time), accuracy_))

        summary, loss_value = self.sess.run([self.merged_summary_op, self.loss], feed_content)
        self.writer.add_summary(summary, self.global_step)
        print('step {}, batch loss: {} and using {:.2f} seconds'.format(self.global_step,
                                                                        loss_value,
                                                                        time.time() - start_time))

    def _reach_display_step(self):
        return self.global_step == 1 or self.global_step % self.display_steps == 0

    def _reach_saving_step(self):
        return self.global_step % self.saving_steps == 0

    def train(self, batches, num_batches, test_batches=None):
        print('start training with total {} batches'.format(num_batches))
        with self.graph.as_default():
            self.merged_summary_op = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(self.log_path, graph=self.graph)
            cur_time = time.time()
            while self.global_step < num_batches:
                feed_content = self._next_feed(batches, 'train', 0.0001)
                _ = self.sess.run([self.train_op], feed_content)
                self.global_step += 1

                if self._reach_saving_step():
                    self._saving_step_run()

                if self._reach_display_step():
                    self._display_step_run(cur_time, feed_content, test_batches)
                    cur_time = time.time()
            self.saver.save(self.sess, os.path.join(self.model_path, 'complete_model'), global_step=self.global_step)