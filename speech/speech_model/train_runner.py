import os
import tensorflow as tf
from tensorflow_model import TensorFlowModel
from data_generator import DataGenerator

def build_NenuralNetwork_model(wav_input, model_settings):
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


COMMON_PATH = os.path.join(os.path.expanduser("~"), 'local_tensorflow_content')

def train_model(model_name='speech_neural_network_model'):
    sess_config = tf.ConfigProto(log_device_placement=False)
    model_path = os.path.join(COMMON_PATH, model_name)
    log_path = os.path.join(COMMON_PATH, model_name, 'log')

    data_path = '/Users/matt.meng/data/speech_competition/processed_data'
    data_generator = DataGenerator(data_path)
    batch_size = 256
    batches = data_generator.generate_batch_iter(batch_size)

    model_settings = {}


if __name__ == '__main__':
    train_model()