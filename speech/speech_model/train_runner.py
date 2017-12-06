import os
import tensorflow as tf
from tensorflow_model import TensorFlowModel
from data_generator import DataGenerator


def build_nenural_network_model(wav_input, model_settings):
    frame_size = model_settings['frame_size']
    frame_num = model_settings['frame_num']
    #batch_size, frame_num_, frame_size_ = tf.unstack(tf.shape(wav_input))
    label_size = model_settings['label_size']
    hidden_layer_output_size = model_settings['hidden_layer_size']

    hidden_layer_weight = tf.Variable(tf.truncated_normal([frame_size * frame_num, hidden_layer_output_size], stddev=0.001),
                                      name='hidden_layer_weight')
    hidden_layer_bias = tf.Variable(tf.zeros([hidden_layer_output_size]), name='hidden_layer_bias')
    flat_wav_input = tf.reshape(wav_input, (-1, frame_size * frame_num))
    hidden_layer_output = tf.add(tf.matmul(flat_wav_input, hidden_layer_weight), hidden_layer_bias)

    output_weights = tf.Variable(tf.truncated_normal([hidden_layer_output_size, label_size]), name='output_weights')
    output_bias = tf.Variable(tf.zeros([label_size]), name='output_bias')
    logits = tf.add(tf.matmul(hidden_layer_output, output_weights), output_bias)
    return logits


COMMON_PATH = os.path.join(os.path.expanduser("~"), 'local_tensorflow_content')
NUM_THREADS = 6

def train_model(model_name='speech_neural_network_model', USE_GPU=False):
    model_path = os.path.join(COMMON_PATH, model_name)
    log_path = os.path.join(COMMON_PATH, model_name, 'log')

    data_path = '/Users/matt.meng/data/speech_competition/processed_data'
    data_generator = DataGenerator(data_path)
    batch_size = 256
    batches = data_generator.generate_batch_iter(batch_size)
    model_settings = {'frame_num': data_generator.frame_num,
                      'frame_size': data_generator.frame_size,
                      'label_size': 12,
                      'hidden_layer_size': 256}
    if USE_GPU:
        model_settings['sess_config'] = tf.ConfigProto(log_device_placement=False)
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # the only way to completely not use GPU
        model_settings['sess_config'] = tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS)

    model = TensorFlowModel('train', log_path, model_path, model_settings, model_building_fn=build_nenural_network_model)
    model.train(batches, 100000)

if __name__ == '__main__':
    train_model()