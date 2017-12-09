import time
import os, glob, random
import numpy as np
import pandas as pd
from scipy.io import wavfile
from chunk_data import ChunkData
from noise_generator import NoiseGenerator
from python_speech_features import logfbank
from speech.speech_model.data_preprocess import categorize_wav_files_by_label, generate_label_dict, \
    generate_proportional_data


def emphasize_signal(samples, pre_emphasis=0.97):
    ## y[t] = x[t] - pre_emphasis * x[t-1]
    return np.append(samples[0], samples[1:] - pre_emphasis * samples[:-1])


def max_sample_index(samples, std_window_length):
    std_array = pd.rolling_std(samples, std_window_length)
    std_array = std_array[std_window_length:]
    return np.argmax(std_array)


def align_wave_with_noise_padding(file_name_, noise_generator, wav_window_length=3000, std_window_length=500):
    sample_rate, samples = wavfile.read(file_name_)
    samples = emphasize_signal(samples)
    center_index = max_sample_index(samples, std_window_length)

    if center_index > wav_window_length:
        left_index = center_index - wav_window_length
        left_half = samples[left_index:center_index]
    else:
        left_padding = noise_generator.random_noise(wav_window_length - center_index)
        left_half = np.append(left_padding, samples[:center_index])
    if center_index + wav_window_length < len(samples):
        return np.append(left_half, samples[center_index:(center_index + wav_window_length)])
    else:
        right_padding = noise_generator.random_noise(wav_window_length + center_index - len(samples))
        return np.concatenate((left_half, samples[center_index:], right_padding), axis=0)


def process_and_split_data_by_chunk(wav_files, noise_generator, chunk_size=2000,
                                    window_length=5000, sample_rate=16000, file_prefix='train',
                                    data_path='/Users/matt.meng/data/speech_competition/processed_data'):
    random.shuffle(wav_files)
    chunk_data = ChunkData(data_path, file_prefix, chunk_size=chunk_size)
    start_time = time.time()
    for i in range(len(wav_files)):
        samples = align_wave_with_noise_padding(wav_files[i][0], noise_generator, window_length)
        if len(samples) != 2*window_length:
            raise ValueError('aligned waveform {} does not have right length'.format(wav_files[i][0]))
        fbank_features = logfbank(samples, sample_rate)
        chunk_data.add_data(fbank_features, wav_files[i][1])
    chunk_data.finish_data_loading()
    print('processed all {} wav_files using {:.2f} seconds'.format(len(wav_files), (time.time() - start_time)))


def preprocess_data(data_path):
    wav_files = glob.glob(os.path.join(data_path, "*", "*.wav"))
    categorized_wav_files_, categorized_sample_num_ = categorize_wav_files_by_label(wav_files)
    label2index_, index2label_ = generate_label_dict(categorized_wav_files_.keys())
    print ('the categorized sample numbers:', categorized_sample_num_)
    percentage_list = [0.8, 0.2]
    samples_by_percentage = generate_proportional_data(categorized_wav_files_, categorized_sample_num_, label2index_,
                                                       percentage_list)

    print ('the sample numbers:', len(samples_by_percentage[0]), len(samples_by_percentage[1]))
    noise_label = '_background_noise_'
    noise_generator = NoiseGenerator(data_path, noise_label, 50, auto_shuffle=True, shuffle_frequency=500)
    process_and_split_data_by_chunk(samples_by_percentage[0], noise_generator, file_prefix='train')
    process_and_split_data_by_chunk(samples_by_percentage[1], noise_generator, file_prefix='test')


def main():
    data_path = '/Users/matt.meng/data/speech_competition/train/audio'
    preprocess_data(data_path)

if __name__ == '__main__':
    main()