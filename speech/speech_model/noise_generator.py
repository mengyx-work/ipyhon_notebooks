import os, re
import  glob
import numpy as np
from scipy.io import wavfile


def collect_wav_files_by_label(data_path, label='_background_noise_'):
    '''
    collect all the wav files by given ``label``
    :param data_path: the parent path for wav files
    :param label: the label for wav files
    :return: a list of wav files
    '''
    if not isinstance(data_path, str):
        raise ValueError('need a data path')
    wav_files = glob.glob(os.path.join(data_path, "*", "*.wav"))
    if len(wav_files) == 0:
        raise ValueError('{} is an empty data path'.format(data_path))

    selected_wav_files = []
    for wav_file in wav_files:
        word = re.search('.*/([^/]+)/.*.wav', wav_file).group(1).lower()
        if word == label:
            selected_wav_files.append(wav_file)
    return selected_wav_files


def _random_noise(file_name_, length, norm=300):
    '''
    generate a segment of white noise from ``file_name_``
    :param file_name_:
    :param length:
    :param norm:
    :return:
    '''
    if length == 0:
        return np.array([])
    sample_rate, samples = wavfile.read(file_name_)
    if len(samples) < length:
        raise ValueError('the noise waveform is not long enough')
    start_index = np.random.randint(len(samples)-length)
    #print 'the start_index: ', start_index
    selected_samples = samples[start_index:(start_index + length)].astype(float)
    max_amplitude = np.max(np.abs(selected_samples))
    if max_amplitude == 0:
        max_amplitude = 1
    selected_samples = selected_samples*(norm/max_amplitude)
    return selected_samples.astype(int)


class NoiseGenerator(object):
    MAX_RANDOM_SEED = 99999

    def __init__(self, data_path, noise_label, norm=50, auto_shuffle=False, shuffle_frequency=500, seed=999):
        self.noise_files = collect_wav_files_by_label(data_path, noise_label)
        self.counter = 0
        self.random_seed = seed
        self.amplitude_norm = norm
        np.random.seed(self.random_seed)
        self.auto_shuffle = auto_shuffle
        self.shuffle_frequency = shuffle_frequency
        self.file_index = np.random.randint(len(self.noise_files))

    def update_amplitude_norm(self, norm):
        self.amplitude_norm = norm

    def update_random_seed(self, new_seed=None):
        if new_seed is not None:
            self.random_seed = new_seed
        else:
            self.random_seed = np.random.randint(self.MAX_RANDOM_SEED)
        np.random.seed(self.random_seed)

    def update_file_index(self):
        self.file_index = np.random.randint(len(self.noise_files))

    def random_noise(self, length):
        self.counter += 1
        if self.counter % self.shuffle_frequency == 0:
            self.counter = 0
            if self.auto_shuffle:
                self.update_file_index()
        return _random_noise(self.noise_files[self.file_index], length, self.amplitude_norm)