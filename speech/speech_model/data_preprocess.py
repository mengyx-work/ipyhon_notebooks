import os, glob, re, math, random
import time
from scipy.io import wavfile
import cPickle as pickle
from python_speech_features import logfbank


def check_wav_files(wav_files, expected_sample_rate=16000, expected_sample_length=16000):
    word_dict = {}
    for wav_file in wav_files:
        sample_rate, samples = wavfile.read(wav_files[20])
        word = re.search('.*/([^/]+)/.*.wav', wav_file).group(1).lower()
        word_dict[word] = word_dict.get(word, 0) + 1
        if sample_rate != expected_sample_rate:
            print('for word {} at {}, the sample rate is different'.format(word, sample_rate))
        if len(samples) != expected_sample_length:
            print('for word {} at {}, the sample is different'.format(word, len(samples)))
    return word_dict


def categorize_wav_files_by_label(wav_files):
    '''
    categorize the wave file paths by label and hash.  `_background_noise_` does not have hash,
    use the file name.
    '''
    categorized_wav_files = {}
    categorized_sample_num = {}
    for wav_file in wav_files:
        label = re.search('.*/([^/]+)/.*.wav', wav_file).group(1).lower()
        categorized_sample_num[label] = categorized_sample_num.get(label, 0) + 1
        if label == '_background_noise_':
            hash_name = re.search('/([^/]+).wav', wav_file).group(1).lower()
        else:
            hash_name = re.search('/([^/]+)_nohash', wav_file).group(1).lower()

        if label not in categorized_wav_files:
            categorized_wav_files[label] = {}
            categorized_wav_files[label][hash_name] = [wav_file]
        else:
            if hash_name not in categorized_wav_files[label]:
                categorized_wav_files[label][hash_name] = [wav_file]
            else:
                categorized_wav_files[label][hash_name].append(wav_file)
    return categorized_wav_files, categorized_sample_num


def generate_proportional_data_sets(categorized_wav_files, categorized_sample_num, label2index,
                                    training_percentage=0.7, test_percentage=0.15, validate_percentage=0.15):
    excluded_category = ['_background_noise_']
    total_file_num = sum([categorized_sample_num[key] for key in categorized_sample_num if key not in excluded_category])
    training_samples, test_samples, validate_samples = [], [], []
    for category in categorized_wav_files.keys():
        if category in excluded_category:
            continue
        tot_training_samples = math.ceil(training_percentage * categorized_sample_num[category])
        tot_validate_samples = math.ceil(validate_percentage * categorized_sample_num[category])
        count = 0
        for hash_name in categorized_wav_files[category]:
            if count < tot_training_samples:
                for wave_file in categorized_wav_files[category][hash_name]:
                    training_samples.append((wave_file, label2index[category]))
                    count += 1
            elif count < (tot_training_samples + tot_validate_samples):
                for wave_file in categorized_wav_files[category][hash_name]:
                    validate_samples.append((wave_file, label2index[category]))
                    count += 1
            else:
                for wave_file in categorized_wav_files[category][hash_name]:
                    test_samples.append((wave_file, label2index[category]))
                    count += 1
    return training_samples, test_samples, validate_samples


class ChunkData(object):
    def __init__(self, data_path, file_prefix, chunk_size=200, feature_key='feature', label_key='label'):
        self.label_chunk = []
        self.feature_chunk = []
        self.chunk_counter = 0
        self.chunk_record_counter = 0
        self.data_path = data_path
        self.label_key = label_key
        self.chunk_size = chunk_size
        self.feature_key = feature_key
        self.file_prefix = file_prefix
        self.cur_chunk_time = time.time()

    def add_data(self, feature, label):
        self.feature_chunk.append(feature)
        self.label_chunk.append(label)
        self.chunk_record_counter += 1
        if self.chunk_record_counter == self.chunk_size:
            self._dump_chunk_data()
            self.chunk_record_counter = 0
            self.cur_chunk_time = time.time()

    def _current_file_suffix(self):
        self.chunk_counter += 1
        return str(self.chunk_counter).zfill(5)

    def _dump_chunk_data(self):
        if len(self.feature_chunk) != len(self.label_chunk):
            raise ValueError('feature data size is different from that of label')
        chunk_data = {self.feature_key: self.feature_chunk, self.label_key: self.label_chunk}
        file_suffix = self._current_file_suffix()
        file_path = os.path.join(self.data_path, '{}_data_{}.pkl'.format(self.file_prefix, file_suffix))
        with open(file_path, 'wb') as f:
            pickle.dump(chunk_data, f)
        print('saving {} records to {} using {:.2f} seconds'.format(len(self.label_chunk), file_path, time.time() - self.cur_chunk_time))

    def finish_data_loading(self):
        if len(self.feature_chunk) == 0:
            return
        self._dump_chunk_data()

def split_data_into_chunks(wav_files, chunk_size=2000, prefix='training',
                           data_path='/Users/matt.meng/data/speech_competition/processed_data'):
    random.shuffle(wav_files)
    chunk_data = ChunkData(data_path, prefix='train', chunk_size=chunk_size)
    start_time = time.time()
    for i in range(len(wav_files)):
        sample_rate, samples = wavfile.read(wav_files[i][0])
        fbank_features = logfbank(samples, sample_rate)
        chunk_data.add_data(fbank_features, wav_files[i][1])
    chunk_data.finish_data_loading()
    print('processed all {} wav_files using {:.2f} seconds'.format(len(wav_files), (time.time() - start_time)))


def generate_label_dict(data_labels):
    '''
    generate two dictionaries: label2index and index2label.
    :param data_labels (list): all the unique labels from training data
    :return label2index (dict): label -> index dict
    :return index2label (dict): index -> label dict, dict to decode the results
    '''
    known_labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence']
    unknown_label = 'unknown'
    label2index, index2label = {}, {}
    for i, label in enumerate(known_labels):
        label2index[label] = i
        index2label[i] = label
    label2index[unknown_label] = len(known_labels)
    index2label[len(known_labels)] = unknown_label

    for label in data_labels:
        if label in known_labels or label == unknown_label:
            continue
        label2index[label] = label2index[unknown_label]
    return label2index, index2label


def preprocess_raw_wav_files(train_main_path='/Users/matt.meng/data/speech_competition/train/audio'):
    wav_files = glob.glob(os.path.join(train_main_path, "*", "*.wav"))
    categorized_wav_files_, categorized_sample_num_ = categorize_wav_files_by_label(wav_files)
    label2index_, index2label_ = generate_label_dict(categorized_wav_files_.keys())
    print ('the categorized sample numbers: \n', categorized_sample_num_)
    training_samples_, test_samples_, validate_samles_ = generate_proportional_data_sets(categorized_wav_files_,
                                                                                         categorized_sample_num_,
                                                                                         label2index_)
    print ('the sample numbers:', len(training_samples_), len(test_samples_), len(validate_samles_))
    split_data_into_chunks(training_samples_)