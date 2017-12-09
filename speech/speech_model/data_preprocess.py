import os, glob, re, math, random
import time
from scipy.io import wavfile
import cPickle as pickle
from python_speech_features import logfbank
from chunk_data import ChunkData

SEED = 448
random.seed(SEED)


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


def generate_proportional_data(categorized_wav_files, categorized_sample_num, label2index, percentage_list=[]):
    if sum(percentage_list) != 1.:
        raise ValueError('the train/validation/test split percentage does not match')
    excluded_category = ['_background_noise_']
    samples_by_percentage = [[] for _ in percentage_list]
    print('total samples: {}'.format(sum(categorized_sample_num.values())))
    for category in categorized_wav_files.keys():
        if category in excluded_category:
            continue
        hash_name_list = categorized_wav_files[category].keys()
        hash_name_index = 0
        category_total = 0
        for i, percentage in enumerate(percentage_list):
            tot_counts = math.ceil(percentage * categorized_sample_num[category])
            count = 0
            while count <= tot_counts and hash_name_index < len(hash_name_list):
                hash_name = hash_name_list[hash_name_index]
                hash_name_index += 1
                for wave_file in categorized_wav_files[category][hash_name]:
                    samples_by_percentage[i].append((wave_file, label2index[category]))
                    count += 1
            category_total += count
        if category_total != categorized_sample_num[category]:
            print('error! expect {}, found {} after processing'.format(categorized_sample_num[category],
                                                                       category_total))
    return samples_by_percentage


def generate_fixed_proportional_data(categorized_wav_files, categorized_sample_num, label2index,
                                     training_percentage=0.7, test_percentage=0.15, validate_percentage=0.15):
    if (training_percentage + test_percentage + validate_percentage) != 1.:
        raise ValueError('the train/validation/test split percentage does not match')
    excluded_category = ['_background_noise_']
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


def read_compressed_chunk_data(data_file, feature_key='feature', label_key='label'):
    with open(data_file, 'rb') as f:
        content = pickle.load(f)
    if len(content[feature_key]) != len(content[label_key]):
        raise ValueError('the raw data dimension is not consistent')
    return content[feature_key], content[label_key], len(content[label_key])


def process_and_split_data_by_chunk(wav_files, chunk_size=2000, prefix='training',
                                    data_path='/Users/matt.meng/data/speech_competition/processed_data'):
    random.shuffle(wav_files)
    chunk_data = ChunkData(data_path, file_prefix='train', chunk_size=chunk_size)
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

'''
def preprocess_data(train_main_path='/Users/matt.meng/data/speech_competition/train/audio'):
    wav_files = glob.glob(os.path.join(train_main_path, "*", "*.wav"))
    categorized_wav_files_, categorized_sample_num_ = categorize_wav_files_by_label(wav_files)
    label2index_, index2label_ = generate_label_dict(categorized_wav_files_.keys())
    print ('the categorized sample numbers:', categorized_sample_num_)
    percentage_list = [0.8, 0.2]
    samples_by_percentage = generate_proportional_data(categorized_wav_files_, categorized_sample_num_, label2index_, percentage_list)
    process_and_split_data_by_chunk(samples_by_percentage[0], prefix='train')
    process_and_split_data_by_chunk(samples_by_percentage[1], prefix='train')


def main():
    raw_wav_path = '/Users/matt.meng/data/speech_competition/train/audio'
    preprocess_data(raw_wav_path)

if __name__ == '__main__':
    main()
'''