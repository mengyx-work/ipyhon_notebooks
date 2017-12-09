import unittest
import glob, os
from speech.speech_model.noise_generator import NoiseGenerator
from speech.speech_model.data_alignment import align_wave_with_noise_padding, preprocess_data, \
    process_and_split_data_by_chunk
from speech.speech_model.data_preprocess import categorize_wav_files_by_label, generate_label_dict, \
    generate_fixed_proportional_data


class TestDataAlignment(unittest.TestCase):
    @unittest.skip("local run")
    def test_align_wave_with_noise_padding(self):
        data_path = '/Users/matt.meng/data/speech_competition/train/audio'
        noise_label = '_background_noise_'
        file_name = '/Users/matt.meng/data/speech_competition/train/audio/_background_noise_/dude_miaowing.wav'
        noise_generator = NoiseGenerator(data_path, noise_label, 50)
        window_length = 5000
        wav = align_wave_with_noise_padding(file_name, noise_generator, window_length)
        self.assertEqual(len(wav), 2*window_length)

    @unittest.skip("local run")
    def test_preprocess_data(self):
        data_path = '/Users/matt.meng/data/speech_competition/train/audio'
        preprocess_data(data_path)

    def test_process_and_split_data_by_chunk(self):
        data_path = '/Users/matt.meng/data/speech_competition/train/audio'
        noise_label = '_background_noise_'

        wav_files = glob.glob(os.path.join(data_path, "*", "*.wav"))
        categorized_wav_files_, categorized_sample_num_ = categorize_wav_files_by_label(wav_files)
        label2index_, index2label_ = generate_label_dict(categorized_wav_files_.keys())
        print ('the categorized sample numbers:', categorized_sample_num_)
        training_samples_, test_samples_, validate_samles_ = generate_fixed_proportional_data(categorized_wav_files_,
                                                                                              categorized_sample_num_,
                                                                                              label2index_)
        noise_generator = NoiseGenerator(data_path, noise_label, 50, auto_shuffle=True, shuffle_frequency=1)
        chunk_size = 5000
        process_and_split_data_by_chunk(test_samples_[:chunk_size], noise_generator)
