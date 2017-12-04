import os
import unittest, glob
from speech.speech_model.data_preprocess import preprocess_raw_wav_files, categorize_wav_files_by_label, \
    generate_label_dict, generate_proportional_data_sets


class TestDataPreprocess(unittest.TestCase):

    @unittest.skip("system run")
    def test_preprocess_raw_wav_files(self):
        raw_wav_path = '/Users/matt.meng/data/speech_competition/train/audio'
        preprocess_raw_wav_files(raw_wav_path)

    @unittest.skip("system run")
    def test_generate_label_dict(self):
        train_main_path = '/Users/matt.meng/data/speech_competition/train/audio'
        wav_files = glob.glob(os.path.join(train_main_path, "*", "*.wav"))
        categorized_wav_files_, categorized_sample_num_ = categorize_wav_files_by_label(wav_files)
        label2index_, index2label_ = generate_label_dict(categorized_wav_files_.keys())
        self.assertEqual(len(label2index_), 33)
        self.assertEqual(len(index2label_), 12)

    def test_generate_proportional_data_sets(self):
        train_main_path = '/Users/matt.meng/data/speech_competition/train/audio'
        wav_files = glob.glob(os.path.join(train_main_path, "*", "*.wav"))
        categorized_wav_files_, categorized_sample_num_ = categorize_wav_files_by_label(wav_files)
        label2index_, index2label_ = generate_label_dict(categorized_wav_files_.keys())
        print ('the categorized sample numbers:', categorized_sample_num_)
        training_samples_, test_samples_, validate_samles_ = generate_proportional_data_sets(categorized_wav_files_,
                                                                                             categorized_sample_num_,
                                                                                             label2index_,
                                                                                             training_percentage=0.5,
                                                                                             test_percentage=0.25,
                                                                                             validate_percentage=0.25)
        self.assertAlmostEqual(1.*(len(training_samples_) - 2.*len(test_samples_))/len(training_samples_), 0., 2)
        self.assertAlmostEqual(1.*len(test_samples_)/len(training_samples_), 1.*len(validate_samles_)/len(training_samples_), 2)

if __name__ == '__main__':
    unittest.main()