import unittest, time
from speech.speech_model.data_generator import DataGenerator


class TestDataGenerator(unittest.TestCase):

    @unittest.skip("local run")
    def test_prep_data(self):
        data_path = '/Users/matt.meng/data/speech_competition/processed_data'
        data_generator = DataGenerator(data_path)
        self.assertGreater(len(data_generator._raw_file_list), 1)

    @unittest.skip("local run")
    def test_next_file_index(self):
        data_path = '/Users/matt.meng/data/speech_competition/processed_data'
        data_generator = DataGenerator(data_path)
        data_generator._cur_file_index = 5
        self.assertEqual(data_generator._next_file_index(), 6)

    @unittest.skip("system run")
    def test_load_file(self):
        data_path = '/Users/matt.meng/data/speech_competition/processed_data'
        data_generator = DataGenerator(data_path)
        start_time = time.time()
        iter_num = 10
        for i in range(iter_num):
            file_index = data_generator._next_file_index()
            data_generator._load_file_by_index(file_index)
        print('for {} iterations, the average time is {:.2f} seconds'.format(iter_num, (time.time()-start_time)/iter_num))

    @unittest.skip("local run")
    def test_generate_batch_iter(self):
        data_path = '/Users/matt.meng/data/speech_competition/processed_data'
        data_generator = DataGenerator(data_path)
        batch_size = 25
        batches = data_generator.generate_batch_iter(batch_size)
        data_content, label_content = next(batches)
        print 'data shape:', data_content[0].shape
        print 'the label:', label_content[0]

    @unittest.skip("local run")
    def test_short_generate_batch_iter(self):
        data_path = '/Users/matt.meng/data/speech_competition/processed_data'
        data_generator = DataGenerator(data_path)
        batch_size = 300
        batches = data_generator.generate_batch_iter(batch_size)
        data_content, label_content = next(batches)
        self.assertEqual(len(data_content), batch_size)
        self.assertEqual(len(label_content), batch_size)

    @unittest.skip("local run")
    def test_long_generate_batch_iter(self):
        data_path = '/Users/matt.meng/data/speech_competition/processed_data'
        data_generator = DataGenerator(data_path)
        batch_size = 6000
        batches = data_generator.generate_batch_iter(batch_size)
        data_content, label_content = next(batches)
        self.assertEqual(len(data_content), batch_size)
        self.assertEqual(len(label_content), batch_size)

if __name__ == '__main__':
    unittest.main()