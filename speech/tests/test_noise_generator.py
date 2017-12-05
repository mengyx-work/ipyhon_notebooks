import unittest
from speech.speech_model.noise_generator import NoiseGenerator


class TestNoiseGenerator(unittest.TestCase):
    @unittest.skip("local run")
    def test_update_amplitude_norm(self):
        data_path = '/Users/matt.meng/data/speech_competition/train/audio'
        noise_label = '_background_noise_'
        noise_generator = NoiseGenerator(data_path, noise_label, 50)
        new_norm = 100
        noise_generator.update_amplitude_norm(new_norm)
        self.assertEqual(noise_generator.amplitude_norm, new_norm)

    @unittest.skip("local run")
    def test_update_file_index(self):
        data_path = '/Users/matt.meng/data/speech_competition/train/audio'
        noise_label = '_background_noise_'
        noise_generator = NoiseGenerator(data_path, noise_label, 50)
        file_index = noise_generator.file_index
        noise_generator.update_file_index()
        self.assertNotEqual(noise_generator.file_index, file_index)

    @unittest.skip("local run")
    def test_random_noise(self):
        data_path = '/Users/matt.meng/data/speech_competition/train/audio'
        noise_label = '_background_noise_'
        noise_generator = NoiseGenerator(data_path, noise_label, 50)
        length = 500
        noise = noise_generator.random_noise(length)
        self.assertEqual(len(noise), length)

    @unittest.skip("local run")
    def test_auto_shuffle(self):
        data_path = '/Users/matt.meng/data/speech_competition/train/audio'
        noise_label = '_background_noise_'
        noise_generator = NoiseGenerator(data_path, noise_label, 50, auto_shuffle=True, shuffle_frequency=1)
        file_index = noise_generator.file_index
        _ = noise_generator.random_noise(500)
        self.assertNotEqual(noise_generator.file_index, file_index)
