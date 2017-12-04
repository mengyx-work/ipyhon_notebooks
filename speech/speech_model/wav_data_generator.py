import os
import glob
import cPickle as pickle


def read_raw_data(data_file):
    with open(data_file, 'rb') as f:
        content = pickle.load(f)
    if len(content['data']) != len(content['label']):
        raise ValueError('the raw data dimension is not consistent')
    return content['data'], content['label'], len(content['label'])


class WaveDataGenerator(object):
    def __init__(self, data_path):
        self.data_path = data_path
        self.raw_file_list = []
        self._cur_file_index = 0
        self._cur_record_index = 0

    def _prep_data(self):
        if not os.path.isdir(self.data_path):
            raise ValueError('given data path {} is not valid'.format(self.data_path))
        self.raw_file_list = sorted(glob.glob(self.data_path))
        if len(self.raw_file_list) == 0:
            raise ValueError('given data {} is empty'.format(self.data_path))
        self.data_buffer, self.label_buffer, self.buffer_size = _read_raw_data(self.raw_file_list[self._cur_file_index])

    def next_file_index(self):
        if self._cur_file_index == len(self.raw_file_list) - 1:
            self._cur_file_index = 0
        else:
            self._cur_file_index += 1
        return self._cur_file_index

    def generate_batch_iter(self, batch_size=32):
        while True:
            if self._cur_record_index + batch_size <= self.buffer_size:
                start_index = self._cur_record_index
                self._cur_record_index += batch_size
                yield (self.data_buffer[start_index:self._cur_record_index],
                       self.label_buffer[start_index:self._cur_record_index])
            else:
                start_index = self._cur_record_index
                self._cur_record_index = batch_size - (self.buffer_size - self._cur_record_index)
                data_content = self.data_buffer[start_index:]
                label_content = self.label_buffer[start_index:]
                # read partial data from next raw file
                self.data_buffer, self.label_buffer, self.buffer_size = read_raw_data(self.raw_file_list[self.next_file_index()])
                data_content.extend(self.data_buffer[:self._cur_record_index])
                label_content.extend(self.label_content[:self._cur_record_index])
                yield (data_content, label_content)

