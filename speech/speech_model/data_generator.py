import os, math
import glob
from data_preprocess import read_compressed_chunk_data


class DataGenerator(object):
    def __init__(self, data_path, name):
        self.name = name
        self.data_path = data_path
        self.raw_file_list = []
        self._cur_file_index = 0
        self._cur_record_index = 0
        self._prep_data()

    def _data_size(self):
        self.data_buffer, self.label_buffer, self.buffer_size = read_compressed_chunk_data(self.raw_file_list[0])
        self.frame_num = len(self.data_buffer[0])
        self.frame_size = len(self.data_buffer[0][0])
        if len(self.raw_file_list) > 1:
            _, _, tail_file_record_count = read_compressed_chunk_data(self.raw_file_list[-1])
        else:
            tail_file_record_count = self.buffer_size
        self.total_record_count = (len(self.raw_file_list) - 1) * self.buffer_size + tail_file_record_count

    def iter_num(self, batch_size):
        return int(math.ceil(1. * self.total_record_count / batch_size))

    def _prep_data(self):
        if not os.path.isdir(self.data_path):
            raise ValueError('given data path {} is not valid in generator {}'.format(self.data_path), self.name)
        self.raw_file_list = sorted(glob.glob(os.path.join(self.data_path, "*.pkl")))
        if len(self.raw_file_list) == 0:
            raise ValueError('given data {} in `DataGenerator` {} is empty'.format(self.data_path, self.name))
        self._data_size()

    def _load_file(self, file_index):
        self.data_buffer, self.label_buffer, self.buffer_size = read_compressed_chunk_data(self.raw_file_list[file_index])

    def _next_file_index(self):
        if self._cur_file_index >= len(self.raw_file_list) - 1:
            print('iterator from {} completes one epoch'.format(self.name))
            self._cur_file_index = 0
        else:
            self._cur_file_index += 1
        return self._cur_file_index

    def generate_batch_iter(self, batch_size=32):
        print('iterator created from {}, each epoch contains {} iterations'.format(self.name, self.iter_num(batch_size)))
        while True:
            if self._cur_record_index + batch_size <= self.buffer_size:
                start_index = self._cur_record_index
                self._cur_record_index += batch_size
                yield self.data_buffer[start_index:self._cur_record_index],\
                      self.label_buffer[start_index:self._cur_record_index]
            else:
                start_index = self._cur_record_index
                self._cur_record_index = batch_size - (self.buffer_size - self._cur_record_index)
                data_content = self.data_buffer[start_index:]
                label_content = self.label_buffer[start_index:]
                self._load_file(self._next_file_index())
                # handle the case when `batch_size` is much bigger than `buffer_size`
                while self._cur_record_index > self.buffer_size:
                    self._cur_record_index -= self.buffer_size
                    data_content.extend(self.data_buffer)
                    label_content.extend(self.label_buffer)
                    self._load_file(self._next_file_index())
                data_content.extend(self.data_buffer[:self._cur_record_index])
                label_content.extend(self.label_buffer[:self._cur_record_index])
                yield data_content, label_content


def main():
    data_path = '/Users/matt.meng/data/speech_competition/processed_data'
    data_generator = DataGenerator(data_path)
    print(data_generator.iter_num(32))

if __name__ == '__main__':
    main()