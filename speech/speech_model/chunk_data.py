import os, time
import cPickle as pickle


class ChunkData(object):
    def __init__(self, data_path, file_prefix, chunk_size=200, feature_key='feature', label_key='label'):
        self.label_chunk = []
        self.feature_chunk = []
        self.chunk_counter = 0
        self.chunk_record_counter = 0
        self.feature_num = None
        self.feature_size = None
        self.data_path = data_path
        self.label_key = label_key
        self.chunk_size = chunk_size
        self.feature_key = feature_key
        self.file_prefix = file_prefix
        self.cur_chunk_time = time.time()

    def _reset_buffer(self):
        self.label_chunk = []
        self.feature_chunk = []
        self.chunk_record_counter = 0
        self.cur_chunk_time = time.time()

    def add_data(self, feature, label):
        if self.feature_num is None or self.feature_size is None:
            self.feature_num = len(feature)
            self.feature_size = len(feature[0])
        else:
            if self.feature_num != len(feature):
                raise ValueError('feature with wrong ``feature_num`` is found')
            if self.feature_size != len(feature[0]):
                raise ValueError('feature with wrong ``feature_size`` is found')

        self.feature_chunk.append(feature)
        self.label_chunk.append(label)
        self.chunk_record_counter += 1
        if self.chunk_record_counter == self.chunk_size:
            self._dump_chunk_data()
            self._reset_buffer()

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