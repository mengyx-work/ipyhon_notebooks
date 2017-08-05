import cPickle as pickle

class DataGenerator(object):

    def __init__ (self, pickle_file_path):
        self._cur_index = 0
        with open(pickle_file_path, 'rb') as input_stream:
            self.data = pickle.load(input_stream)
        self.titles = self.data['titles']
        self.reverse_token_dict = self.data['reverse_token_dict']
        self.data_size = len(self.titles)
    
    def generate_sequence(self, batch_size):
        if batch_size >= 2 * self.data_size:
            raise ValueError("the batch_size can not be more than two times the data_size")
        
        while True:
            if self._cur_index + batch_size <= self.data_size:
                start_index = self._cur_index
                self._cur_index += batch_size
                yield self.titles[start_index : self._cur_index]
            else:
                start_index = self._cur_index
                self._cur_index = self._cur_index + batch_size - self.data_size
                yield self.titles[start_index : self.data_size].extend(self.titles[0 : self._cur_index])

