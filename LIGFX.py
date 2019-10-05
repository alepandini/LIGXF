import pandas as pd


class LIGFX:
    def __init__(self, input_data_filename):
        self.input_data = self.__read_input(input_data_filename)
        self.norm_data = self.__normalise()

    @staticmethod
    def __read_input(input_data_filename):
        input_data = pd.read_csv(input_data_filename, header=0)
        return input_data

    def __normalise(self):
        norm_data = (self.input_data - self.input_data.min()) / (self.input_data.max() - self.input_data.min())
        return norm_data
