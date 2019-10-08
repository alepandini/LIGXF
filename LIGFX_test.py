import unittest
from LIGFX import *


class TestLIGFX(unittest.TestCase):

    def test_constructor(self):
        ligfx_object = LIGFX("test/test001_input.csv")
        dataset_dim = (133, 28)
        self.assertEqual(ligfx_object.input_x.shape, dataset_dim,
                         "Dataset size should [%d,%d]" % (dataset_dim[0], dataset_dim[1]))

    def test_normalisation(self):
        ligfx_object = LIGFX("test/test001_input.csv", normalise=True)
        data_point = 0.5258266956311851
        self.assertEqual(ligfx_object.input_x.iloc[0][0], data_point, "Dataset not correctly normalised")


if __name__ == '__main__':
    unittest.main()
