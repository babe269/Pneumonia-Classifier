import os
import pathlib
import unittest
from os.path import isfile, join
import sys

import src.maincd



class UnitTest(unittest.TestCase):

    # Used to test if all correct folders have been made
    def test_directory_structure(self):
        data_dir = "../chest_xray"
        data_dir = pathlib.Path(data_dir)
        self.assert_(os.path.exists(os.path.join(data_dir,"BACTERIAL")))
        self.assert_(os.path.exists(os.path.join(data_dir, "NORMAL")))
        self.assert_(os.path.exists(os.path.join(data_dir, "VIRAL")))

    # Used to test if folders contain correct files
    def test_files(self):
        data_dir = "../chest_xray"
        data_dir = pathlib.Path(data_dir)
        for filename in os.listdir(os.path.join(data_dir, "BACTERIAL")):
             self.assert_(filename.split('_')[1] == 'bacteria')
        for filename in os.listdir(os.path.join(data_dir, "VIRAL")):
             self.assert_(filename.split('_')[1] == 'virus')

    # Used to test if all files are of the correct file type
    def test_file_type(self):
        data_dir = "../chest_xray"
        data_dir = pathlib.Path(data_dir)
        for filename in os.listdir(data_dir):
            if(isfile(data_dir)):
                self.assert_(filename.split('.')[1] == 'jpg' or filename.split('.')[1] == 'jpeg')

    # Used to test if preprocessing function returns correct number of classes
    def test_number_of_classes(self):
        data_dir = "../chest_xray"
        data_dir = pathlib.Path(data_dir)
        test_ds, val_ds = src.main.preproccess_data(data_dir, 180, 180, 32)
        self.assert_(len(test_ds.class_names) == 3 )

    # Used to test if preprocessing function returns correct classes
    def test_class_names(self):
        data_dir = "../chest_xray"
        data_dir = pathlib.Path(data_dir)
        test_ds, val_ds = src.main.preproccess_data(data_dir, 180, 180, 32)
        self.assert_(test_ds.class_names[0] == 'BACTERIAL')
        self.assert_(test_ds.class_names[1] == 'NORMAL')
        self.assert_(test_ds.class_names[2] == 'VIRAL')


if __name__ == '__main__':
    unittest.main()