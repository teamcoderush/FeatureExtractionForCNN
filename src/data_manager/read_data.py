#! /src/data_manager/read_data.py
# This file load the data from the CSV files
import os

__author__ = "Dulanjaya Tennekoon"


class ReadData(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()