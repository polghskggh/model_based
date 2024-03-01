import csv
from abc import abstractmethod


class CsvWriter:
    def __init__(self, filename, header: list[str]):
        self.filename = filename + ".csv"
        self.prepare_header(header)

    # prepare csv header
    def prepare_header(self, header: list[str]):
        with open(self.filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows([header])

    # flush data into file
    def store_data(self, data: list[list]):
        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)
            file.flush()
