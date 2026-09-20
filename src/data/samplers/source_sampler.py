import pandas as pd

from .base_sampler import IterableSampler


class SourceSampler(IterableSampler):
    @property
    def source(self):
        return self._source

    def __init__(self):
        super(SourceSampler, self).__init__()

        self._source = None

    def __len__(self):
        return len(self.source)


class CSVSampler(SourceSampler):
    def __init__(self, filename):
        super(CSVSampler, self).__init__()

        self._source = pd.read_csv(filename)

    def __getitem__(self, index):
        return self.source.iloc[index].to_dict()
