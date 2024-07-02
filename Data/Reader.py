import pandas as pd
from glob import glob
from tqdm import tqdm
from typing import NoReturn, MutableMapping

class Reader(object):

    def __init__(self, files: list = None, index_col: str = 'Unnamed: 0'):
        self.files = files
        self.parsed_files: MutableMapping = {}
        self.index_col = index_col
        self.read()

    def read(self) -> NoReturn:
        if self.__exists__:
            for file in tqdm(self.files): self.parsed_files = {**self.parsed_files, **{file:pd.read_csv(file, index_col=self.index_col)}}
        else:
            raise TypeError('files not provided')
        
    @property
    def __exists__(self): return isinstance(self.files, type(None))