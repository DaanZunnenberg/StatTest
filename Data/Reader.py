import pandas as pd
from glob import glob
from tqdm import tqdm
from typing import NoReturn, MutableMapping
from pathlib import Path
import os
from pprint import pprint

__all__: list = ['Reader']

dir_prelim: str = str(Path(os.path.dirname(os.path.abspath(__file__))).parent).__add__('\\Simulations')

class Reader(object):

    def __init__(self, files: list = None, index_col: str = 'Unnamed: 0'):
        self.files = files
        self.parsed_files: MutableMapping = {}
        self.index_col = index_col
        self.read()

    def read(self) -> NoReturn:
        if self.__exists__:
            for file in tqdm(self.files): self.parsed_files = {**self.parsed_files, **{self.rename(file):pd.read_csv(file, index_col=self.index_col)}}
        else:
            raise TypeError('files not provided')
    
    @staticmethod
    def rename(string: str) -> str: return string.split('\\')[-1]

    @property
    def __exists__(self): return not isinstance(self.files, type(None))

if __name__ == '__main__':
    files: list = [
        f'{dir_prelim}\\Simulation_BivariateCorrelatedBM_gauss_200_50.csv',
        f'{dir_prelim}\\Simulation_BivariateCorrelatedBM_gauss_200_150.csv',
        f'{dir_prelim}\\Simulation_BivariateCorrelatedBM_gauss_200_365.csv',

        f'{dir_prelim}\\Simulation_BivariateNonHomogeneous_gauss_200_50.csv',
        f'{dir_prelim}\\Simulation_BivariateNonHomogeneous_gauss_200_150.csv',
        f'{dir_prelim}\\Simulation_BivariateNonHomogeneous_gauss_200_365.csv',

        f'{dir_prelim}\\Simulation_BivariateOUProcess_gauss_200_50.csv',
        f'{dir_prelim}\\Simulation_BivariateOUProcess_gauss_200_150.csv',
        f'{dir_prelim}\\Simulation_BivariateOUProcess_gauss_200_365.csv',

        f'{dir_prelim}\\SubSimulation_dist_BivariateNonHomogeneous_gauss_200_50.csv',
        f'{dir_prelim}\\SubSimulation_dist_BivariateNonHomogeneous_gauss_200_150.csv',
        f'{dir_prelim}\\SubSimulation_dist_BivariateNonHomogeneous_gauss_200_365.csv',
       ]
    
    content = Reader(files = files)

    pprint(content.parsed_files)