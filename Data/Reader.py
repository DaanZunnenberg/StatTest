import pandas as pd
from glob import glob
from tqdm import tqdm
from typing import NoReturn, MutableMapping
from pathlib import Path
import numpy as np
from itertools import accumulate
import os
from pprint import pprint
import matplotlib.pyplot as plt
import requests
from pathlib import Path
from io import BytesIO

__all__: list = ['Reader']

dir_prelim: str = str(Path(os.path.dirname(os.path.abspath(__file__))).parent).__add__('\\SimulationsV1')

class DropBoxNoAPILoader:

    def __init__(self, run_on_call: bool = True): 
        self.links: list = [
            'https://www.dropbox.com/scl/fi/yq8i3zci8weumdubp3ytj/Simulation_BivariateOUProcess_process_200_365.csv?rlkey=7squbojzsp25vtoc11nh7waq0&st=scsgc5gc&dl=0',
            'https://www.dropbox.com/scl/fi/twhqtxox7jw2t21o3raqb/Simulation_BivariateOUProcess_gauss_200_365.csv?rlkey=au2e8tefaebw9mepntmgymhnv&st=9nl1su8t&dl=0',
            'https://www.dropbox.com/scl/fi/tsc374qbp9gvf05x4ncuo/Simulation_BivariateCorrelatedBM_process_200_50.csv?rlkey=9io5ir72ditdt3lpngg52fesd&st=wt4sf6kq&dl=0',
            'https://www.dropbox.com/scl/fi/lz59or4yth1gz5c18ntjj/Simulation_BivariateCorrelatedBM_gauss_200_50.csv?rlkey=jbw3vnraeck1kra1p0qmuzec2&st=a1cw2v97&dl=0',
            'https://www.dropbox.com/scl/fi/1zpiwkqfbfe5vgb06bqt1/Simulation_BivariateCorrelatedBM_process_200_150.csv?rlkey=n23y00zv93qa0e8e92m8d51ki&st=apukwxu4&dl=0',
            'https://www.dropbox.com/scl/fi/v7dxclvlqv62v37irb1h7/Simulation_BivariateCorrelatedBM_gauss_200_150.csv?rlkey=khusk31flfhj1mwzj3twc7qm2&st=m44atfif&dl=0',
            'https://www.dropbox.com/scl/fi/ienc4sel9pwihvgmuhxfx/Simulation_BivariateCorrelatedBM_process_200_365.csv?rlkey=hgqoeyu9wuxi5b71bn9mq9jxu&st=82q8lzob&dl=0',
            'https://www.dropbox.com/scl/fi/6j13ldlfp1st8ynxefs3m/Simulation_BivariateCorrelatedBM_gauss_200_365.csv?rlkey=zw4btdpmdxd6j5kcwcm8kzx1o&st=m0r689op&dl=0',
            'https://www.dropbox.com/scl/fi/upppcm6c91ib4k7p4y02z/Simulation_BivariateNonHomogeneous_process_200_50.csv?rlkey=wuqqrofp22pzf3u2rk77k1c1l&st=1tubjj05&dl=0',
            'https://www.dropbox.com/scl/fi/7ljpsbtm42qryuoif6iro/Simulation_BivariateNonHomogeneous_gauss_200_50.csv?rlkey=dp7cuep28itv2yeovbnt5t0lm&st=leo4wgis&dl=0',
            'https://www.dropbox.com/scl/fi/mvqfmv32x9668jq1lzwla/Simulation_BivariateNonHomogeneous_process_200_150.csv?rlkey=9dkwfl116w967qvsiffcyg752&st=pxuu1fuq&dl=0',
            'https://www.dropbox.com/scl/fi/pi1b03rrdv4cr9vudwdw3/Simulation_BivariateNonHomogeneous_gauss_200_150.csv?rlkey=80b5ckq01xyr944pcez4ojml7&st=49p20z5p&dl=0',
            'https://www.dropbox.com/scl/fi/5233k0nnhfx0ctw0pjkm2/Simulation_BivariateNonHomogeneous_process_200_365.csv?rlkey=eba9z38lwxjaky2ickjwjweb3&st=rktqd8ct&dl=0',
            'https://www.dropbox.com/scl/fi/s5otaof1jykwue31irft7/Simulation_BivariateNonHomogeneous_gauss_200_365.csv?rlkey=1hfjop2ymtobhj4h8ckr6bdzk&st=hxdmyp3s&dl=0',
            'https://www.dropbox.com/scl/fi/j4x35i7t7d6f38385fkcv/Simulation_BivariateOUProcess_process_200_150.csv?rlkey=tfso1fs4ifpm73s4erdr7m0qm&st=ia5pgbwe&dl=0',
            'https://www.dropbox.com/scl/fi/7anvvgkdqolsk7nkuffxe/Simulation_BivariateOUProcess_process_200_50.csv?rlkey=ew34xv2law5llve9k4mwswcr4&st=23tj5ksw&dl=0',
            'https://www.dropbox.com/scl/fi/foosmbhmst0ngeek9ssv5/Simulation_BivariateOUProcess_gauss_200_50.csv?rlkey=1qqty7vj3rtzpupj7zqzfyh57&st=kd4a4ny5&dl=0',
            'https://www.dropbox.com/scl/fi/qtt3lags03zab28uvh3iy/Simulation_BivariateOUProcess_gauss_200_150.csv?rlkey=pv30f00s332jh0mc8pby62nul&st=sxb4nvtd&dl=0',
        ]
        if run_on_call: self.main()

    @staticmethod
    def load_file_from_shared_link(shared_link):
        # Transform the shared link to a direct download link
        dl_link = shared_link.replace('www.dropbox.com', 'dl.dropboxusercontent.com').replace('?dl=0', '')

        try:
            # Send a GET request to the transformed link
            response = requests.get(dl_link)
            response.raise_for_status()  # Check if the request was successful

            content_type = response.headers.get('Content-Type')
            print(f"Content-Type: {content_type}")

            # Check the content type and handle accordingly
            if 'text/csv' in content_type or 'application/octet-stream' in content_type:
                try:
                    df = pd.read_csv(BytesIO(response.content), index_col = 'Unnamed: 0')
                except pd.errors.ParserError as e:
                    print(f"ParserError: {e}")
                    print("Inspecting file content...")
                    print(response.content.decode('utf-8'))
                    return None
            elif 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' in content_type:
                df = pd.read_excel(BytesIO(response.content))
            else:
                raise ValueError("Unsupported file type. Only CSV and Excel files are supported.")
            
            print("File loaded successfully into DataFrame")
            return df
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
        except ValueError as e:
            print(e)

    @staticmethod
    def get_name(s: str):
        return 'Simulation_Bivariate' + s.split('Simulation_Bivariate')[1].split('.csv')[0]
    
    def main(self):
        self.parsed = True
        self.file_names = []
        for shared_link in tqdm(self.links):
            name = self.get_name(shared_link)
            self.file_names.append(name)
            self.__setattr__(name, self.load_file_from_shared_link(shared_link))

    def info(self):
        assert self.__dict__.get('parsed', False)
        for name in self.file_names: 
            print(f' {name} '.center(50, '-'), end = '\n\n')
            print(self.__getattribute__(name).info())
            print('\n')
            print(f''.center(50, '-'), end = '\n')

class Reader(object):
    """
    Reader class object
    """
    def __init__(self, files: list = None, index_col: str = 'Unnamed: 0', call_dir_delim: bool = True):
        if call_dir_delim: files = [self.dir_delim + f'\\{file}' for file in files]
        self.files = files
        self.parsed_files: MutableMapping = {}
        self.index_col = index_col
        self.read()

    def read(self) -> NoReturn:
        if self.__exists__:
            try: 
                for file in tqdm(self.files): self.parsed_files = {**self.parsed_files, **{self.rename(file):pd.read_csv(file, index_col=self.index_col)}}
            except FileNotFoundError:
                self.dropbox_data = DropBoxNoAPILoader()
                for file in tqdm(self.dropbox_data.file_names): self.parsed_files = {**self.parsed_files, **{file + '.csv':self.dropbox_data.__getattribute__(file)}}
        else:
            self.dropbox_data = DropBoxNoAPILoader()
            for file in tqdm(self.dropbox_data.file_names): self.parsed_files = {**self.parsed_files, **{file + '.csv':self.dropbox_data.__getattribute__(file)}}
    
    @property
    def dir_delim(self) -> str: return str(Path(os.path.dirname(os.path.abspath(__file__))).parent).__add__('\\Simulations')

    @staticmethod
    def running_maximum(X): return list(accumulate(np.abs(X), max))

    @staticmethod
    def rename(string: str) -> str: return string.split('\\')[-1]

    @property
    def __exists__(self): return not isinstance(self.files, type(None))

    def __repr__(self): return str(self.parsed_files)

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

    data = DropBoxNoAPILoader()
    data.file_names
    data.Simulation_BivariateOUProcess_process_200_365