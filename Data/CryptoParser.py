#!/usr/bin/env python
# -*- coding: utf-8 -*-
r""" 
The Gueant-Lehalle-Fernandez-Tapia equations implementation for optimal quoting in the market making strategy
PS **
Python 3.10.8 (tags/v3.10.8:aaaf517, Feb 14 2023, 16:50:30) [MSC v.1933 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
Authors:
    - Daan

Date:
    - 2024

----------------------

Issues:

    - **
    
    - **

TODO:

    - **
"""
from statsmodels.tsa.\
    vector_ar.var_model import *
from statsmodels.tsa.\
    ar_model            import *
from numpy              import *
from pandas             import *
from glob               import *
from tardis_dev         import datasets
from typing import Any, Union
from sys import path

class bit_loader():
    import nest_asyncio
    nest_asyncio.apply()
    def __init__(self, **kwargs) -> None:

        self.kwargs = (lambda parser : parser if type(parser) in list([dict]) else dict())(kwargs)

    def download(self):
        datasets.download(
            **self.kwargs
        )

    def load(self):
        return concat((read_csv(f) for f in glob(self.__dict__.get('kwargs').get('download_dir').__add__(f'\\*.gz'))), ignore_index=True)

def main_wrapper(transform: bool | None = False, 
                 cols: list | str = '', 
                 api_key: str | None = "API_KEY",
                 **arg: dict) -> Any | DataFrame | matrix:
    set_args = arg
    _setter = lambda parser : to_datetime(multiply(parser, 10 ** 3))
    match set_args:
        case _ if str('cols') not in set_args.keys():
            pass
        case _ if str('cols') in set_args.keys():
            set_args.pop('cols')

    if path.exists(arg.get('download_dir', '-').__add__(f'\\*.gz')):
        match transform:
            case _ if transform:
                if type(cols) == list:
                    _s = bit_loader(**set_args).load()
                    _s.index = _setter(_s.timestamp)
                    return _s[cols]
                else:
                    _s = bit_loader(**set_args).load()
                    _s.index = _setter(_s.timestamp)
                    return _s
            case _ if not transform:
                if type(cols) == list:
                    return bit_loader(**set_args).load()[cols]
                else:
                    return bit_loader(**set_args).load()[cols]
    else:
        bit_loader(**set_args, api_key = api_key).download()
        match transform:
            case _ if transform:
                if type(cols) == list:
                    _s = bit_loader(**set_args).load()
                    _s.index = _setter(_s.timestamp)
                    return _s[cols]
                else:
                    _s = bit_loader(**set_args).load()
                    _s.index = _setter(_s.timestamp)
                    return _s
            case _ if not transform:
                if type(cols) == list:
                    return bit_loader(**set_args).load()[cols]
                else:
                    return bit_loader(**set_args).load()[cols]
                

def main(_d: Any | DataFrame | Union[int, int],**kwargs: Any | tuple | dict ) -> Union[matrix, matrix] | DataFrame:
    match _d:
        case _ if type(_d) in list([DataFrame, ndarray, matrix, array]):pass
        case _:raise LookupError()