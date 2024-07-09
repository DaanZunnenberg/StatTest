#!/usr/bin/env python
# -*- coding: utf-8 -*-
r""" 
PS **
Python 3.10.11 (tags/v3.10.8:aaaf517, Jul 09 2024, 00:50:30) [MSC v.1933 64 bit (AMD64)] on win32
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
import typing, datetime
import requests, urllib, warnings, math, os, nest_asyncio, functools, typing, glob, json, sys, platform, datetime, inspect, itertools, pprint
from numpy.core import overrides
from urllib.request import urlopen 
from sys import path
from tqdm import tqdm
import cryptocompare
from pandas._typing import (
    F,
    T,
)

class version:

    def __init__(self) -> None:
        self._version = '\n'.join(
            [
        'Current version: 1.0.3',
        'Last updated: 2024/07/08'
        ])

    @property
    def __version__(self):
        print(self._version)
        return True
    
    def __str__(self):
        return self.__version__

    def __repr__(self):
        return self._version

VERSION = version()
__VERSION__ = VERSION.__repr__()
assert VERSION.__version__

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

def deprecate_kwarg(
    old_arg_name: str,
    new_arg_name: str | None,
    mapping: typing.Mapping[typing.Any, typing.Any] | typing.Callable[[typing.Any], typing.Any] | None = None,
    stacklevel: int = 2,
) -> typing.Callable[[F], F]:
    """
    Decorator to deprecate a keyword argument of a function.

    Parameters
    ----------
    old_arg_name : str
        Name of argument in function to deprecate
    new_arg_name : str or None
        Name of preferred argument in function. Use None to raise warning that
        ``old_arg_name`` keyword is deprecated.
    mapping : dict or callable
        If mapping is present, use it to translate old arguments to
        new arguments. A callable must do its own value checking;
        values not found in a dict will be forwarded unchanged.

    
    """
    if mapping is not None and not hasattr(mapping, "get") and not callable(mapping):
        raise TypeError(
            "mapping from old to new argument values must be dict or callable!"
        )

    def _deprecate_kwarg(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> typing.Callable[..., typing.Any]:
            old_arg_value = kwargs.pop(old_arg_name, None)

            if old_arg_value is not None:
                if new_arg_name is None:
                    msg = (
                        f"the {repr(old_arg_name)} keyword is deprecated and "
                        "will be removed in a future version. Please take "
                        f"steps to stop the use of {repr(old_arg_name)}"
                    )
                    warnings.warn(msg, FutureWarning, stacklevel=stacklevel)
                    kwargs[old_arg_name] = old_arg_value
                    return func(*args, **kwargs)

                elif mapping is not None:
                    if callable(mapping):
                        new_arg_value = mapping(old_arg_value)
                    else:
                        new_arg_value = mapping.get(old_arg_value, old_arg_value)
                    msg = (
                        f"the {old_arg_name}={repr(old_arg_value)} keyword is "
                        "deprecated, use "
                        f"{new_arg_name}={repr(new_arg_value)} instead."
                    )
                else:
                    new_arg_value = old_arg_value
                    msg = (
                        f"the {repr(old_arg_name)}' keyword is deprecated, "
                        f"use {repr(new_arg_name)} instead."
                    )

                warnings.warn(msg, FutureWarning, stacklevel=stacklevel)
                if kwargs.get(new_arg_name) is not None:
                    msg = (
                        f"Can only specify {repr(old_arg_name)} "
                        f"or {repr(new_arg_name)}, not both."
                    )
                    raise TypeError(msg)
                else:
                    kwargs[new_arg_name] = new_arg_value
            return func(*args, **kwargs)

        return typing.cast(F, wrapper)

    return _deprecate_kwarg

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



class historical_data:
    """
    Class to load data without an API key
    """

    
    def __init__(self, module: typing.Any = cryptocompare, **kwargs) -> ...:
        self.__construct(module = module)
        self.func_args = {**kwargs, **{'coin':kwargs.get('ticker_symbol')}}
        self.snipped_func_args = self.func_args.copy()
        self.flag_attr_setter: bool = False
        return None
    
    @classmethod
    def func_error(NameError: NameError = NameError):
        print(NameError.__doc__)
        
    def __construct(self, module: typing.Any = cryptocompare):
        for __name, __func in inspect.getmembers(cryptocompare, inspect.isfunction):
            self.__setattr__(__name, __func)
            #print(__name, 'set')
        print('\nFunction ready for use!')

    def __repr__(self) -> str:
        print('\033[95m' + 'The historical_data class has the following attributes', '\033[0m')
        __s: str = ''
        for attr in self.__dict__.keys():
            if attr == 'func_args':
                continue
            __s += f' historical_data.{attr}(...) '.center(100, '-') + '\n' + ''.join([self.__getattribute__(attr).__doc__.replace(':','\t-')]) + '\n'
        return __s
    
    @property
    def info(self):
        print(self.__repr__())
    
    @staticmethod
    def chuncks(from_date, to_date, chunks: float = 10.):
        from_date = pd.to_datetime(from_date)
        to_date = pd.to_datetime(to_date)
        chunks = min((to_date  - from_date).total_seconds() / 60, chunks)
        diff = max((to_date  - from_date) / chunks, datetime.timedelta(minutes = 60))
        for i in arange(chunks + 1):
            yield (from_date + diff * i).strftime("%Y-%m-%d %H:%M:%S")

    def main_chunck(self, func: str = 'get_price', f: typing.Any = lambda x : x):
        raise NotImplementedError
        func: typing.Any = self.__getattribute__(func)
        __func__args = dict((key,value) for key, value in self.func_args.items() if key in inspect.getfullargspec(func)[0])
        try: 
            _f = f(func(**__func__args))
            assert len(_f)
            return _f
        except AssertionError as e: print(e.__doc__)
        
        __chunks = list(self.chuncks(from_date = self.func_args['toTs'] - datetime.timedelta(days = 6), to_date = self.func_args['toTs'], chunks = 6 * 24 * 60 / 3000))
        _tr = __func__args.pop('toTs')
        _tr = __func__args.pop('limit')
        __final = pd.DataFrame()
        for l, r in list(zip(__chunks[:-1], __chunks[1:])):
            print(r)
            __final = __final.append(f(func(limit = 3000, toTs = r, **__func__args)))

        return __final
    
    @property
    def combinatorics(self):
        __exchanges : list = list(self.snipped_func_args.get('exchange_name'))
        __symbols   : list = list(self.snipped_func_args.get('coin'))
        __valuators : list = list(self.snipped_func_args.get('currency'))
        __final     : list = list(itertools.product(__exchanges, __symbols, __valuators, repeat = 1))
        print(f'@combinatorics combinations: {len(__final)}')
        return __final
    
    def overwrite(self, **kwargs):
        for __attr, __val in kwargs.items():
            self.func_args[__attr] = __val

    @staticmethod
    def assert_inputs(__func__args: typing.Any, __func: typing.Any):
        __func_arg: list = inspect.getfullargspec(__func)[0]
        for __arg in __func_arg:
            try: 
                assert __arg in __func__args
                return True
            except AssertionError as e: 
                print(e.__doc__, f'Missing argument: {__arg}')
                return False

    def main(self, func: str = 'get_price', f: typing.Any = pd.DataFrame.from_dict, keys: list = ['close', 'identify']) -> ...:
        ret : pd.DataFrame = pd.DataFrame()
        func: typing.Any = self.__getattribute__(func)
        for idx, (__Exch, __Asset, __Valuator) in enumerate(self.combinatorics):
            self.overwrite(exchange_name = __Exch, coin = __Asset, currency = __Valuator)
            __func__args = dict((key,value) for key, value in self.func_args.items() if key in inspect.getfullargspec(func)[0])
            if self.assert_inputs(__func__args, func):
                __appendable = f(func(**__func__args))
                __appendable[['symbol','exchange','identify']] = list([__Asset + __Valuator, __Exch, __Asset + __Valuator + ' ' + __Exch])
                ret = pd.concat([ret, __appendable], ignore_index=True)
            else:
                return
        self.data = ret
        try: return self.__graph__(ret)
        except: return ret[keys]
        else: return None
    
    @deprecate_kwarg('toTs', 'to_date')
    def load_ex(self, from_date: str, to_date: str, **kwargs):
        _final: pd.DataFrame = pd.DataFrame()
        _dd: typing.Union[int, datetime.datimetime, typing.Any] = pd.to_datetime(from_date) + datetime.timedelta(hours = kwargs.get('limit', 2000))
        _df: typing.Union[int, datetime.datimetime, typing.Any] = pd.to_datetime(to_date)
        while _dd <= _df:
            self.overwrite(toTs = _dd)
            _sub = self.main('get_historical_price_hour', f = pd.DataFrame.from_dict)
            try: _sub.index = pd.to_datetime(_sub.time)
            except: pass
            _final = pd.concat([_final, _sub], ignore_index=False)
            _dd += datetime.timedelta(hours = kwargs.get('limit', 2000))
        
        return _final

    @staticmethod
    def __graph__(ret, key = ['close', 'identify']):
        ret.set_index("time", inplace=True)
        ret.index = pd.to_datetime(ret.index, unit='s')
        ret['datetimes'] = ret.index
        ret['datetimes'] = ret['datetimes'].dt.strftime('%Y-%m-%d')
        return ret[key]
    
    @staticmethod
    def reduce_dict_complexity(_dict: dict, _ET_trades: typing.Any):
        __update: dict = {}
        for key, value in _dict.items():
            __value = value[['exchange']]
            __indices: list = []
            for __date in tqdm(_ET_trades):
                __indices.append(__value[__value.index <= __date].index[-1])
            __update = {**__update, **{key:value.loc[__indices]}}
        return __update

    def daily_statistics(self, orderbook_dict, trade_timestamps, trade_amounts, trade_sides, **kwargs):
        __orderbook_dict: dict = orderbook_dict if self.reduced_orderbook_dict == -1 else self.reduced_orderbook_dict #self.reduce_dict_complexity(orderbook_dict, trade_timestamps)
        __final: pd.DataFrame = pd.DataFrame(columns = [f'filled_{exchange}' for exchange in __orderbook_dict.keys()])
        for timestamp, amount, side in list(zip(trade_timestamps, trade_amounts, trade_sides)):
            try:
                __sub = pd.DataFrame.from_dict(self.main(__orderbook_dict, amount, timestamp, side, track_trades = True, **kwargs))
                __sub.index = __sub.timestamp
                __sub = __sub[[_ for _ in __sub.columns if _ != 'timestamp']]
                __sub.index.name = 'timestamp'
                __final = pd.concat([__final, __sub])
            except:
                print(f'\033[93m@root> Size order ({amount}) too large for aggregated orderbook at time {timestamp}\033[0m')
        __final.index.name = 'timestamp'
        return __final

    def create_dict(self, trade_timestamps, exchanges, **kwargs):
        __final: dict = {}
        for __exch in exchanges:
            __appendable: pd.DataFrame = pd.DataFrame(data = zeros((kwargs.get('depth', 50) * 4 + 2, 1)).T, 
                                                columns = ['exchange', 'symbol'] + list(itertools.chain.from_iterable([[f'asks[{i}].price', f'asks[{i}].amount', f'bids[{i}].price', f'bids[{i}].amount'] for i in range(kwargs.get('depth', 50))])))
            for ts in tqdm(trade_timestamps):
                __appendable = self.convert(
                    json_string = self.historical_book_snapshot(minute_ts = self.ts_to_mts(ts), market = __exch, **kwargs), 
                    appendable = __appendable,
                    timestamp = ts
                )
            
            __final = {**__final, **{__exch:__appendable[1:]}}
        return __final

    @staticmethod
    def ts_to_mts(date_string: str = str(datetime.datetime.now())):
        return int(pd.to_datetime(date_string).value / (10 ** 9)) - 3600

    def historical_book_snapshot(
        self,
        market: str = 'binance',
        instrument: str = 'BTC-USDT',
        minute_ts: int = 0,
        depth: int = 50,
        apply_mapping: str = 'true',
        response_format: str = 'JSON',
        return_404_on_empty_response: str = 'true',
        api_key: str = '0000000000000000'
    ) -> typing.Any:
        minute_ts = minute_ts if type(minute_ts) == int else self.ts_to_mts(minute_ts)
        url: str = f'https://data-api.cryptocompare.com/spot/v1/historical/orderbook/l2/snapshot/minute?market={market}&instrument={instrument}&minute_ts={minute_ts}&depth={depth}&apply_mapping={apply_mapping}&response_format={response_format}&return_404_on_empty_response={return_404_on_empty_response}&api_key={api_key}'
        print(url)
        response = urlopen(url) 
        return json.loads(response.read()) 

    def main_generator(self, trade_timestamps, exchanges, **kwargs):
        _matrix = matrix(zeros((len(trade_timestamps), kwargs.get('depth', 50) * 4)))
        for _exch in exchanges:
            j = 0
            for generated in tqdm(self.data_generator(trade_timestamps = trade_timestamps, _exch = _exch, **kwargs)):
                _matrix[j:j + 1,] = generated
                j += 1
            print(_matrix)
            
    def data_generator(self, trade_timestamps, _exch, **kwargs):
        for ts in trade_timestamps:
            yield self.data_parser(self.historical_book_snapshot(minute_ts = self.ts_to_mts(ts), market = _exch, **kwargs))
        
    @staticmethod
    def data_parser(json_string: typing.Any):
        bids_price, bids_value = [], []
        asks_price, asks_value = [], []
        for idx, value in enumerate(json_string.get('Data').get('BIDS')):
            bids_price.append(value.get('PRICE'))
            bids_value.append(value.get('QUANTITY'))
        for idx, value in enumerate(json_string.get('Data').get('ASKS')):
            asks_price.append(value.get('PRICE'))
            asks_value.append(value.get('QUANTITY'))
        return bids_price + bids_value + asks_price + asks_value

    @staticmethod
    def convert(json_string: typing.Any, appendable: pd.DataFrame, **kwargs) -> ...:
        depth = kwargs.get('depth', 50)
        columns = ['exchange', 'symbol'] + [f'bids[{i}].price' for i in range(depth)] + [f'bids[{i}].amount' for i in range(depth)] + [f'asks[{i}].price' for i in range(depth)] + [f'asks[{i}].amount' for i in range(depth)]
        bids_price, bids_value = [], []
        asks_price, asks_value = [], []
        for idx, value in enumerate(json_string.get('Data').get('BIDS')):
            bids_price.append(value.get('PRICE'))
            bids_value.append(value.get('QUANTITY'))
        for idx, value in enumerate(json_string.get('Data').get('ASKS')):
            asks_price.append(value.get('PRICE'))
            asks_value.append(value.get('QUANTITY'))
        _data_dict_values = [json_string.get('Data').get('MARKET'), json_string.get('Data').get('INSTRUMENT')] + bids_price + bids_value + asks_price + asks_value

        __df: pd.DataFrame = pd.DataFrame(dict(zip(columns, _data_dict_values)), index = [pd.to_datetime(kwargs.get('timestamp'))])
        return pd.concat([appendable, __df], ignore_index = False)

class information(historical_data):

    def __init__(self, module: typing.Any = cryptocompare, **kwargs) -> ...:
        super().__init__(module, **kwargs)
        
    def __repr__(self) -> str:
        return ''
    
    @staticmethod
    def wrapper(**kwargs):
        return {'exchanges':'get_exchanges','coins':'get_coin_list'}
    
    def main(self, __information: typing.Union[str, tuple] = ('exchanges',), __ret: bool = False, visualisation: str = 'dataframe', **kwargs):
        for __func in __information:
            func: typing.Any = self.__getattribute__(self.wrapper().get(__func))
            match visualisation:
                case 'dataframe':
                    return pd.DataFrame.from_dict(func()).T
                case _:
                    pprint.pprint(func())
            # if __ret:
            #     yield func()

if __name__ == '__main__':
    #ccdata      = Crypto_Compare(api_path = os.getcwd())
    #information = information()

    func_args = {
        'ticker_symbol'         : ('BTC',),
        'currency'              : ('USDT',),
        'limit'                 : 2000,
        'exchange_name'         : ('binance',),
    }

    hd = historical_data(**func_args)
    df = hd.load_ex(from_date = '2023-01-01', to_date = '2023-12-01', func = 'get_historical_price_day', f = pd.DataFrame.from_dict)
    hd.main(func = 'get_historical_price_day', f = pd.DataFrame.from_dict)
    
    # df.datetimes = pd.to_datetime(df.datetimes)
    import matplotlib.pyplot as plt
    plt.plot(df.index, df.close)