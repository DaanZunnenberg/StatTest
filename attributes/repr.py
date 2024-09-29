from typing import (
    NoReturn,
    Any,
    Callable, 
)

import attributes.attributes as attributes
import numpy as np

def ArrayLikeMax(seq: Any = []) -> float | int: return np.max(np.abs(seq))

class differential_equations:
    """
    string representation to diff. eq.
    """
    def __init__(self, model: str = 'ou', spec: str = None) -> None:
        self.model = model
        self.spec = spec

    def __show__(self):
        match self.model.lower():
            case 'diffusion':
                return f'{self.model.capitalize()}: dS(t) = \u03C3dW(t)'
            case 'drift diffusion':
                return f'{self.model.capitalize()}: dS(t) = \u03BCdt + \u03C3dW(t)'
            case 'ou':
                return f'Ornstein-Uhlenbeck ({self.model.upper()}): dS(t) = \u03BCS(t)dt + \u03C3dW(t) (\u03BC < 0)'
            case 'garch':
                return f'{self.model.capitalize()}: S(t) = \u03BC + \u03C3(t)\u03B5(t), \u03C3(t+1)² = \u03C9 + \u03B1S(t)² + \u03B2\u03C3(t)² + f(X)'
            case _:
                if not self.spec: return f'Model "{self.model.capitalize()}" not implemented yet'
                else: return self.spec

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.__show__()
    
    def __repr__(self) -> str:
        return self.__show__()
    
class maps:

    def __init__(self) -> None:
        pass

    @staticmethod
    def std_out(literal: str, c: str, **kwargs) -> ...:
        print(c + literal + '\033[0m', **kwargs)

class operators:

    def __init__(self) -> None:
        pass
        
    @attributes.running_decorator
    def terminate(self, *args, **kwargs):
        # deletes all downloaded .gz files in a folder
        match kwargs:
            case _ if attributes.array(args).__len__() != 0:
                for arg in args:
                    attributes.os.remove(arg)
                return
            case _ if attributes.array(args).__len__() == 0:
                return

    @attributes.lru_cache
    def folder_creator(self, *args, **kwargs):
        for arg in args:
            if not attributes.os.path.exists(attributes.os.getcwd().__add__(f'\\{arg}')):
                print('Creating path {}'.format(attributes.os.getcwd().__add__(f'\\{arg}')))
                attributes.os.makedirs(attributes.os.getcwd().__add__(f'\\{arg}'))
            else:
                try:
                    match kwargs:
                        case kwargs.get('keep', False):
                            self.terminate(*attributes.glob.glob(attributes.os.getcwd().__add__(f'\\{arg}\\*')))
                        case _:
                            continue
                except:
                    continue

class SingletonClass(object):

    _inst = None
    _init_called = False

    def __new__(typ, *args, **kwargs):

        if SingletonClass._inst is None:
            obj = object.__new__(typ, *args, **kwargs)
            SingletonClass._inst = obj

        return SingletonClass._inst

    def __init__(self):

        if SingletonClass._init_called:
            return True
        SingletonClass._init_called = True

class __globals__:

    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        print(f'Updated self.object with {kwargs}')
        self.__dict__.update(kwargs)

class bcolors:

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        try:
            self.__info__()
        except:
            pass
    
    def __repr__(self) -> str:
        return self.__str__()

    def __info__(self) -> ...:
        maps.std_out(f'Avaiable colors: ', c = '')
        for key in self.__colors__.keys():
            maps.std_out(f'\t-{key}', c = self.__colors__.get(key))
    
    def __str__(self):
        s = f'Avaiable colors: \n'
        for key in self.__colors__.keys():
            s += '\t- ' + self.__colors__.get(key) + f'{key}' + '\033[0m\n'
        return s
    
    def colors(self, color):
        return self.__colors__.get(color.upper(), f'Invalid color code: {color}')
        
    @property
    def __colors__(self):
        _ = {
            'HEADER'    : '\033[95m',
            'OKBLUE'    : '\033[94m',
            'OKCYAN'    : '\033[96m',
            'OKGREEN'   : '\033[92m',
            'WARNING'   : '\033[93m',
            'FAIL'      : '\033[91m',
            'ENDC'      : '\033[0m',
            'BOLD'      : '\033[1m',
            'UNDERLINE' : '\033[4m',
            } 
    
        return _
    

colors = bcolors()

def equivalence_wrapper(func):
    def _decorator(self, **kwargs):
        if self.callable:
            maps.std_out(f'SUCCES: class method called', c = colors.colors('okgreen'))
            return func(self, **kwargs)
        else:
            maps.std_out(f'FAIL: class method not callable due to wrong config', c = colors.colors('fail'))
            return None
    return _decorator

