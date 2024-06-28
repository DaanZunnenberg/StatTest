import typing, os

__VERSION__ = '1.1.1'
os.system('cls')
print(__name__, 'V.{}'.format(__VERSION__))

__all__: list = ['UtilContainer']

PURP: str = '\033[95m'
WHIT: str = '\033[0m'
_KT = typing.TypeVar("_KT")
_VT = typing.TypeVar("_VT")

class UtilContainer(object):
    def __init__ (self, **kwargs) -> None | typing.NoReturn: self.__utils__(**kwargs)
    def __utils__(self, **kwargs) -> None | typing.NoReturn: 
        for _, __ in kwargs.items(): self.__setattr__(_, __)
    
    def __standard_utils__(self)  -> None | typing.NoReturn:
        self.res    : typing.TypeAlias = list(float)
        self.x      : typing.TypeAlias = list(float)
        self.ll     : typing.TypeAlias = list(float)
        self.innov  : typing.TypeAlias = list(float)
        self.params : typing.TypeAlias = list(float)
        
    def __str__(self) -> object: return self.__repr__()
    def __repr__(self):
        _ret: str = ''
        for _, __ in self.__dict__.items():
            _ret += f'@Self[{type(__)} @ {PURP}0x{id(__)}{WHIT}]: ({_}, {__})\n'
        return _ret
    
if __name__ == '__main__':
    """
    Example usuage
    """
    import scipy
    fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2
    cons = (
        {'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},
        {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
        {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2}
        )
    bnds = ((0, None), (0, None))
    res = scipy.optimize.minimize(fun, (2, 0), 
                method      =   'SLSQP', 
                bounds      =   bnds,
               constraints  =   cons,
               options      =   {
                   'gtol': 1e-6, 
                   'disp': True
                   })

    config: dict | typing.MutableMapping[_KT, _VT] = {key:res[key] for key in res.__dir__()}

    cUtilContainer: UtilContainer = UtilContainer(**config)
    print(cUtilContainer, sep = '', file = None)
