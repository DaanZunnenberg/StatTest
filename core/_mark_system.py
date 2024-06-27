import sys
from typing import Any, Literal

__doc__: ... = 'self initialised'

@staticmethod
def run_parent_packages(*, method = sys.path.insert) -> None: method(0, '..')
def doc_init(__func__):
    def wrapper(*args, **kwargs):
        __func__.__doc__ = __doc__
        print(__func__.__doc__)
        return __func__
    return wrapper

@doc_init
class reference_parent_dir: 
    @property 
    def run_parent_packages(): run_parent_packages()
    def __call__(self, *args: Any, **kwds: Any) -> ...: self.run_parent_packages
    def __str__(self) -> ...: pass

