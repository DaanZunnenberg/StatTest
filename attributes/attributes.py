# -*- coding: utf-8 -*-
import sys, os, glob
import time
from numpy import *
from functools import lru_cache
import matplotlib.pyplot as plt
from typing import Any
import pandas as pd
import numpy as np

def pandas_filter(func, arguments: dict = dict()):
    def func_wrapper(*args, **kwargs):
        value = func(*args, **kwargs)
        if len(arguments.keys()) != 0:
            value = value[value[arguments.get('column')] == arguments.get('value')]
        return value
    return func_wrapper

def psuedo_private(func, hide = False):
    def func_wrapper(*args, **kwargs):
        if hide:
            sys.stdout = open(os.devnull, 'w')
            value = func(*args, **kwargs)
            sys.stdout = sys.__stdout__
        else:
            value = func(*args, **kwargs)
        return value

    return func_wrapper

def private_timer(func):
    def wrapper(*args, **kwargs):
        _: object = time.time()
        __func__ = func(*args, **kwargs)
        print(f'Terminated in {round(1000 * (time.time() - _))} milliseconds')
        return __func__
    return wrapper

def running_decorator(func):
    def wrapper(*args, **kwargs):
        _: object = time.time()
        print(f'Running {func.__name__}')
        __func__ = func(*args, **kwargs)
        print(f'Completed in {round(time.time() - _, 2)} seconds')
        return __func__
    return wrapper

def template_decorator(func):
    def wrapper(*args, **kwargs):
        _: object = time.time()
        print(f'Running {func.__name__}')
        func(*args, **kwargs)
        print(f'Completed in {round(time.time() - _, 2)} seconds')
    return wrapper

def ignore_unhashable(func): 
    import functools
    uncached = func.__wrapped__
    attributes = functools.WRAPPER_ASSIGNMENTS + ('cache_info', 'cache_clear')
    @functools.wraps(func, assigned=attributes) 
    def wrapper(*args, **kwargs): 
        try: 
            return func(*args, **kwargs) 
        except TypeError as error: 
            if 'unhashable type' in str(error): 
                return uncached(*args, **kwargs) 
            raise 
    wrapper.__uncached__ = uncached
    return wrapper
