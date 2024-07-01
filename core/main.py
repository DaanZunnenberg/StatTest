import pandas as pd
import numpy as np
from itertools import accumulate
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from tqdm import tqdm 

import nest_asyncio, inspect, functools, itertools, sys, scipy
nest_asyncio.apply()
sys.path.append('../')

from core import models
from attributes import repr
from attributes.dec import deprecated
from attributes.attributes import *
from attributes.errors import prelim_raise_TypeError
from scipy.linalg import (
    LinAlgError,
    LinAlgWarning,
    bandwidth,
    basic,
    blas,
    block_diag,
    sqrtm,
)

from typing import (
    NoReturn,
    Any,
    Callable, 
)

import warnings
warnings.filterwarnings(action = 'ignore')

__VERSION__: str = '1.0.5'

def running_maximum(X): return list(accumulate(np.abs(X), max))
def simple_sequence(X, pct: float = .3) -> float: return pct * (np.max(X.flatten()) - np.min(X.flatten()))
def isiterable(obj, type = list) -> bool: return isinstance(obj, type)

def initializer(func):
    """
    Automatically assigns the parameters.

    >>> class process:
    ...     @initializer
    ...     def __init__(self, cmd, reachable=False, user='root'):
    ...         pass
    >>> p = process('halt', True)
    >>> p.cmd, p.reachable, p.user
    ('halt', True, 'root')
    """
    names, varargs, keywords, defaults = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)

        for name, default in zip(reversed(names), reversed(defaults)):
            if not hasattr(self, name):
                setattr(self, name, default)

        func(self, *args, **kargs)

    return wrapper

class _dec(deprecated):
    def p_values(z_scores: list, two_sides = True, **kwargs):
        pvals = []
        if two_sides:
            for z_score in z_scores:
                pvals.append(1 - 2 * (scipy.stats.norm.cdf(np.abs(z_score), **kwargs) - .5))
        else: raise NotImplementedError('...')
        
        return np.sort(pvals)

    def Benjamini_Hochberg_Yekutilie(p_vals: list, method: str = 'hochberg', q: float = .05):
        N = len(p_vals)
        if method == 'yekutieli':
            assert all(p_vals[i] <= p_vals[i+1] for i in range(N - 1))
            NS: float = np.sum([1 / i for i in range(1, N + 1)])
            for i in range(1, N + 1):
                if p_vals[i - 1] <= i * q / (N * NS): return 1
            return 0
        
        elif method == 'hochberg':
            assert all(p_vals[i] <= p_vals[i+1] for i in range(N - 1))
            for i in range(1, N + 1):
                if p_vals[i - 1] <= i * q / N: return 1
            return 0

        else: print("Please select a method from ['hochberg','yekutieli']")

    def BHY(z_scores, method: str = 'hochberg', q: list = [.01, .05, .1]):
        rejections = {}
        N = len(z_scores)
        for s in q:
            rejection_count = 0
            for run, z_score in tqdm(z_scores.T.iterrows()):
                pvals = p_values(z_score)
                rejection_count += Benjamini_Hochberg_Yekutilie(pvals, method = method, q = s)
            rejections = {**rejections, **{s:rejection_count / N}}
        return rejections

    def BernoulliV(p, n) -> float: return p * (1 - p) / n

class Kernel(object):

    def __init__(self, *, kernel_params: dict) -> NoReturn: self.kernel_params = kernel_params

    def BaseKernel(self) -> Any:
        bandwidth = self.kernel_params.get('bandwidth', NameError)
        SelcKernel: function = lambda x, y : (np.abs(x - y) <= bandwidth)
        BaseKernel: function = lambda x, y : 1 * SelcKernel(x.item(0), y.item(0)) * SelcKernel(x.item(1), y.item(1)) / (bandwidth ** 2)
        return BaseKernel

@deprecated('Class Test deprecated, use [TestV2]')
class Test(object):

    @prelim_raise_TypeError
    def __init__(self, data, kernel_params: dict, time_params: dict, disable=False, reachable=False, user='root', show_object=True) -> NoReturn:
        self.data = data
        self.kernel_params = kernel_params
        self.time_params = time_params
        self.disable = disable
        self.reachable = reachable
        self.use = user
        self.kernel_estimates = {}
        self.time_estimates = {}
        if show_object: self.info()

    @staticmethod
    def __form(shape: tuple, sep: str = '\n', format: str = r'{}') -> str:
        f: str = ''
        for row in range(shape[1]):
            f += shape[0] * format + sep
        return f
    
    @staticmethod
    def __shape(*_, **__) -> tuple: return (len(__.keys), )
    def __info(self, *_, **__) -> Callable: return self.__repr__()

    def __repr__(self) -> str:
        __repr: str = self.__form(shape = (2, self.__dict__.keys().__len__()))
        return __repr.format(*list(itertools.chain(*list(zip(self.__dict__.keys(), self.__dict__.values())))))
    
    def info(self): 
        for key, val in self.__dict__.items():
            if isinstance(val, dict):
                print(key)
                for _, __ in val.items():
                    print('{:>10} :: {}'.format(_, __))
            elif isinstance(val, pd.DataFrame):
                print(key, '::', val.__dict__.get('name', 'Nameless dataframe object at 0x{}'.format(id(val))))
            else: print('{:>10} :: {}'.format(key, val))

    def rename_attribute(self, __old_name):
        _ = self.__dict__.pop(__old_name)

    def class_operators(self, remove: bool = False, **kwargs):
        for name, value in kwargs.items():
            if not remove:
                self.__setattr__(name, value)
            else:
                if hasattr(self, name):
                    _ = self.__dict__.pop(name)
                    repr.maps.std_out(f'{name} attr popped', c = repr.colors.colors('warning'))
                    #self.rename_attribute(name)
            return self
    
    @property
    def duplication(self):  
        P_D: np.matrix = np.matrix([
            [.5, .0, .0, .0],
            [.0, .5, .5, .0],
            [.0, .0, .0, .5],
            ])
        return P_D
    
    @property
    def VECH_VEC(self):
        P_V: np.matrix = np.matrix([
            [1., .0, .0, .0],
            [.0, 1., .0, .0],
            [.0, .0, .0, 1.],
            ])
        return P_V

    @staticmethod
    def integrate_kernel(bandwidth, p: int = 2):
        """
        NOTE: Only works for indicator kernels
        """
        return (2 * bandwidth) / (bandwidth ** (2 * p))
    
    def state_domain_estimator(self) -> Any:
        """
        Class function to estimate the state domain and the variance given the configuration.

        NOTE:
            Function requires:
                1. @Self
                2. kernel_params: dict = {
                        bandwidth   : float
                        n           : int
                        T           : int
                        kernel      : Callable[Callable]
                }
        """

        bandwidth, n, T, _kernel = self.kernel_params.values()
        X, DELTA = self.data, self.data.diff().fillna(0)
        dt = T / n

        # We initialise the kernel
        # kernel = _kernel(**self.kernel_params)
        estimate: list = []

        for x in tqdm(X.values, disable=self.disable):
            sub_estimate: np.matrix = np.eye(2) * 0
            
            sliced = DELTA.iloc[X[(X - x).abs() <= bandwidth].dropna().index]
            norm: float = sliced.__len__()
            for y in np.matrix(sliced):
                sub_estimate += y.T @ y
            
            estimate.append(sub_estimate / (norm * dt))

        variance = []
        for s in estimate:
            variance.append(
                2 * self.duplication @ np.kron(s, s) @ self.duplication.T
            )

        self.kernel_estimates = {**self.kernel_estimates, **{'estimate':estimate, 'variance':variance}}

    def time_domain_estimator(self) -> Any:
        """
        Class function to estimate the time domain and the variance given the configuration.

        NOTE:
            Function requires:
                1. @Self
                2. time_params: dict = {
                        bandwidth   : float
                        n           : int
                        T           : int
                }
        """
        
        bandwidth, n, T = self.time_params.values()
        X, DELTA = self.data, self.data.diff().fillna(0)
        dt = T / n
        est = []
        for row in np.matrix(DELTA):
            est.append(row.T @ row)

        runup: int = np.ceil(bandwidth / dt).astype(int)
        # We initialise the estimate
        estimate: list = [np.matrix([[np.nan,np.nan],[np.nan,np.nan]])] * runup

        for idx in tqdm(range(runup, len(X)), disable = self.disable):
            estimate.append(
                np.sum(est[idx - runup:idx], axis = 0) / bandwidth
            )
            
        variance = []
        for s in estimate:
            variance.append(
                2 * self.duplication @ np.kron(s, s) @ self.duplication.T
            )

        self.time_estimates = {**self.time_estimates, **{'estimate':estimate, 'variance':variance}}

    def gauss(self) -> ...:
        bandwidth_time, _, __ = self.time_params.values()
        bandwidth_state, n, _, __ = self.kernel_params.values()

        x, y = self.data[['process 1']], self.data[['process 2']]
        
        kde_skl_p1 = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(x)
        log_density_p1 = kde_skl_p1.score_samples(x)

        kde_skl_p2 = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(y)
        log_density_p2 = kde_skl_p2.score_samples(y)

        L = self.VECH_VEC
        estimate_time, variance_time = self.time_estimates.values()
        estimate_kernel, variance_kernel = self.kernel_estimates.values()
        state_kernel_integral = self.integrate_kernel(bandwidth=bandwidth_state, p = 2)

        diff, diff_vec, var, std_inv = [], [], [], []
        for t, s in zip(estimate_time, estimate_kernel):
            diff.append(t - s)
            diff_vec.append((L @ (t-s).reshape(4,1)))
            _ = 2 * self.duplication @ np.kron(s, s) @ self.duplication.T
            _ *= (1 + state_kernel_integral) / (n * (bandwidth_state ** 2))
            var.append(_)
            try: std_inv.append(np.linalg.inv(sqrtm(_)))
            except: std_inv.append(np.matrix([[np.nan,np.nan, np.nan],[np.nan,np.nan, np.nan],[np.nan,np.nan, np.nan]]))
        # for i, (t, s) in enumerate(zip(variance_time, variance_kernel)):
        #     _ = (t / (n * bandwidth_time)) + (state_kernel_integral * s / (n * (bandwidth_state ** 2)))  / np.exp(log_density_p1 + log_density_p2)[i]
        #     var.append(_)
        #     try: std_inv.append(np.linalg.inv(sqrtm(_)))
        #     except: std_inv.append(np.matrix([[np.nan,np.nan, np.nan],[np.nan,np.nan, np.nan],[np.nan,np.nan, np.nan]]))
        
        gaussian = []
        for i in range(len(diff_vec)):
            
            try: VI = std_inv[i]
            except: VI = np.matrix([[np.nan,np.nan, np.nan],[np.nan,np.nan, np.nan],[np.nan,np.nan, np.nan]])
            gaussian.append(
                VI @ diff_vec[i]
            )
        self.gaussian = gaussian
        
    @deprecate
    def _gauss(self) -> ...:
        bandwidth_time, _, __ = self.time_params.values()
        bandwidth_state, n, _, __ = self.kernel_params.values()

        x, y = self.data[['process 1']], self.data[['process 2']]
        
        kde_skl_p1 = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(x)
        log_density_p1 = kde_skl_p1.score_samples(x)

        kde_skl_p2 = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(y)
        log_density_p2 = kde_skl_p2.score_samples(y)

        L = self.VECH_VEC
        estimate_time, variance_time = self.time_estimates.values()
        estimate_kernel, variance_kernel = self.kernel_estimates.values()

        time_kernel_integral = self.integrate_kernel(bandwidth=bandwidth_time, p = 1)
        state_kernel_integral = self.integrate_kernel(bandwidth=bandwidth_state, p = 2)

        diff, diff_vec, var, std_inv = [], [], [], []
        for t, s in zip(estimate_time, estimate_kernel):
            diff.append(t - s)
            diff_vec.append((L @ (t-s).reshape(4,1)))
        for i, (t, s) in enumerate(zip(variance_time, variance_kernel)):
            _ = (t / (n * bandwidth_time)) + (state_kernel_integral * s / (n * (bandwidth_state ** 2)))  / np.exp(log_density_p1 + log_density_p2)[i]
            var.append(_)
            try: std_inv.append(np.linalg.inv(sqrtm(_)))
            except: std_inv.append(np.matrix([[np.nan,np.nan, np.nan],[np.nan,np.nan, np.nan],[np.nan,np.nan, np.nan]]))
        
        gaussian = []
        for i in range(len(diff_vec)):
            
            try: VI = std_inv[i]
            except: VI = np.matrix([[np.nan,np.nan, np.nan],[np.nan,np.nan, np.nan],[np.nan,np.nan, np.nan]])
            gaussian.append(
                VI @ diff_vec[i]
            )
        self.gaussian = gaussian
    
    def transform_1D_gauss(self, alpha: float = .95):
        x = np.log(1 / np.log(1/alpha))

        if not hasattr(self, 'gaussian'): self.gauss()
        n = len(self.gaussian)
        an = [np.sqrt(2 * np.log(z)) for z in range(1, n + 1)]
        bn = [np.sqrt(2 * np.log(z)) - (np.log(np.pi * np.log(z)) / (2 * np.sqrt(2 * np.log(z)))) for z in range(1, n + 1)]
        bound = [np.nan]
        for i in range(1,n):
            bound.append((x / an[i]) + bn[i])

        final = []
        for g in self.gaussian:
            if np.isnan(g).any():
                final.append(0)
            else:
                final.append(np.sum(g) / 3)
        
        return bound, final

class TestV2(object):
    
    @prelim_raise_TypeError
    def __init__(self, data, kernel_params: dict, time_params: dict, disable=False, reachable=False, user='root', show_object=True) -> NoReturn:
        self.data = data
        self.kernel_params = kernel_params
        self.time_params = time_params
        self.disable = disable
        self.reachable = reachable
        self.use = user
        self.kernel_estimates = {}
        self.time_estimates = {}
        if show_object: self.info()

    @staticmethod
    def __form(shape: tuple, sep: str = '\n', format: str = r'{}') -> str:
        f: str = ''
        for row in range(shape[1]):
            f += shape[0] * format + sep
        return f
    
    @staticmethod
    def __shape(*_, **__) -> tuple: return (len(__.keys), )
    def __info(self, *_, **__) -> Callable: return self.__repr__()

    def __repr__(self) -> str:
        __repr: str = self.__form(shape = (2, self.__dict__.keys().__len__()))
        return __repr.format(*list(itertools.chain(*list(zip(self.__dict__.keys(), self.__dict__.values())))))
    
    def info(self): 
        for key, val in self.__dict__.items():
            if isinstance(val, dict):
                print(key)
                for _, __ in val.items():
                    print('{:>10} :: {}'.format(_, __))
            elif isinstance(val, pd.DataFrame):
                print(key, '::', val.__dict__.get('name', 'Nameless dataframe object at 0x{}'.format(id(val))))
            else: print('{:>10} :: {}'.format(key, val))

    def rename_attribute(self, __old_name):
        _ = self.__dict__.pop(__old_name)

    def class_operators(self, remove: bool = False, **kwargs):
        for name, value in kwargs.items():
            if not remove:
                self.__setattr__(name, value)
            else:
                if hasattr(self, name):
                    _ = self.__dict__.pop(name)
                    repr.maps.std_out(f'{name} attr popped', c = repr.colors.colors('warning'))
                    #self.rename_attribute(name)
            return self
    
    @property
    def duplication(self):  
        P_D: np.matrix = np.matrix([
            [.5, .0, .0, .0],
            [.0, .5, .5, .0],
            [.0, .0, .0, .5],
            ])
        return P_D
    
    @property
    def VECH_VEC(self):
        P_V: np.matrix = np.matrix([
            [1., .0, .0, .0],
            [.0, 1., .0, .0],
            [.0, .0, .0, 1.],
            ])
        return P_V
    
    @property
    def projection_mat(self):
        P_P: np.matrix = np.matrix([
            [1., .0, .0, .0],
            [.0, .5, .5, .0],
            [.0, .0, .0, 1.],
            ])
        return P_P
    
    def emp_dist(self, dist: bool):
        if not dist:
            return np.ones(self.kernel_params['n'])
        else:
            x, y = self.data[['process 1']], self.data[['process 2']]
            kde_skl_p1 = KernelDensity(kernel='gaussian', bandwidth = 2 * self.kernel_params['bandwidth']).fit(x)
            log_density_p1 = kde_skl_p1.score_samples(x)

            kde_skl_p2 = KernelDensity(kernel='gaussian', bandwidth = 2 * self.kernel_params['bandwidth']).fit(y)
            log_density_p2 = kde_skl_p2.score_samples(y)
            return np.exp(log_density_p1 + log_density_p2)
    
    @staticmethod
    def tau_scalar(tau) -> float: return (tau * (1 + np.exp(tau)) / (np.exp(tau) - 1))

    def univ_var(self, A: ...):
        final: list = []
        P_P: np.matrix = self.projection_mat
        if isiterable(A):
            for cov_mat in A: final.append(P_P @ np.kron(cov_mat, cov_mat) @ P_P.T)
        else:
            final = P_P @ np.kron(A, A) @ P_P.T
        return final
    
    def time_domain_smoother(self, lamb: float = .94, allow_true_var: bool = False, true_var: Any = None) -> ...:
        """
        Class function to estimate the time domain and the variance given the configuration.

        NOTE:
            Function requires:
                1. @Self
                2. time_params: dict = {
                        bandwidth   : float
                        n           : int
                        T           : int
                }
        """
        
        bandwidth, n, T = self.time_params.values()
        X = self.data
        DELTA = self.data.diff().fillna(0)
        dt = T / n
        est = []
        for row in np.matrix(DELTA):
            est.append(row.T @ row)

        runup: int = np.ceil(bandwidth / dt).astype(int)
        tau = runup * (1 - lamb)
        # We initialise the estimate
        estimate: list = [np.matrix([[np.nan,np.nan],[np.nan,np.nan]])] * runup
        variance: list = [np.matrix([[np.nan,np.nan],[np.nan,np.nan]])] * runup
        truepvar: list = [np.matrix([[np.nan,np.nan],[np.nan,np.nan]])] * runup
        factpvar: list = [np.matrix([[np.nan,np.nan],[np.nan,np.nan]])] * runup

        if allow_true_var: variance = true_var
        for idx in tqdm(range(runup, len(X)), disable = self.disable):
            _est = np.zeros_like(est[idx])
            for sidx in range(runup):
                _est += (lamb ** (sidx)) * est[idx - sidx]
            _est *= ((1 - lamb) / (1 - (lamb ** runup))) / dt
            estimate.append(_est)
            factpvar.append(self.tau_scalar(tau))
            if allow_true_var: 
                variance.append(self.tau_scalar(tau) * self.univ_var(true_var[idx]))
                truepvar.append(self.univ_var(true_var[idx]))
            else: 
                variance.append(self.tau_scalar(tau) * self.univ_var(_est))
                truepvar.append(self.univ_var(_est))

        self.time_factpvar = factpvar
        self.true_est_var = truepvar
        self.time_estimates = {**self.time_estimates, **{'estimate':estimate, 'variance':variance}}

    def state_domain_smoother(self, dist = None) -> ...:
        """
        Class function to estimate the state domain and the variance given the configuration.

        NOTE:
            Function requires:
                1. @Self
                2. kernel_params: dict = {
                        bandwidth   : float
                        n           : int
                        T           : int
                        kernel      : Callable[Callable]
                }
        """
        try: self.true_est_var
        except:
            print('Please call Self@time_domain_smoother(...) for using the state domain smooter.')
            sys.exit(1)
        if not isinstance(dist, list): dist = self.emp_dist(dist)
        self.kernel_dist = dist
        X = self.data
        bandwidth, n, T, _kernel = self.kernel_params.values()
        b, *_ = self.time_params.values()
        X, DELTA = self.data, self.data.diff().fillna(0)
        dt = T / n

        # We initialise the kernel
        # kernel = _kernel(**self.kernel_params)
        estimate: list = []
        variance: list = []
        factpvar: list = []

        for idx, x in enumerate(tqdm(X.values, disable=self.disable)):
            sub_estimate: np.matrix = np.eye(2) * 0
            
            sliced = DELTA.iloc[X[(X - x).abs() <= bandwidth].dropna().index]
            norm: float = sliced.__len__()
            for y in np.matrix(sliced):
                sub_estimate += y.T @ y
            
            factpvar.append(bandwidth ** 2 / dist[idx])
            estimate.append(sub_estimate / (norm * dt))
            variance.append(self.true_est_var[idx] / dist[idx])

        self.state_factpvar = factpvar
        self.kernel_estimates = {**self.kernel_estimates, **{'estimate':estimate, 'variance':variance}}

    @staticmethod
    def integrate_kernel(bandwidth, p: int = 2):
        """
        NOTE: Only works for indicator kernels
        """
        return (2 * bandwidth) / (bandwidth ** (2 * p))
    
    def gauss(self) -> ...:
        """
        Generates a Gaussian process from the normally distributed differences
        """
        b, n, T, _kernel = self.kernel_params.values()
        x, y = self.data[['process 1']], self.data[['process 2']]
        bandwidth_state = self.kernel_params['bandwidth']
        dist = self.kernel_dist
        
        state_norm = self.state_factpvar
        time_norm  = self.time_factpvar
        truepvar = self.true_est_var

        L = self.VECH_VEC

        state_est, state_var = self.time_estimates['estimate'], self.time_estimates['variance']
        time_est, time_var = self.kernel_estimates['estimate'], self.kernel_estimates['variance']
        state_kernel_integral = self.integrate_kernel(bandwidth=bandwidth_state, p = 2)

        diff, diff_vec, var, std_inv = [], [], [], []
        for t, s, v, sn, tn in zip(time_est, state_est, truepvar, state_norm, time_norm):
            diff.append(t - s)
            diff_vec.append((L @ (t-s).reshape(4,1)))

            _ = (2 * sn + tn) * v
            var.append(_)
            try: std_inv.append(np.linalg.inv(sqrtm(_)))
            except: std_inv.append(np.matrix([[np.nan,np.nan, np.nan],[np.nan,np.nan, np.nan],[np.nan,np.nan, np.nan]]))

        gaussian = []
        for i in range(len(diff_vec)):
            
            try: VI = std_inv[i]
            except: VI = np.matrix([[np.nan,np.nan, np.nan],[np.nan,np.nan, np.nan],[np.nan,np.nan, np.nan]])
            gaussian.append(
                np.sqrt(n * b ** 2) * VI @ diff_vec[i]
            )
            
        self.gaussian = gaussian

    def transform_1D_gauss(self, alpha: float = .95):
        x = np.log(1 / np.log(1/alpha))

        if not hasattr(self, 'gaussian'): self.gauss()
        n = len(self.gaussian)
        an = [np.sqrt(2 * np.log(z)) for z in range(1, n + 1)]
        bn = [np.sqrt(2 * np.log(z)) - (np.log(np.pi * np.log(z)) / (2 * np.sqrt(2 * np.log(z)))) for z in range(1, n + 1)]
        bound = [np.nan]
        for i in range(1,n):
            bound.append((x / an[i]) + bn[i])

        final = []
        for g in self.gaussian:
            if np.isnan(g).any():
                final.append(0)
            else:
                final.append(np.sum(g) / 3)
        
        return bound, final

class simulate(object):
    
    def __init__(self, 
                 number_of_runs, 
                 config: dict = {}, 
                 est_config: dict = None,
                 run_on_call: bool = False,
                 ProcessGenerator: Callable = models.BivariateOUProcess) -> ...:
        self.number_of_runs = number_of_runs
        try: est_config.pop('data', False)
        except: 
            est_config.pop('data', False)
            print("Popped 'data' from config")
        self.config = config
        self.est_config = est_config
        self.ProcessGenerator = ProcessGenerator
        self.results = {}
        if True:self.info()
    
    def generate_config(self, data, T, n):
        """
        Standard configuration with a correction from discussion in Fan-Fan-Lv (https://arxiv.org/pdf/math/07011070)
        """
        if not self.est_config: 
            if T >= 200:
                config: dict = {
                    'data'  : data,
                    'kernel_params' : {
                        'bandwidth' : 4.5 / ((n ** (1 / 5))),
                        'n'         : n,
                        'T'         : T,
                        'kernel'    : Kernel.BaseKernel
                    },
                    'time_params'   : {
                        'bandwidth' : 200 * T / n,
                        'n'         : n,
                        'T'         : T
                    },
                }
            elif T >= 100:
                config: dict = {
                    'data'  : data,
                    'kernel_params' : {
                        'bandwidth' : 3 / ((n ** (1 / 5))),
                        'n'         : n,
                        'T'         : T,
                        'kernel'    : Kernel.BaseKernel
                    },
                    'time_params'   : {
                        'bandwidth' : 100 * T / n,
                        'n'         : n,
                        'T'         : T
                    },
                }
            else:
                config: dict = {
                    'data'  : data,
                    'kernel_params' : {
                        'bandwidth' : 2 / ((n ** (1 / 5))),
                        'n'         : n,
                        'T'         : T,
                        'kernel'    : Kernel.BaseKernel
                    },
                    'time_params'   : {
                        'bandwidth' : 100 * T / n,
                        'n'         : n,
                        'T'         : T
                    },
                }
        else: 
            config: dict = {**{'data'  : data}, **self.est_config}
        return config

    def info(self): 
        for key, val in self.__dict__.items():
            if isinstance(val, dict):
                print(key)
                for _, __ in val.items():
                    print('{:>10} :: {}'.format(_, __))
            elif isinstance(val, pd.DataFrame):
                print(key, '::', val.__dict__.get('name', 'Nameless dataframe object at 0x{}'.format(id(val))))
            else: print('{:>10} :: {}'.format(key, val))
    
    def run(self, seed: list = [], state_kwargs: dict = {}, time_kwargs: dict = {}, **Test_kwargs):
        if len(seed) > 0: print('Simulation started on custom seed.')
        process = self.ProcessGenerator(**self.config)
        for run in tqdm(range(self.number_of_runs)):
            try: np.random.seed(seed[run])
            except: print('No seed set', end = '\r')
            process.simulate()
            X, T, n = process.config()
            test = TestV2(**{**self.generate_config(X, T, n), **Test_kwargs})

            test.time_domain_smoother(**time_kwargs)
            test.state_domain_smoother(**state_kwargs)
            test.gauss()

            bound, scalar_gauss = test.transform_1D_gauss()

            self.results = {**self.results, 
                            **{ f'run_{run}'   : {
                                    f'gauss'   : scalar_gauss,
                                    f'process' : X,
                            }}}
        self.results = {**self.results, **{'bound':bound}}

    def bound(self, alpha):
        x = -np.log(np.log(1/alpha))

        n = len(self.results[f'run_{0}']['gauss'])
        an = [np.sqrt(2 * np.log(z)) for z in range(1, n + 1)]
        bn = [np.sqrt(2 * np.log(z)) - ((np.log(np.log(z)) + np.log(np.pi)) / (2 * np.sqrt(2 * np.log(z)))) for z in range(1, n + 1)]
        bound = [np.nan]
        for i in range(1,n):
            bound.append((x / an[i]) + bn[i])
        return bound
    
    def summary(self, alphas:  list = [0.95]):
        bounds = []
        for alpha in alphas: bounds.append(self.bound(alpha))
        for alpha, bound in zip(alphas, bounds):
            print('Number of runs: {}, rejection rate: {}% (\u03B1 = {}%)'.format(
                self.number_of_runs, 
                np.round(100 * np.sum([1 * (running_maximum(self.results[f'run_{run}']['gauss'])[-1] > bound[-1]) for run in range(self.number_of_runs)]) / self.number_of_runs, 2),
                np.round(100 * alpha, 0)
            ))

    def bound(self, alpha):
        n = len(self.results[f'run_{0}']['gauss'])
        x = -np.log(np.log(1/alpha))
        an = [np.sqrt(2 * np.log(z)) for z in range(1, n + 1)]
        bn = [np.sqrt(2 * np.log(z)) - ((np.log(np.log(z)) + np.log(np.pi)) / (2 * np.sqrt(2 * np.log(z)))) for z in range(1, n + 1)]
        bound = [np.nan]
        for i in range(1,n):
            bound.append((x / an[i]) + bn[i])
        return bound
    
    
    def plot_results(self, **kwargs):
        bound1, bound5, bound10, bound50 = self.bound(.99), self.bound(.95), self.bound(.90), self.bound(.5)
        fig, axs = plt.subplots(1, 2, figsize = (18,8))
        for run in range(self.number_of_runs):
            rr = self.results[f'run_{run}']
            if run == 0: axs[0].plot(running_maximum(rr['gauss']), c = 'C0', lw = 1, alpha = .5, label = r"Emperical 95% running maximum")
            else: axs[0].plot(running_maximum(rr['gauss']), c = 'C0', lw = 1, alpha = .5)
            
        axs[0].plot(bound1, color = 'grey', ls = '-', lw = 1, label = r"Theoretical running maximum (1%)")
        axs[0].plot(bound5, color = 'grey', ls = 'dashed', lw = 1, label = r"Theoretical running maximum (5%)")
        axs[0].plot(bound10, color = 'grey', ls = 'dotted', lw = 1, label = r"Theoretical running maximum (10%)")
        axs[0].plot(bound50, color = 'red', ls = 'dashed', lw = 1, label = r"Theoretical running maximum (50%)")
        axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
        axs[1].hist([running_maximum(self.results[f'run_{run}']['gauss'])[-1] for run in range(self.number_of_runs)], **kwargs)
        plt.show()

class Container(object):

    def __init__(self, **kwargs): 
        for _, __ in kwargs.items(): self.__setattr__(_, __)
    
    @property
    def n(self): return self.X.__len__()

    def check(self):
        for name in ['T', 'X']:
            assert self.__dict__[name]

    def optimal_bandwidth(self, type = 'time'):
        match type:
            case 'time':
                return 5 * self.T * np.sum(np.std(self.X)) / ((self.n ** .5) * np.log(self.n))
            case 'state':
                return 5 * np.sum(np.std(X)) / ((self.n ** (1 / 5)) * np.log(self.n))
            case _:
                raise ValueError("""Please select type from ['time','state'] (default is 'time')""")

    @property
    def default_config(self):
        self.check()
        config: dict = {
            'data'  : self.X,
            'kernel_params' : {
                'bandwidth' : self.optimal_bandwidth(type = 'state'),
                'n'         : n,
                'T'         : self.T,
                'kernel'    : Kernel.BaseKernel
            },
            'time_params'   : {
                'bandwidth' : self.optimal_bandwidth(type = 'time'),
                'n'         : self.n,
                'T'         : self.T
            },
        }
        return config

class graph(Test):

    def __init__(self, cl): 
        for _, __ in cl.__dict__.items(): self.__setattr__(_, __)

    def bound(self, alpha):
        x = -np.log(np.log(1/alpha))

        if not hasattr(self, 'gaussian'): self.gauss()
        n = len(self.gaussian)
        an = [np.sqrt(2 * np.log(z)) for z in range(1, n + 1)]
        bn = [np.sqrt(2 * np.log(z)) - ((np.log(np.log(z)) + np.log(np.pi)) / (2 * np.sqrt(2 * np.log(z)))) for z in range(1, n + 1)]
        bound = [np.nan]
        for i in range(1,n):
            bound.append((x / an[i]) + bn[i])
        return bound
    
    def plot_running_maximum(self):
        _, scalar_gauss = self.transform_1D_gauss()
        bound1, bound5, bound10, bound50 = self.bound(.99), self.bound(.95), self.bound(.90), self.bound(.5)
        fig, axs = plt.subplots(2,1, figsize = (12,8))
        axs[0].plot(running_maximum(scalar_gauss), color = 'black', lw = 1, label = r"Theoretical 95% running maximum")
        axs[0].plot(bound1, color = 'grey', ls = '-', lw = 1, label = r"Emperical running maximum (1%)")
        axs[0].plot(bound5, color = 'grey', ls = 'dashed', lw = 1, label = r"Emperical running maximum (5%)")
        axs[0].plot(bound10, color = 'grey', ls = 'dotted', lw = 1, label = r"Emperical running maximum (10%)")
        axs[0].plot(bound50, color = 'red', ls = 'dashed', lw = 1, label = r"Emperical running maximum (50%)")
        axs[0].plot(scalar_gauss, color = 'C0', lw = 1, label = r"Standardised Gaussian Process")
        axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=3)
        axs[0].set_title('Running maximum')
        axs[0].grid(True)

        t, x, y = np.linspace(0, self.kernel_params.get('T'), self.kernel_params.get('n')), self.data['process 1'], self.data['process 2']
        axs[1].plot(t, x, label='Process 1')
        axs[1].plot(t, y, label='Process 2')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Value')
        axs[1].set_title('Processes')
        axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2)
        axs[1].grid(True)
        fig.tight_layout()
        fig.show()

    def plot_estimates(self):
        est_time = self.time_estimates['estimate']
        var_time = self.time_estimates['variance']

        est_state = self.kernel_estimates['estimate']
        var_state = self.kernel_estimates['variance']

        fig, axs = plt.subplots(2,1, figsize = (12,6))
        axs[0].plot([item.item(0) ** .5 for item in est_state], c = 'C1', ls = '-', lw = 1, label = 'state estimate')
        # plt.plot([item.item(1) ** .5 for item in est_state], c = 'C1', ls = 'dashed', lw = 1)
        axs[1].plot([item.item(3) ** .5 for item in est_state], c = 'C1', ls = '-', lw = 1, label = 'state estimate')
        axs[0].plot(ou_config['sigma1'], c = 'C2', ls = '-', lw = 1, label = 'true volatility')
        axs[0].plot([item.item(0) ** .5 for item in est_time], c = 'C0', ls = '-', lw = 1, label = 'time estimate')
        # plt.plot([item.item(1) ** .5 for item in est_time], c = 'C0', ls = 'dashed', lw = 1)
        axs[1].plot([item.item(3) ** .5 for item in est_time], c = 'C0', ls = '-', lw = 1, label = 'time estimate')
        axs[1].plot(ou_config['sigma2'], c = 'C2', ls = '-', lw = 1, label = 'true volatility')
        axs[0].set_title('Process 1')
        axs[1].set_title('Process 2')
        for ax in axs: ax.legend()
        fig.tight_layout()
        plt.show()

if __name__ == "__main__":
    # simulator = simulate(number_of_runs=100, config = {'T':250, 'dt':0.1})
    # simulator.run()

    ou_config = {'sigma1': 0.5, 'sigma2': 0.8, 'mu1': 0, 'mu2': 0, 'theta1': .02, 'theta2': .02}
    ou_process = models.BivariateOUProcess(T = 1000, dt = 0.1, **ou_config)
    ou_process.simulate()
    X, T, n = ou_process.config()
    dt = T / n
    ou_process.plot()
    
    config: dict = {
        'data'  : X,
        'kernel_params' : {
            'bandwidth' : 5 * np.sum(np.std(X)) / ((n ** (1 / 5)) * np.log(n)),
            'n'         : n,
            'T'         : T,
            'kernel'    : Kernel.BaseKernel
        },
        'time_params'   : {
            'bandwidth' : 100 * T / n,
            'n'         : n,
            'T'         : T
        },
    }

    test = TestV2(**config)
    test.time_domain_smoother()
    
    # est_time = test.time_estimates['estimate']
    # plt.plot([item.item(0) ** .5 for item in est_time])
    # plt.axhline(ou_config['sigma1'])
    # plt.plot([item.item(3) ** .5 for item in est_time])
    # plt.axhline(ou_config['sigma2'], c='C1')
    # # plt.plot(X.diff().rolling(1000).std())
    # plt.show()
    

    test.state_domain_smoother()
    test.gauss()

    bound, scalar_gauss = test.transform_1D_gauss()

    plt.plot(running_maximum(scalar_gauss))
    plt.plot(bound)
    plt.show()
    
    est_time = test.time_estimates['estimate']
    var_time = test.time_estimates['variance']

    est_state = test.kernel_estimates['estimate']
    var_state = test.kernel_estimates['variance']

    fig, axs = plt.subplots(2,1, figsize = (8,6))
    axs[0].plot([item.item(0) ** .5 for item in est_state], c = 'C1', ls = '-', lw = 1, label = 'state estimate')
    # plt.plot([item.item(1) ** .5 for item in est_state], c = 'C1', ls = 'dashed', lw = 1)
    axs[1].plot([item.item(3) ** .5 for item in est_state], c = 'C1', ls = '-', lw = 1, label = 'state estimate')
    axs[0].axhline(ou_config['sigma1'], c = 'black', ls = '-', lw = 1, label = 'true volatility')
    axs[0].plot([item.item(0) ** .5 for item in est_time], c = 'C0', ls = '-', lw = 1, label = 'time estimate')
    # plt.plot([item.item(1) ** .5 for item in est_time], c = 'C0', ls = 'dashed', lw = 1)
    axs[1].plot([item.item(3) ** .5 for item in est_time], c = 'C0', ls = '-', lw = 1, label = 'time estimate')
    axs[1].axhline(ou_config['sigma2'], c = 'black', ls = '-', lw = 1, label = 'true volatility')
    axs[0].set_title('Process 1')
    axs[1].set_title('Process 2')
    for ax in axs: ax.legend()
    fig.tight_layout()
    plt.show()

    plt.plot([item.item(0) ** .5 for item in var_time])
    plt.plot([item.item(1) ** .5 for item in var_time])
    plt.plot([item.item(2) ** .5 for item in var_time])
    plt.plot([item.item(0) ** .5 for item in var_state])
    plt.plot([item.item(1) ** .5 for item in var_state])
    plt.plot([item.item(2) ** .5 for item in var_state])
    plt.show()

    difference = np.array(est_time) - np.array(est_state)

    fig, axs = plt.subplots(2,1, figsize = (8,8))
    axs[0].plot([item.item(0) for item in difference])
    axs[0].plot([item.item(2) for item in difference])
    axs[0].plot([item.item(3) for item in difference])

    axs[1].plot(running_maximum(scalar_gauss))
    axs[1].plot(bound)
    axs[1].plot(scalar_gauss)
    plt.show()

    # bw_config = {'T':1000, 'dt':0.1,'mu1': 2, 'mu2': 1, 'rho': -.05, 'sigma1': 0.1, 'sigma2': 0.1}
    # bw_process = models.BivariateCorrelatedDiffusion(**bw_config)
    # bw_process.simulate()
    # X, T, n = bw_process.config()
    # bw_process.plot()

    # config: dict = {
    #     'data'  : X,
    #     'kernel_params' : {
    #         'bandwidth' : 5 * np.sum(np.std(X)) / ((n ** (1 / 5)) * np.log(n)),
    #         'n'         : n,
    #         'T'         : T,
    #         'kernel'    : Kernel.BaseKernel
    #     },
    #     'time_params'   : {
    #         'bandwidth' : 5 * np.sum(np.std(X)) / ((n ** .5) * np.log(n)),
    #         'n'         : n,
    #         'T'         : T
    #     },
    # }

    # test = Test(**config)

    # test.state_domain_estimator()
    # test.time_domain_estimator()
    # test.gauss()

    # bound, scalar_gauss = test.transform_1D_gauss()

    # plt.plot(running_maximum(scalar_gauss))
    # plt.plot(bound)
    # plt.show()
    
    # est_time = test.time_estimates['estimate']
    # var_time = test.time_estimates['variance']

    # est_state = test.kernel_estimates['estimate']
    # var_state = test.kernel_estimates['variance']

    # fig, axs = plt.subplots(2,1, figsize = (8,6))
    # axs[0].plot([item.item(0) ** .5 for item in est_state], c = 'C1', ls = '-', lw = 1, label = 'state estimate')
    # # plt.plot([item.item(1) ** .5 for item in est_state], c = 'C1', ls = 'dashed', lw = 1)
    # axs[1].plot([item.item(3) ** .5 for item in est_state], c = 'C1', ls = '-', lw = 1, label = 'state estimate')
    # axs[0].plot(bw_config['sigma1'] * X['process 1'], c = 'C2', ls = '-', lw = 1, label = 'true volatility')
    # axs[0].plot([item.item(0) ** .5 for item in est_time], c = 'C0', ls = '-', lw = 1, label = 'time estimate')
    # # plt.plot([item.item(1) ** .5 for item in est_time], c = 'C0', ls = 'dashed', lw = 1)
    # axs[1].plot([item.item(3) ** .5 for item in est_time], c = 'C0', ls = '-', lw = 1, label = 'time estimate')
    # axs[1].plot(bw_config['sigma2'] * X['process 2'], c = 'C2', ls = '-', lw = 1, label = 'true volatility')
    # axs[0].set_title('Process 1')
    # axs[1].set_title('Process 2')
    # for ax in axs: ax.legend()
    # fig.tight_layout()
    # plt.show()

    # plt.plot([item.item(0) ** .5 for item in var_time])
    # plt.plot([item.item(1) ** .5 for item in var_time])
    # plt.plot([item.item(2) ** .5 for item in var_time])
    # plt.plot([item.item(0) ** .5 for item in var_state])
    # plt.plot([item.item(1) ** .5 for item in var_state])
    # plt.plot([item.item(2) ** .5 for item in var_state])
    # plt.show()

    # difference = np.array(est_time) - np.array(est_state)

    # fig, axs = plt.subplots(2,1, figsize = (8,8))
    # axs[0].plot([item.item(0) for item in difference])
    # axs[0].plot([item.item(2) for item in difference])
    # axs[0].plot([item.item(3) for item in difference])

    # axs[1].plot(running_maximum(scalar_gauss))
    # axs[1].plot(bound)
    # axs[1].plot(scalar_gauss)
    # plt.show()
