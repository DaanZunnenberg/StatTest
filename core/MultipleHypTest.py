import pandas as pd
import numpy as np
import scipy 
from typing import NoReturn
from tqdm import tqdm
from functools import cache
import sys

sys.path.append('../')
from attributes.attributes import ignore_unhashable

class MultipleHypTest(object):

    def __init__(self, z_scores, two_sides: bool = True, remove_zero_rows: bool = True, run_on_call: bool = False, **kwargs) -> NoReturn:
        if remove_zero_rows: z_scores = z_scores[z_scores != 0].dropna()
        self.z_scores = z_scores
        self.two_sides = two_sides
        self.N = len(z_scores)
        self.S = len(z_scores.T)
        self.rejections = {}
        if kwargs.get('name', False): 
            name = kwargs.pop('name')
            print(f' {name} '.center(70, '-'))
        if run_on_call: self.run(**kwargs)
            

    @ignore_unhashable
    @cache
    def p_values(self, z_scores, **kwargs):
        pvals = []
        if self.two_sides:
            for z_score in z_scores:
                pvals.append(1 - 2 * (scipy.stats.norm.cdf(np.abs(z_score), **kwargs) - .5))
        else: raise NotImplementedError('...')
        
        self.pvals = np.sort(pvals)
        return np.sort(pvals)

    @ignore_unhashable
    @cache
    def Benjamini_Hochberg_Yekutilie(self, p_vals, method: str = 'hochberg', q: float = .05):

        if method == 'yekutieli':
            assert all(p_vals[i] <= p_vals[i+1] for i in range(self.N - 1))
            NS: float = np.sum([1 / i for i in range(1, self.N + 1)])
            for i in range(1, self.N + 1):
                if p_vals[i - 1] <= i * q / (self.N * NS): return 1
            return 0
        
        elif method == 'hochberg':
            assert all(p_vals[i] <= p_vals[i+1] for i in range(self.N - 1))
            for i in range(1, self.N + 1):
                if p_vals[i - 1] <= i * q / self.N: return 1
            return 0

        else: print("Please select a method from ['hochberg','yekutieli']")

    @ignore_unhashable
    @cache
    def BHY(self, method: str = 'hochberg', q: list = [.1, .05, .01], ret: bool = False):
        rejections = {}
        for s in q:
            rejection_count = 0
            for run, z_score in tqdm(self.z_scores.T.iterrows()):
                pvals = self.p_values(z_score)
                rejection_count += self.Benjamini_Hochberg_Yekutilie(pvals, method = method, q = s)
            rejections = {**rejections, **{s:rejection_count / self.S}}
            self.rejections = {**self.rejections, **{method:rejections}}
        if ret: return rejections
    
    def run(self, q: list = [.1, .05, .01]) -> NoReturn:
        for method in ['hochberg','yekutieli']:
            print(f'MultipleHypTest@Self.run: running [{method}]...')
            self.BHY(method = method, q = q)
            self.print(method)
        print('Finished'.center(70, '-'))

    def print(self, method: str):
        print(f'\n{method.title()}:')
        for key, val in self.rejections.get(method, {'':''}).items():
            if key != '':
                print(
                    'Rejection rate: {}% (\u03B1 = {}%)'.format(self.lrjust(val), self.lrjust(key)))
            else: pass
        print('\n', end = '')

    @staticmethod
    def BernoulliV(p, n) -> float: return p * (1 - p) / n

    @staticmethod
    def lrjust(flt: float):
        if flt < .1: return str(float(np.round(100 * flt))).rjust(4, ' ').ljust(5, '0')
        else: return str(float(np.round(100 * flt))).ljust(5, '0')
    
    def __repr__(self): return ''

if __name__ == '__main__':
    # Usuage
    X = pd.DataFrame(np.random.normal(0,1,(500,50)))
    test = MultipleHypTest(X)
    
    test.run()
