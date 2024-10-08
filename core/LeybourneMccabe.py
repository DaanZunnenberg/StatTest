import sys
import os
import time
import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.tsatools import lagmat
from numpy.testing import assert_equal, assert_almost_equal


class Leybourne(object):
    """
    Class wrapper for Leybourne-Mccabe stationarity test
    """

    def __init__(self):
        """
        Asymptotic critical values for the two different models specified
        for the Leybourne-McCabe stationarity test. Asymptotic CVs are the
        same as the asymptotic CVs for the KPSS stationarity test.

        Notes
        -----
        The p-values are generated through Monte Carlo simulation using
        1,000,000 replications and 10,000 data points.
        """
        self.__leybourne_critical_values = {}
        # constant-only model
        self.__c = ((99.9999, 0.00819), (99.999, 0.01050), (99.99, 0.01298),
                    (99.9, 0.01701), (99.8, 0.01880), (99.7, 0.02005),
                    (99.6, 0.02102), (99.5, 0.02186), (99.4, 0.02258),
                    (99.3, 0.02321), (99.2, 0.02382), (99.1, 0.02437),
                    (99.0, 0.02488), (97.5, 0.03045), (95.0, 0.03662),
                    (92.5, 0.04162), (90.0, 0.04608), (87.5, 0.05024),
                    (85.0, 0.05429), (82.5, 0.05827), (80.0, 0.06222),
                    (77.5, 0.06621), (75.0, 0.07026), (72.5, 0.07439),
                    (70.0, 0.07859), (67.5, 0.08295), (65.0, 0.08747),
                    (62.5, 0.09214), (60.0, 0.09703), (57.5, 0.10212),
                    (55.0, 0.10750), (52.5, 0.11315), (50.0, 0.11907),
                    (47.5, 0.12535), (45.0, 0.13208), (42.5, 0.13919),
                    (40.0, 0.14679), (37.5, 0.15503), (35.0, 0.16403),
                    (32.5, 0.17380), (30.0, 0.18443), (27.5, 0.19638),
                    (25.0, 0.20943), (22.5, 0.22440), (20.0, 0.24132),
                    (17.5, 0.26123), (15.0, 0.28438), (12.5, 0.31242),
                    (10.0, 0.34699), (7.5, 0.39354), (5.0, 0.45995),
                    (2.5, 0.58098), (1.0, 0.74573), (0.9, 0.76453),
                    (0.8, 0.78572), (0.7, 0.81005), (0.6, 0.83863),
                    (0.5, 0.87385), (0.4, 0.91076), (0.3, 0.96501),
                    (0.2, 1.03657), (0.1, 1.16658), (0.01, 1.60211),
                    (0.001, 2.03312), (0.0001, 2.57878))
        self.__leybourne_critical_values['c'] = np.asarray(self.__c)
        # constant+trend model
        self.__ct = ((99.9999, 0.00759), (99.999, 0.00870), (99.99, 0.01023),
                     (99.9, 0.01272), (99.8, 0.01378), (99.7, 0.01454),
                     (99.6, 0.01509), (99.5, 0.01559), (99.4, 0.01598),
                     (99.3, 0.01637), (99.2, 0.01673), (99.1, 0.01704),
                     (99.0, 0.01731), (97.5, 0.02029), (95.0, 0.02342),
                     (92.5, 0.02584), (90.0, 0.02791), (87.5, 0.02980),
                     (85.0, 0.03158), (82.5, 0.03327), (80.0, 0.03492),
                     (77.5, 0.03653), (75.0, 0.03813), (72.5, 0.03973),
                     (70.0, 0.04135), (67.5, 0.04298), (65.0, 0.04464),
                     (62.5, 0.04631), (60.0, 0.04805), (57.5, 0.04981),
                     (55.0, 0.05163), (52.5, 0.05351), (50.0, 0.05546),
                     (47.5, 0.05753), (45.0, 0.05970), (42.5, 0.06195),
                     (40.0, 0.06434), (37.5, 0.06689), (35.0, 0.06962),
                     (32.5, 0.07252), (30.0, 0.07564), (27.5, 0.07902),
                     (25.0, 0.08273), (22.5, 0.08685), (20.0, 0.09150),
                     (17.5, 0.09672), (15.0, 0.10285), (12.5, 0.11013),
                     (10.0, 0.11917), (7.5, 0.13104), (5.0, 0.14797),
                     (2.5, 0.17775), (1.0, 0.21801), (0.9, 0.22282),
                     (0.8, 0.22799), (0.7, 0.23387), (0.6, 0.24109),
                     (0.5, 0.24928), (0.4, 0.25888), (0.3, 0.27173),
                     (0.2, 0.28939), (0.1, 0.32200), (0.01, 0.43218),
                     (0.001, 0.54708), (0.0001, 0.69538))
        self.__leybourne_critical_values['ct'] = np.asarray(self.__ct)

    def __leybourne_crit(self, stat, model='c'):
        """
        Linear interpolation for Leybourne p-values and critical values

        Parameters
        ----------
        stat : float
            The Leybourne-McCabe test statistic
        model : {'c','ct'}
            The model used when computing the test statistic. 'c' is default.

        Returns
        -------
        pvalue : float
            The interpolated p-value
        cvdict : dict
            Critical values for the test statistic at the 1%, 5%, and 10%
            levels

        Notes
        -----
        The p-values are linear interpolated from the quantiles of the
        simulated Leybourne-McCabe (KPSS) test statistic distribution
        """
        table = self.__leybourne_critical_values[model]
        # reverse the order
        y = table[:, 0]
        x = table[:, 1]
        # LM cv table contains quantiles multiplied by 100
        pvalue = np.interp(stat, x, y) / 100.0
        cv = [1.0, 5.0, 10.0]
        crit_value = np.interp(cv, np.flip(y), np.flip(x))
        cvdict = {"1%": crit_value[0], "5%": crit_value[1],
                  "10%": crit_value[2]}
        return pvalue, cvdict

    def _tsls_arima(self, x, arlags, model):
        """
        Two-stage least squares approach for estimating ARIMA(p, 1, 1)
        parameters as an alternative to MLE estimation in the case of
        solver non-convergence

        Parameters
        ----------
        x : array_like
            data series
        arlags : int
            AR(p) order
        model : {'c','ct'}
            Constant and trend order to include in regression
            * 'c'  : constant only
            * 'ct' : constant and trend

        Returns
        -------
        arparams : int
            AR(1) coefficient plus constant
        theta : int
            MA(1) coefficient
        olsfit.resid : ndarray
            residuals from second-stage regression
        """
        endog = np.diff(x, axis=0)
        exog = lagmat(endog, arlags, trim='both')
        # add constant if requested
        if model == 'ct':
            exog = add_constant(exog)
        # remove extra terms from front of endog
        endog = endog[arlags:]
        if arlags > 0:
            resids = lagmat(OLS(endog, exog).fit().resid, 1, trim='forward')
        else:
            resids = lagmat(-endog, 1, trim='forward')
        # add negated residuals column to exog as MA(1) term
        exog = np.append(exog, -resids, axis=1)
        olsfit = OLS(endog, exog).fit()
        if model == 'ct':
            arparams = olsfit.params[1:(len(olsfit.params)-1)]
        else:
            arparams = olsfit.params[0:(len(olsfit.params)-1)]
        theta = olsfit.params[len(olsfit.params)-1]
        return arparams, theta, olsfit.resid

    def _autolag(self, x):
        """
        Empirical method for Leybourne-McCabe auto AR lag detection.
        Set number of AR lags equal to the first PACF falling within the
        95% confidence interval. Maximum nuber of AR lags is limited to
        the smaller of 10 or 1/2 series length. Minimum is zero lags.

        Parameters
        ----------
        x : array_like
            data series

        Returns
        -------
        arlags : int
            AR(p) order
        """
        p = pacf(x, nlags=min(int(len(x)/2), 10), method='ols')
        ci = 1.960 / np.sqrt(len(x))
        arlags = max(0, ([n-1 for n, i in enumerate(p) if abs(i) < ci]
                         + [len(p)-1])[0])
        return arlags

    def run(self, x, arlags=1, regression='c', method='mle', varest='var94'):
        """
        Leybourne-McCabe stationarity test

        The Leybourne-McCabe test can be used to test for stationarity in a
        univariate process.

        Parameters
        ----------
        x : array_like
            data series
        arlags : int
            number of autoregressive terms to include, default=None
        regression : {'c','ct'}
            Constant and trend order to include in regression
            * 'c'  : constant only (default)
            * 'ct' : constant and trend
        method : {'mle','ols'}
            Method used to estimate ARIMA(p, 1, 1) filter model
            * 'mle' : condition sum of squares maximum likelihood (default)
            * 'ols' : two-stage least squares
        varest : {'var94','var99'}
            Method used for residual variance estimation
            * 'var94' : method used in original Leybourne-McCabe paper (1994)
                        (default)
            * 'var99' : method used in follow-up paper (1999)

        Returns
        -------
        lmstat : float
            test statistic
        pvalue : float
            based on MC-derived critical values
        arlags : int
            AR(p) order used to create the filtered series
        cvdict : dict
            critical values for the test statistic at the 1%, 5%, and 10%
            levels

        Notes
        -----
        H0 = series is stationary

        Basic process is to create a filtered series which removes the AR(p)
        effects from the series under test followed by an auxiliary regression
        similar to that of Kwiatkowski et al (1992). The AR(p) coefficients
        are obtained by estimating an ARIMA(p, 1, 1) model. Two methods are
        provided for ARIMA estimation: MLE and two-stage least squares.

        Two methods are provided for residual variance estimation used in the
        calculation of the test statistic. The first method ('var94') is the
        mean of the squared residuals from the filtered regression. The second
        method ('var99') is the MA(1) coefficient times the mean of the squared
        residuals from the ARIMA(p, 1, 1) filtering model.

        An empirical autolag procedure is provided. In this context, the number
        of lags is equal to the number of AR(p) terms used in the filtering
        step. The number of AR(p) terms is set equal to the to the first PACF
        falling within the 95% confidence interval. Maximum nuber of AR lags is
        limited to 1/2 series length.

        References
        ----------
        Kwiatkowski, D., Phillips, P.C.B., Schmidt, P. & Shin, Y. (1992).
        Testing the null hypothesis of stationarity against the alternative of
        a unit root. Journal of Econometrics, 54: 159–178.

        Leybourne, S.J., & McCabe, B.P.M. (1994). A consistent test for a
        unit root. Journal of Business and Economic Statistics, 12: 157–166.

        Leybourne, S.J., & McCabe, B.P.M. (1999). Modified stationarity tests
        with data-dependent model-selection rules. Journal of Business and
        Economic Statistics, 17: 264-270.

        Schwert, G W. (1987). Effects of model specification on tests for unit
        roots in macroeconomic data. Journal of Monetary Economics, 20: 73–103.
        """
        if regression not in ['c', 'ct']:
            raise ValueError(
                'LM: regression option \'%s\' not understood' % regression)
        if method not in ['mle', 'ols']:
            raise ValueError(
                'LM: method option \'%s\' not understood' % method)
        if varest not in ['var94', 'var99']:
            raise ValueError(
                'LM: varest option \'%s\' not understood' % varest)
        x = np.asarray(x)
        if x.ndim > 2 or (x.ndim == 2 and x.shape[1] != 1):
            raise ValueError(
                'LM: x must be a 1d array or a 2d array with a single column')
        x = np.reshape(x, (-1, 1))
        # determine AR order if not specified
        if arlags is None:
            arlags = self._autolag(x)
        elif not isinstance(arlags, int) or arlags < 0 \
                or arlags > int(len(x) / 2):
            raise ValueError(
                'LM: arlags must be an integer in range [0..%s]'
                % str(int(len(x) / 2)))
        # estimate the reduced ARIMA(p, 1, 1) model
        if method == 'mle':
            if regression == 'ct':
                reg = 't'
            else:
                reg = None
            arfit = ARIMA(x, order=(arlags, 1, 1), trend=reg).fit()
            resids = arfit.resid
            arcoeffs = []
            if arlags > 0:
                arcoeffs = arfit.arparams
            theta = arfit.maparams[0]
        else:
            arcoeffs, theta, resids = self._tsls_arima(x, arlags,
                                                       model=regression)
        # variance estimator from (1999) LM paper
        var99 = abs(theta * np.sum(resids**2) / len(resids))
        # create the filtered series:
        #   z(t) = x(t) - arcoeffs[0]*x(t-1) - ... - arcoeffs[p-1]*x(t-p)
        z = np.full(len(x) - arlags, np.inf)
        for i in range(len(z)):
            z[i] = x[i + arlags]
            for j in range(len(arcoeffs)):
                z[i] -= arcoeffs[j] * x[i + arlags - j - 1]
        # regress the filtered series against a constant and
        # trend term (if requested)
        if regression == 'c':
            resids = z - z.mean()
        else:
            resids = OLS(z, add_constant(np.arange(1, len(z) + 1))).fit().resid
        # variance estimator from (1994) LM paper
        var94 = np.sum(resids**2) / len(resids)
        # compute test statistic with specified variance estimator
        eta = np.sum(resids.cumsum()**2) / (len(resids)**2)
        if varest == 'var99':
            lmstat = eta / var99
        else:
            lmstat = eta / var94
        # calculate pval
        crit = self.__leybourne_crit(lmstat, regression)
        lmpval = crit[0]
        cvdict = crit[1]
        return lmstat, lmpval, arlags, cvdict

    def __call__(self, x, arlags=None, regression='c', method='mle',
                 varest='var94'):
        return self.run(x, arlags=arlags, regression=regression, method=method,
                        varest=varest)


# output results
def _print_res(res, st):
    print("  lmstat =", "{0:0.5f}".format(res[0]), " pval =",
          "{0:0.5f}".format(res[1]), " arlags =", res[2])
    print("    cvdict =", res[3])
    print("    time =", "{0:0.5f}".format(time.time() - st))


# unit tests taken from Schwert (1987) and verified against Matlab
def main():
    print("Leybourne-McCabe stationarity test...")
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    run_dir = os.path.join(cur_dir, "results")
    files = ['BAA.csv', 'DBAA.csv', 'DSP500.csv', 'DUN.csv', 'SP500.csv',
             'UN.csv']
    lm = Leybourne()
    # turn off solver warnings
    warnings.simplefilter("ignore")
    for file in files:
        print(" test file =", file)
        mdl_file = os.path.join(run_dir, file)
        mdl = np.asarray(pd.read_csv(mdl_file))
        st = time.time()
        if file == 'BAA.csv':
            res = lm(mdl, regression='ct')
            _print_res(res=res, st=st)
            assert_equal(res[2], 3)
            assert_almost_equal(res[0], 5.4438, decimal=3)
            assert_almost_equal(res[1], 0.0000, decimal=3)
            st = time.time()
            res = lm(mdl, regression='ct', method='ols')
            _print_res(res=res, st=st)
            assert_equal(res[2], 3)
            assert_almost_equal(res[0], 5.4757, decimal=3)
            assert_almost_equal(res[1], 0.0000, decimal=3)
        elif file == 'DBAA.csv':
            res = lm(mdl)
            _print_res(res=res, st=st)
            assert_equal(res[2], 2)
            assert_almost_equal(res[0], 0.1173, decimal=3)
            assert_almost_equal(res[1], 0.5072, decimal=3)
            st = time.time()
            res = lm(mdl, regression='ct')
            _print_res(res=res, st=st)
            assert_equal(res[2], 2)
            assert_almost_equal(res[0], 0.1175, decimal=3)
            assert_almost_equal(res[1], 0.1047, decimal=3)
        elif file == 'DSP500.csv':
            res = lm(mdl)
            _print_res(res=res, st=st)
            assert_equal(res[2], 0)
            assert_almost_equal(res[0], 0.3118, decimal=3)
            assert_almost_equal(res[1], 0.1256, decimal=3)
            st = time.time()
            res = lm(mdl, varest='var99')
            _print_res(res=res, st=st)
            assert_equal(res[2], 0)
            assert_almost_equal(res[0], 0.3145, decimal=3)
            assert_almost_equal(res[1], 0.1235, decimal=3)
        elif file == 'DUN.csv':
            res = lm(mdl, regression='ct')
            _print_res(res=res, st=st)
            assert_equal(res[2], 3)
            assert_almost_equal(res[0], 0.0252, decimal=3)
            # assert_almost_equal(res[1], 0.9318, decimal=3)
            assert_almost_equal(res[1], 0.9286, decimal=3)
            st = time.time()
            res = lm(mdl, regression='ct', method='ols')
            _print_res(res=res, st=st)
            assert_equal(res[2], 3)
            assert_almost_equal(res[0], 0.0938, decimal=3)
            assert_almost_equal(res[1], 0.1890, decimal=3)
        elif file == 'SP500.csv':
            res = lm(mdl, arlags=4, regression='ct')
            _print_res(res=res, st=st)
            assert_almost_equal(res[0], 1.8761, decimal=3)
            assert_almost_equal(res[1], 0.0000, decimal=3)
            st = time.time()
            res = lm(mdl, arlags=4, regression='ct', method='ols')
            _print_res(res=res, st=st)
            assert_almost_equal(res[0], 1.9053, decimal=3)
            assert_almost_equal(res[1], 0.0000, decimal=3)
        elif file == 'UN.csv':
            res = lm(mdl, varest='var99')
            _print_res(res=res, st=st)
            assert_equal(res[2], 4)
            # assert_almost_equal(res[0], 285.6100, decimal=3)
            assert_almost_equal(res[0], 285.5181, decimal=3)
            assert_almost_equal(res[1], 0.0000, decimal=3)
            st = time.time()
            res = lm(mdl, method='ols', varest='var99')
            _print_res(res=res, st=st)
            assert_equal(res[2], 4)
            assert_almost_equal(res[0], 556.0444, decimal=3)
            assert_almost_equal(res[1], 0.0000, decimal=3)


if __name__ == "__main__":
    sys.exit(int(main() or 0))