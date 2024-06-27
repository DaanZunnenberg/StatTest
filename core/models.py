import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def Fourier4(_: float, __: float) -> float: return 1 / (1 + _ ** 2) ** (2 * __)

class inv_measure(object):

    def __init__(self, **kwargs): 
        for arg, kwarg in kwargs.items(): self.__setattr__(arg, kwarg)
    
    def inv_measure(self, model: str) -> ...:
        match model:
            case 'BivariateOU':
                ...

class BivariateOUProcess:
    def __init__(self, theta1=0.015, theta2=0.006, mu1=0.5, mu2=0.5, sigma1=1.5, sigma2=3.0, rho=0.5, T=200, dt=1/20):
        self.theta1 = theta1  # Mean reversion rate for process 1
        self.theta2 = theta2  # Mean reversion rate for process 2
        self.mu1 = mu1        # Long-term mean for process 1
        self.mu2 = mu2        # Long-term mean for process 2
        self.sigma1 = sigma1  # Volatility for process 1
        self.sigma2 = sigma2  # Volatility for process 2
        self.rho = rho        # Correlation coefficient between the two processes
        self.T = T            # Total time
        self.dt = dt          # Time step
        self.N = int(T / dt)  # Number of time steps
        self.t = np.linspace(0, T, self.N)  # Time vector

        # Correlation matrix and Cholesky decomposition
        self.corr_matrix = np.array([[1, self.rho], [self.rho, 1]])
        self.L = np.linalg.cholesky(self.corr_matrix)

        # Initialize the processes
        self.x = np.zeros(self.N)
        self.y = np.zeros(self.N)

        # Initial values
        self.x[0] = self.mu1
        self.y[0] = self.mu2

    def sigma1_fun(self, x):
        return x ** 0.5

    def sigma2_fun(self, x):
        return x ** 0.5

    def simulate(self, seed=False):
        if seed: np.random.seed(seed)
        # Simulate the bivariate OU process
        for i in range(1, self.N):
            z = np.random.normal(size=2)
            dz = np.dot(self.L, z) * np.sqrt(self.dt)
            
            self.x[i] = self.x[i-1] + self.theta1 * (self.mu1 - self.x[i-1]) * self.dt + self.sigma1 * dz[0]
            self.y[i] = self.y[i-1] + self.theta2 * (self.mu2 - self.y[i-1]) * self.dt + self.sigma2 * dz[1]

    def dataframe(self, **kwargs):
        dataframe: pd.DataFrame = pd.DataFrame(np.matrix([self.x, self.y]).T, columns = ['process 1', 'process 2'])
        dataframe.name = 'Ornstein Uhlenbeck'
        for _, __ in kwargs.items(): dataframe.__setattr__(_, __)
        return dataframe
    
    def config(self):
        return self.dataframe(), self.T, self.N
    
    def plot(self):
        # Plot the results
        plt.figure(figsize=(12, 6))
        plt.plot(self.t, self.x, label='OU Process 1')
        plt.plot(self.t, self.y, label='OU Process 2')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Bivariate Correlated Ornstein-Uhlenbeck Process')
        plt.legend()
        plt.grid(True)
        plt.show()
        
class BivariateCorrelatedBM:
    def __init__(self, mu1=0.01, mu2=0.01, sigma1=1.0, sigma2=1.0, rho=0.5, T=1.0, dt=0.01):
        self.mu1 = mu1        # Drift for process 1
        self.mu2 = mu2        # Drift for process 2
        self.sigma1 = sigma1  # Volatility for process 1
        self.sigma2 = sigma2  # Volatility for process 2
        self.rho = rho        # Correlation coefficient between the two processes
        self.T = T            # Total time
        self.dt = dt          # Time step
        self.N = int(T / dt)  # Number of time steps
        self.t = np.linspace(0, T, self.N)  # Time vector

        # Correlation matrix and Cholesky decomposition
        self.corr_matrix = np.array([[1, self.rho], [self.rho, 1]])
        self.L = np.linalg.cholesky(self.corr_matrix)

        # Initialize the processes
        self.x = np.zeros(self.N)
        self.y = np.zeros(self.N)

        # Initial values
        self.x[0] = 0
        self.y[0] = 0

    def simulate(self, seed=False):
        if seed: np.random.seed(seed)
        # Simulate the bivariate correlated Brownian motion
        for i in range(1, self.N):
            z = np.random.normal(size=2)
            dz = np.dot(self.L, z) * np.sqrt(self.dt)
            
            self.x[i] = self.x[i-1] + self.mu1 * self.dt + self.sigma1 * dz[0]
            self.y[i] = self.y[i-1] + self.mu2 * self.dt + self.sigma2 * dz[1]

    def dataframe(self, **kwargs):
        dataframe: pd.DataFrame = pd.DataFrame(np.matrix([self.x, self.y]).T, columns = ['process 1', 'process 2'])
        dataframe.name = 'Correlated Brownian motion'
        for _, __ in kwargs.items(): dataframe.__setattr__(_, __)
        return dataframe
    
    def config(self):
        return self.dataframe(), self.T, self.N
    
    def plot(self):
        # Plot the results
        plt.figure(figsize=(12, 6))
        plt.plot(self.t, self.x, label='Correlated BM Process 1')
        plt.plot(self.t, self.y, label='Correlated BM Process 2')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Bivariate Correlated Brownian Motion')
        plt.legend()
        plt.grid(True)
        plt.show()

class BivariateNonHomogeneous:
    def __init__(self, T=1.0, dt=0.01, rho = .5):
        self.rho = rho
        self.T = T            # Total time
        self.dt = dt          # Time step
        self.N = int(T / dt)  # Number of time steps
        self.t = np.linspace(0, T, self.N)  # Time vector

        # Correlation matrix and Cholesky decomposition
        self.corr_matrix = np.array([[1, self.rho], [self.rho, 1]])
        self.L = np.linalg.cholesky(self.corr_matrix)

        # Initialize the processes
        self.x = np.zeros(self.N)
        self.y = np.zeros(self.N)

        # Initial values
        self.x[0] = np.random.normal()
        self.y[0] = np.random.normal()

    
    @staticmethod
    def mu(x,t): return -np.max([2 * t * x, 20 * x])
    
    @staticmethod
    def sigm(t): return 2.01 + 2 * np.sin(np.pi * t)

    def simulate(self, seed=False):
        if seed: np.random.seed(seed)
        # Simulate the bivariate correlated Brownian motion
        for i in range(1, self.N):
            z = np.random.normal(size=2)
            dz = np.dot(self.L, z) * np.sqrt(self.dt)
            self.x[i] = self.x[i-1] + self.mu(self.x[i-1], i * self.dt) * self.dt + self.sigm(i * self.dt) * dz[0]
            self.y[i] = self.y[i-1] + self.mu(self.y[i-1], i * self.dt) * self.dt + self.sigm(i * self.dt) * dz[1]
            # self.x[i] = self.x[i-1] + self.mu1 * self.dt + 0.5 * np.sqrt(np.max([self.x[i-1], 0])) * dz[0]
            # self.y[i] = self.y[i-1] + self.mu2 * self.dt + 0.5 * np.sqrt(np.max([self.y[i-1], 0])) * dz[1]

    def dataframe(self, **kwargs):
        dataframe: pd.DataFrame = pd.DataFrame(np.matrix([self.x, self.y]).T, columns = ['process 1', 'process 2'])
        dataframe.name = 'Correlated Diffusion'
        for _, __ in kwargs.items(): dataframe.__setattr__(_, __)
        return dataframe
    
    def config(self):
        return self.dataframe(), self.T, self.N
    
    def plot(self):
        # Plot the results
        plt.figure(figsize=(12, 6))
        plt.plot(self.t, self.x, label='Correlated diffusion Process 1')
        plt.plot(self.t, self.y, label='Correlated diffusion Process 2')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Bivariate non-homogeneous Diffusion')
        plt.legend()
        plt.grid(True)
        plt.show()

class BivariateCorrelatedDiffusion:
    def __init__(self, mu1=0.0, mu2=0.0, rho=0.5, T=1.0, dt=0.01, sigma1 = .1, sigma2 = .1, gamma = 1):
        self.mu1 = mu1        # Drift for process 1
        self.mu2 = mu2        # Drift for process 2
        self.sigma1 = sigma1  # Variance of process 1
        self.sigma2 = sigma2  # Variance of process 2
        self.gamma = gamma    # Recurrent setting
        self.rho = rho        # Correlation coefficient between the two processes
        self.T = T            # Total time
        self.dt = dt          # Time step
        self.N = int(T / dt)  # Number of time steps
        self.t = np.linspace(0, T, self.N)  # Time vector

        # Correlation matrix and Cholesky decomposition
        self.corr_matrix = np.array([[1, self.rho], [self.rho, 1]])
        self.L = np.linalg.cholesky(self.corr_matrix)

        # Initialize the processes
        self.x = np.zeros(self.N)
        self.y = np.zeros(self.N)

        # Initial values
        self.x[0] = np.random.normal()
        self.y[0] = np.random.normal()

    def Milstein_method_correction(self, x, dz, dt):
        bo = (1 + x ** 2) ** self.gamma
        bp = 2 * x * self.gamma * (1 + x ** 2) ** (self.gamma - 1)
        fo = .5 * bo * bp * (dz ** 2 - dt)
        return fo
    
    def simulate(self, seed=False):
        if seed: np.random.seed(seed)
        # Simulate the bivariate correlated Brownian motion
        for i in range(1, self.N):
            z = np.random.normal(size=2)
            dz = np.dot(self.L, z) * np.sqrt(self.dt)
            self.x[i] = self.x[i-1] + self.mu1 * self.dt + self.sigma1 * (1 + self.x[i-1] ** 2) ** self.gamma * dz[0] + self.Milstein_method_correction(self.x[i-1], dz[0], self.dt)
            self.y[i] = self.y[i-1] + self.mu2 * self.dt + self.sigma1 * (1 + self.y[i-1] ** 2) ** self.gamma * dz[1] + self.Milstein_method_correction(self.y[i-1], dz[1], self.dt)
            # self.x[i] = self.x[i-1] + self.mu1 * self.dt + 0.5 * np.sqrt(np.max([self.x[i-1], 0])) * dz[0]
            # self.y[i] = self.y[i-1] + self.mu2 * self.dt + 0.5 * np.sqrt(np.max([self.y[i-1], 0])) * dz[1]

    def dataframe(self, **kwargs):
        dataframe: pd.DataFrame = pd.DataFrame(np.matrix([self.x, self.y]).T, columns = ['process 1', 'process 2'])
        dataframe.name = 'Correlated Diffusion'
        for _, __ in kwargs.items(): dataframe.__setattr__(_, __)
        return dataframe
    
    def config(self):
        return self.dataframe(), self.T, self.N
    
    def plot(self):
        # Plot the results
        plt.figure(figsize=(12, 6))
        plt.plot(self.t, self.x, label='Correlated diffusion Process 1')
        plt.plot(self.t, self.y, label='Correlated diffusion Process 2')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Bivariate Correlated Diffusion')
        plt.legend()
        plt.grid(True)
        plt.show()

