#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 07:25:06 2022

@author: alejomonbar
"""
from qiskit_finance.data_providers import RandomDataProvider
import datetime
import matplotlib.pyplot as plt
import numpy as np
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit import Aer
from qiskit.algorithms import VQE, QAOA, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SLSQP
from qiskit_optimization.algorithms import MinimumEigenOptimizer, CplexOptimizer 
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit import IBMQ
from qiskit_optimization.runtime import QAOAClient, QAOAProgram
from qiskit.algorithms.optimizers import SPSA, COBYLA
from qiskit.circuit.library import TwoLocal
import pandas as pd
from qiskit.providers.basicaer import QasmSimulatorPy  # local simulator
from qiskit.algorithms import VQE, QAOA

from portfolioFunctions import mu_fun, cov_matrix, portfolioOptimization, Optimization_QAOA, Optimization_VQE
from portfolioFunctions import profits, transaction_costs, portfolioOptimization_NewApproach

import warnings


from qiskit_optimization.runtime import QAOAClient, VQEClient

warnings.filterwarnings('ignore')

IBMQ.load_account()

provider_7 = IBMQ.get_provider(hub='ibm-q-research', group='guanajuato-1',project='main')
backend_7 = provider_7.get_backend("ibmq_jakarta")

backend_sim = provider_7.get_backend("ibmq_qasm_simulator")

stocks_name = ["AAPL","ABB", "ABBV","CHL", "DUK", "HSBC", "TOT", "WMT"]
stocks_forecasting = {}
stocks_real = {}
kappa = {}
period_of_test = 90 # Days known the real price but not used during training
for name in stocks_name:
    fore = np.load(f"./Data/Stocks_prediction/data_{name}_test.npy", allow_pickle=True)
    stocks_forecasting[name] = fore[:,0,1]
    stocks_real[name] = np.array(pd.read_csv(f"./Data/Stocks/{name}.csv")["Close"])
    kappa[name] = (np.abs(fore[:period_of_test,0,1] - fore[:period_of_test,0,0]) / fore[:period_of_test,0,0]).mean()
    
m = {}

# Set parameters for assets and risk factor
m["num_assets"] = 4 # set number of assets
m["gamma"] = 0.5   # risk aversion to 0.5
m["lambda"] = 0.001 # Transaction Cost
m["holding_period"] = 30 # Days of keeping the assets
m["periods"] = 4
m["max_investment"] = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]) # Maximum investment per asset of the total budget
m["assets"] = ["AAPL","ABB", "ABBV","CHL", "DUK", "HSBC", "TOT", "WMT"]
m["rho"] = 5 # Lagrange multiplier for the budget constraint 

m["data"] = [stocks_forecasting[name] for name in m["assets"]]

m["mu"] = mu_fun(m["data"], m["holding_period"])[-m["periods"]:]
m["sigma"] = cov_matrix(m["data"], m["holding_period"])
m["qp"] = portfolioOptimization(m["mu"], m["sigma"][-m["periods"]:], m["gamma"],
                           m["max_investment"], m["lambda"], m["rho"])
# Create a converter from quadratic program to quadratic unconstrained binary optimization (QUBO) representation
m["qubo"] = QuadraticProgramToQubo().convert(m["qp"])


m["QAOA"] = {}

# m["QAOA"]["SPSA"] = Optimization_QAOA(m["qubo"], reps=1, optimizer=SPSA(maxiter=50), backend=backend_sim,
#                                       provider=provider_7)


qaoa_mes = QAOAProgram(provider=provider_7, backend=backend_sim, reps=1,
                             shots=1024, optimizer=COBYLA(maxiter=1))
qaoa = MinimumEigenOptimizer(qaoa_mes)
result = qaoa.solve(m["qubo"])


