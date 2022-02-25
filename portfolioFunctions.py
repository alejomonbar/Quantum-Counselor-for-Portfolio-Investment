#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 11:52:37 2022

@author: alejomonbar
"""
import numpy as np
from docplex.mp.model import Model
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.runtime import QAOAClient, VQEClient
from qiskit.algorithms.optimizers import SPSA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.providers.basicaer import QasmSimulatorPy  # local simulator
from qiskit.algorithms import VQE, QAOA
import matplotlib.pyplot as plt

def mu_fun(data, holding_period):
    """
    assets’ forecast returns at time t

    Parameters
    ----------
    data : np.array(num_time_steps)
        Price of the asset.
    holding_period: period to divide the data.

    Returns
    -------
    None.

    """
    min_t = min([len(d) for d in data])
    num_assets = len(data)
    mu = []
    for asset in range(num_assets):
        mu.append([data[asset][t+1]/data[asset][t] - 1 if data[asset][t] != 0 else 1 for t in range(min_t-1)])
    mu = np.array(mu)
    split =  min_t // holding_period
    mus = np.array([mu[:,i * holding_period:(i+1) * holding_period].sum(axis=1) for i in range(split)])
    return np.array(mus)

def cov_matrix(data, holding_period):
    min_t = min([len(d) for d in data])
    num_assets = len(data)
    mu = []
    for asset in range(num_assets):
        mu.append([data[asset][t+1]/data[asset][t] - 1 if data[asset][t] != 0 else 1 for t in range(min_t-1)])
    mu = np.array(mu)
    split =  min_t // holding_period
    cov =  [np.cov(mu[:,i*holding_period:(i+1)*holding_period], rowvar=True) for i in range(split)]
    return np.array(cov)

def portfolioOptimization(mu, sigma, risk_aversion, max_invest, Lambda=0.001, rho=1.0, simplified=False):
    """
    

    Parameters
    ----------
    mu : list
        assets’ forecast returns.
    sigma : matrix
        the assets’ covariance
    risk_aversion : float
        risk aversion.
    max_invest : list[float]
        percentage of individual assets maximum invesment. between (0,1]
    Lambda: floar
        tranasaction cost multiplier
    rho: Multiplier of the investment restriction that the budget should be 
        equal to the investment.

    Returns
    -------
    op : Docplex file.
        Quadratic program encoding the optimization.

    """
    
    periods, num_assets = mu.shape
    mdl = Model("portfolioOptimization")
    w = [mdl.binary_var_list(num_assets, name=f"w{i}") for i in range(periods)] 
    # w is a variable decision that tells if invest or not in some specific asset at an specific time
    risk = 0
    returns = 0
    eq_constraint = 0
    transaction_cost = Lambda * max_invest[0] * np.dot(w[0],w[0])
    for i in range(periods):
        risk += (max_invest * w[i]).T @ sigma[i] @ (max_invest * w[i])
        returns += np.dot(mu[i], max_invest * w[i])
        eq_constraint += (mdl.sum(max_invest * w[i]) - 1) ** 2 
        if i > 0:
            dw = [max_invest[j] * (w[i][j] - w[i-1][j]) for j in range(num_assets)]
            transaction_cost += Lambda * np.dot(dw, dw)
    if simplified:
        mdl.minimize(0.5 * risk_aversion * risk - returns)
    else:
        mdl.minimize(0.5 * risk_aversion * risk - returns + transaction_cost + rho * eq_constraint)
    op = from_docplex_mp(mdl)
    return op

def portfolioOptimization_NewApproach(mu, sigma, risk_aversion, kappa, max_invest, U,
                                      beta=0.1, Lambda=0.001, rho=0.1, simplified=False):
    """
    

    Parameters
    ----------
    mu : list
        assets’ forecast returns.
    sigma : matrix
        the assets’ covariance
    kappa: list[float]
        mean relative error of the prediction vs the data for the case test
        of the QNN training
    risk_aversion : float
        risk aversion.
    max_invest : list[float]
        percentage of individual assets maximum invesment. between (0,1]
    beta: float
        Lagrange multiplier of the forecasting risk
    Lambda: floar
        tranasaction cost multiplier
    rho: Multiplier of the investment restriction that the budget should be 
        equal to the investment.
    U: List[float]
        the mean value of mu value for each asset from a historical standpoint.
    
    Returns
    -------
    op : Docplex file.
        Quadratic program encoding the optimization.

    """
    
    periods, num_assets = mu.shape
    mdl = Model("portfolioOptimization")
    w = [mdl.binary_var_list(num_assets, name=f"w{i}") for i in range(periods)] 
    # w is a variable decision that tells if invest or not in some specific asset at an specific time
    risk = 0
    returns = 0
    eq_constraint = 0
    risk_forecasting = 0 # Forecasting associated risk
    transaction_cost = Lambda * max_invest[0] * np.dot(w[0],w[0])
    for i in range(periods):
        risk += (max_invest * w[i]).T @ sigma[i] @ (max_invest * w[i])
        risk_forecasting += kappa.T @ (max_invest * w[i])
        returns += np.dot(mu[i], max_invest * w[i])
        eq_constraint += (np.sum(kappa)/np.sum(mu[i]))*(mdl.sum(max_invest * w[i]) - 1) ** 2 
        if i > 0:
            dw = [max_invest[j] * (w[i][j] - w[i-1][j]) for j in range(num_assets)]
            transaction_cost += Lambda * np.dot(dw, dw)
    if simplified:
        mdl.minimize(0.5 * risk_aversion * risk - returns)
    else:
        mdl.minimize(0.5 * risk_aversion * risk - returns + transaction_cost +
                     rho * eq_constraint + beta * risk_forecasting)
    op = from_docplex_mp(mdl)
    return op


def Optimization_QAOA(qubo, reps=1, optimizer=SPSA(maxiter=50), backend=None,
                      shots=1024, alpha=0.75, provider=None, local=False):
    intermediate_info = {'nfev': [],
                         'parameters': [],
                         'stddev': [],
                         'mean': []
                             }
    
    def callback(nfev, parameters, mean, stddev):
        intermediate_info['nfev'].append(nfev)
        intermediate_info['parameters'].append(parameters)
        intermediate_info['mean'].append(mean)
        intermediate_info['stddev'].append(stddev)
    
    if local:
        qaoa_mes = QAOA(optimizer=optimizer, reps=reps, quantum_instance=QasmSimulatorPy(),
                        callback=callback)
    else:
        qaoa_mes = QAOAClient(provider=provider, backend=backend, reps=reps, alpha=alpha,
                             shots=shots, callback=callback, optimizer=optimizer,
                             optimization_level=3)
    qaoa = MinimumEigenOptimizer(qaoa_mes)
    result = qaoa.solve(qubo)
    return result, intermediate_info

def Optimization_VQE(qubo, ansatz, optimizer=SPSA(maxiter=50), backend=None,
                     shots=1024, provider=None, local=False):

    intermediate_info = {'nfev': [],
                         'parameters': [],
                         'stddev': [],
                         'mean': []
                         }

    def callback(nfev, parameters, mean, stddev):
        intermediate_info['nfev'].append(nfev)
        intermediate_info['parameters'].append(parameters)
        intermediate_info['mean'].append(mean)
        intermediate_info['stddev'].append(stddev)
        
    if local:
        vqe_mes = VQE(ansatz=ansatz, quantum_instance=QasmSimulatorPy(),
                        callback=callback, optimizer=optimizer)
    else:
        vqe_mes = VQEClient(ansatz=ansatz, provider=provider, backend=backend, shots=shots,
                        callback=callback, optimizer=optimizer)
    vqe = MinimumEigenOptimizer(vqe_mes)
    result = vqe.solve(qubo)
    return result, intermediate_info

def transaction_costs(w, v, periods, max_invest):
    w = max_invest * w
    cost = [v * np.sum(w[0])]
    for i in range(periods-1):
        cost.append(v * np.sum(np.abs(w[i+1] - w[i])))
    return np.array(cost) 

def profits(w, mu, v, periods, max_invest):
    cost = transaction_costs(w, v, periods, max_invest)
    w = max_invest * w
    profit = []
    for i in range(periods):
        profit.append(mu[i].T @ w[i] - cost[i])
    return np.array(profit)

def get_figure(items_in_bins, weights, max_weight, title=None):
    """Get plot of the solution of the Bin Packing Problem.

    Args:
        result : The calculated result of the problem

    Returns:
        fig: A plot of the solution, where x and y represent the bins and
        sum of the weights respectively.
    """
    colors = plt.cm.get_cmap("jet", len(weights))
    num_bins = len(items_in_bins)
    fig, axes = plt.subplots()
    for _, bin_i in enumerate(items_in_bins):
        sum_items = 0
        for item in bin_i:
            axes.bar(_, weights[item], bottom=sum_items, label=f"Item {item}", color=colors(item))
            sum_items += weights[item]
    axes.hlines(max_weight, -0.5, num_bins - 0.5, linestyle="--", color="tab:red", label="Max Weight")
    axes.set_xticks(np.arange(num_bins))
    axes.set_xlabel("Bin")
    axes.set_ylabel("Weight")
    axes.legend()
    if title:
        axes.set_title(title)
    return fig

