import pandas as pd
import numpy as np
import pickle
from MFCF_main import MFCF_Forest
from gain_function import gf_sumsquares_gen
from utils_mfcf import j_LoGo
from set_up import *
import random
import os

# load model
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
model_path = os.path.join(parent_dir, 'dt.sav')
loaded_model = pickle.load(open(model_path, 'rb'))
#loaded_model = pickle.load(open('dt.sav', 'rb'))

def correlation_from_covariance(covariance):
    # Step 1: Calculate standard deviations
    std_deviations = np.sqrt(np.diag(covariance))
    # Step 2: Replace 0 standard deviations with a small value (e.g., 1e-6)
    std_deviations[std_deviations == 0] = 1e-6
    # Step 3: Calculate correlation matrix
    correlation_matrix = covariance / np.outer(std_deviations, std_deviations)
    return correlation_matrix

# mfcf
def generate_cov_mfcf(return_matrix, model_input, rd_date,training_window,lag=21,ct_control = None, 
                                 max_clique_size=2,min_clique_size=1,
                      threshold = 0,coordiation_number=np.inf,drop_sep=False,method='mean'):

    df = pd.DataFrame(np.array(return_matrix), columns=['col_{}'.format(i) for i in range(len(np.array(return_matrix)[0]))]) 

    if method == 'mean':
        sample_mean = df.mean()
        sample_mean = sample_mean*252
    elif method == 'ema':
        sample_mean = return_matrix.ewm(span=len(return_matrix), adjust=False).mean().iloc[-1]
        sample_mean = sample_mean*252

    elif method == 'std_o':
        prediction = []
        sample_mean = df.std()*np.sqrt(252)


    elif method == 'std':
        prediction = []
        for i in range(len(return_matrix.columns)):
            symbol = return_matrix.columns[i]
            predicting_base = model_input[model_input['Ticker']==symbol]
            predicting_base = predicting_base[predicting_base['Index']==rd_date] 
            predicting_base = predicting_base[['Volume', 'call_put_ratio_200', 'SQZ', 'MACD', 'vix_fix_gauge',
            'Greater_than_MA99',  'Greater_than_MA125', 'Index', 'Ticker_label','Close']]
            pre_X = loaded_model.predict(predicting_base)[0]
            if pre_X  ==1:
                prediction.append(1)
            else:
                prediction.append(-1)
        prediction_result = pd.Series(prediction, index=['col_{}'.format(i) for i in range(len(np.array(return_matrix)[0]))])
        sample_mean = df.std()*np.sqrt(252)*prediction_result


    elif method == 'std_random':
        prediction = [random.choice([1, -1]) for _ in range(len(return_matrix.columns))]
        prediction_result = pd.Series(prediction, index=['col_{}'.format(i) for i in range(len(np.array(return_matrix)[0]))])
        sample_mean = df.std()*np.sqrt(252)*prediction_result
    
        
    elif method == 'capm':
        prediction = []
        for i in range(len(return_matrix.columns)):
            symbol = return_matrix.columns[i]
            predicting_base = model_input[model_input['Ticker']==symbol]
            predicting_base = predicting_base[predicting_base['Index']<=rd_date] 
            predicting_base = predicting_base[predicting_base['Index']>rd_date-training_window] 
            # risk free
            risk_free_rate = predicting_base['tresuary_bills_60'].mean() 
            # expected market return
            expected_market_return = predicting_base['mkt_return'].mean()
            # Calculate beta
            predicting_base.reset_index(inplace=True,drop=True)
            stock_returns = predicting_base['Close'].pct_change().dropna()
            market_returns = predicting_base['mkt_return'][1:]
            # Covariance of the stock returns with the market returns
            covariance = np.cov(stock_returns, market_returns)[0, 1]
            # Variance of the market returns
            market_variance = np.var(market_returns)
            # Calculate beta
            beta = covariance / market_variance

            # Calculate the expected return using CAPM
            expected_return = risk_free_rate + beta * (expected_market_return - risk_free_rate)
            prediction.append(expected_return)
        sample_mean = pd.Series(prediction, index=['col_{}'.format(i) for i in range(len(np.array(return_matrix)[0]))])
        sample_mean = sample_mean*252     

    cov = df.cov()*252
    W = np.array(correlation_from_covariance(cov)**2)
    
    if ct_control is None:
        ct_control = {
            'max_clique_size': max_clique_size,
            'min_clique_size': min_clique_size,
            'threshold': threshold,
            'coordination_num':coordiation_number,
            'drop_sep': drop_sep
        }
    else:
        if 'max_clique_size' not in ct_control.keys():
            ct_control['max_clique_size']=max_clique_size
        if 'min_clique_size' not in ct_control.keys():
            ct_control['min_clique_size']=min_clique_size
        if 'coordination_num' not in ct_control.keys():
            ct_control['coordination_num']=coordiation_number
        if 'drop_sep' not in ct_control.keys():
            ct_control['drop_sep']=drop_sep

    cliques, separators, peo, tree = MFCF_Forest(W,ct_control,gf_sumsquares_gen)

    J = j_LoGo(np.array(cov), cliques, separators)
    adj_matrix = pd.DataFrame(J, columns=['col_{}'.format(i) for i in range(len(J[0]))])   
    return sample_mean, cov, adj_matrix


# mfcf
def sortino(return_matrix, model_input, rd_date,training_window,lag=21,ct_control = None, 
                                 max_clique_size=2,min_clique_size=1,
                      threshold = 0,coordiation_number=np.inf,drop_sep=False,method='mean'):

    df = pd.DataFrame(np.array(return_matrix), columns=['col_{}'.format(i) for i in range(len(np.array(return_matrix)[0]))]) 

    if method == 'std':
        prediction = []
        for i in range(len(return_matrix.columns)):
            symbol = return_matrix.columns[i]
            predicting_base = model_input[model_input['Ticker']==symbol]
            predicting_base = predicting_base[predicting_base['Index']==rd_date] 
            predicting_base = predicting_base[['Volume', 'call_put_ratio_200', 'SQZ', 'MACD', 'vix_fix_gauge',
            'Greater_than_MA99',  'Greater_than_MA125', 'Index', 'Ticker_label','Close']]
            pre_X = loaded_model.predict(predicting_base)[0]
            if pre_X  ==1:
                prediction.append(1)
            else:
                prediction.append(-1)
        prediction_result = pd.Series(prediction, index=['col_{}'.format(i) for i in range(len(np.array(return_matrix)[0]))])
        sample_mean = df.std()*np.sqrt(252)*prediction_result


        predicting_base = model_input[model_input['Ticker']==symbol]
        predicting_base = predicting_base[predicting_base['Index']<=rd_date] 
        predicting_base = predicting_base[predicting_base['Index']>rd_date-training_window] 
        # risk free
        risk_free_rate = predicting_base['tresuary_bills_60'].mean() 


    
        
    elif method == 'capm':
        prediction = []
        for i in range(len(return_matrix.columns)):
            symbol = return_matrix.columns[i]
            predicting_base = model_input[model_input['Ticker']==symbol]
            predicting_base = predicting_base[predicting_base['Index']<=rd_date] 
            predicting_base = predicting_base[predicting_base['Index']>rd_date-training_window] 
            # risk free
            risk_free_rate = predicting_base['tresuary_bills_60'].mean() 
            # expected market return
            expected_market_return = predicting_base['mkt_return'].mean()
            # Calculate beta
            predicting_base.reset_index(inplace=True,drop=True)
            stock_returns = predicting_base['Close'].pct_change().dropna()
            market_returns = predicting_base['mkt_return'][1:]
            # Covariance of the stock returns with the market returns
            covariance = np.cov(stock_returns, market_returns)[0, 1]
            # Variance of the market returns
            market_variance = np.var(market_returns)
            # Calculate beta
            beta = covariance / market_variance

            # Calculate the expected return using CAPM
            expected_return = risk_free_rate + beta * (expected_market_return - risk_free_rate)
            prediction.append(expected_return)
        sample_mean = pd.Series(prediction, index=['col_{}'.format(i) for i in range(len(np.array(return_matrix)[0]))])
        sample_mean = sample_mean*252     

    cov = df.cov()*252
    W = np.array(correlation_from_covariance(cov)**2)
    
    if ct_control is None:
        ct_control = {
            'max_clique_size': max_clique_size,
            'min_clique_size': min_clique_size,
            'threshold': threshold,
            'coordination_num':coordiation_number,
            'drop_sep': drop_sep
        }
    else:
        if 'max_clique_size' not in ct_control.keys():
            ct_control['max_clique_size']=max_clique_size
        if 'min_clique_size' not in ct_control.keys():
            ct_control['min_clique_size']=min_clique_size
        if 'coordination_num' not in ct_control.keys():
            ct_control['coordination_num']=coordiation_number
        if 'drop_sep' not in ct_control.keys():
            ct_control['drop_sep']=drop_sep

    cliques, separators, peo, tree = MFCF_Forest(W,ct_control,gf_sumsquares_gen)

    J = j_LoGo(np.array(cov), cliques, separators)
    adj_matrix = pd.DataFrame(J, columns=['col_{}'.format(i) for i in range(len(J[0]))])   
    return sample_mean, cov, adj_matrix