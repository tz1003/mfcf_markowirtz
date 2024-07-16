import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
np.random.seed(123)

def generate_nasdaq():
    # load data
    stock_data = pd.read_csv('NASDAQ_100_MCap.csv')

    # list out all the symbol
    symbol_list = [stock_data.columns[i][:-10] for i in range(len(stock_data.columns))][1:]
    df = stock_data.copy()
    df.drop(['date'],axis=1,inplace=True)
    # Calculate the daily return rate using vectorized operations
    return_rate_matrix = df / df.shift(1) - 1
    # Remove the first row as it will be NaN due to the shift operation
    return_rate_matrix = return_rate_matrix.iloc[1:]
    return_rate_matrix.columns = symbol_list
    return_rate_matrix.insert(loc=0,column = 'date',value = stock_data['date'].tolist()[1:])
    return_rate_matrix.reset_index(inplace=True,drop=True)

    # remove market closed date
    drop_list = []
    for i in range(len(return_rate_matrix)):
        if return_rate_matrix['MSFT'][i]==0:
            drop_list.append(i)

    for i in drop_list:
        if return_rate_matrix['CSCO'][i]!=0:
            drop_list.remove(i)
        
    return_rate_matrix.drop(drop_list,axis = 0,inplace=True)
    return_rate_matrix.insert(loc=0,column = 'Index',value = np.arange(len(return_rate_matrix)))
    return_rate_matrix.reset_index(drop=True,inplace=True)

    # discarding the depedent stocks
    return_rate_matrix.drop(['DISCB','DISCK','LBTYK','UAL','LBTYA','LBTYB','MASI'],axis=1,inplace=True)


    # remove market closed date for BL
    drop_list = []
    for i in range(len(stock_data)):
        if stock_data['date'][i] not in return_rate_matrix['date'].tolist():
            drop_list.append(i)

    stock_data.drop(drop_list,axis = 0,inplace=True)
    stock_data.reset_index(drop=True,inplace=True)

    # input data for prediction of model
    model_input = pd.read_csv('model_input_nasdaq.csv')
    # Drop rows where Ticker is 'LBTYB'
    model_input = model_input[model_input['Ticker'] != 'LBTYB']
    model_input.reset_index(drop=True,inplace=True)
    model_input = model_input[['Volume', 'call_put_ratio_200',  'SQZ',
        'MACD', 'vix_fix_gauge',  'Greater_than_MA99',#'mkt_cap',
        'Greater_than_MA125','Index','Ticker_label','Close','Ticker',
        'tresuary_bills_60', 'mkt_return']]


    # select column present in the prediction
    available_column = list(set(model_input['Ticker'].tolist()))+['Index','date']
    # Filter out column names that exist in both DataFrame and list A
    selected_columns = [col for col in available_column if col in return_rate_matrix.columns]
    # Select the DataFrame with the filtered columns
    return_rate_matrix = return_rate_matrix[selected_columns]

    return return_rate_matrix,model_input

def generate_sp():
    # directly from prediction data
    # input data for prediction of model
    predicting = pd.read_csv('sumup_with_ticker_0701_sp.csv')
    predicting = predicting[predicting['Index']>846]
    predicting.reset_index(inplace=True,drop=True)
    ticker_list = predicting['Ticker'].unique().tolist()
    stock_data = pd.DataFrame({'time':np.sort(predicting.time.unique())})

    for ticker in ticker_list:
        temp = predicting[predicting['Ticker']==ticker][['time','Close']]
        temp = temp.rename(columns={'Close':ticker})
        stock_data = stock_data.merge(temp,how='left',left_on='time',right_on='time')

    df = stock_data.copy()
    df.drop(['time'],axis=1,inplace=True)
    # Calculate the daily return rate using vectorized operations
    return_rate_matrix = df / df.shift(1) - 1
    # Remove the first row as it will be NaN due to the shift operation
    return_rate_matrix = return_rate_matrix.iloc[1:]
    return_rate_matrix.insert(loc=0,column = 'time',value = stock_data['time'].tolist()[1:])
    return_rate_matrix.drop(['DOW','PYPL','KHC','GOOG'],axis=1,inplace=True)
    #return_rate_matrix.dropna(inplace=True)
    return_rate_matrix.reset_index(inplace=True,drop=True)

    idx_matching = predicting[['Index','time']]
    idx_matching.drop_duplicates(inplace=True)
    #predicting = predicting.merge(originalmatch,how='inner',left_on=['time'],right_on=['date'])
    predicting = predicting[['Volume', 'call_put_ratio_200',  'SQZ',
        'MACD', 'vix_fix_gauge',  'Greater_than_MA99',#'mkt_cap',
        'Greater_than_MA125','Index','Ticker_label','Close','Ticker',
        #'tresuary_bills_60', 'mkt_return'
        ]]
    return_rate_matrix = return_rate_matrix.merge(idx_matching,how='left',left_on='time',right_on='time')
    return_rate_matrix = return_rate_matrix.rename(columns={'time':'date'})

    return return_rate_matrix

def generate_chinese():
    # directly from prediction data
    # input data for prediction of model
    predicting = pd.read_csv('sumup_with_ticker_0701_sh.csv')
    predicting = predicting[predicting['Index']>846]
    predicting.reset_index(inplace=True,drop=True)
    ticker_list = predicting['Ticker'].unique().tolist()
    stock_data = pd.DataFrame({'time':np.sort(predicting.time.unique())})

    for ticker in ticker_list:
        temp = predicting[predicting['Ticker']==ticker][['time','Close']]
        temp = temp.rename(columns={'Close':ticker})
        stock_data = stock_data.merge(temp,how='left',left_on='time',right_on='time')

    df = stock_data.copy()
    df.drop(['time'],axis=1,inplace=True)
    # Calculate the daily return rate using vectorized operations
    return_rate_matrix = df / df.shift(1) - 1
    # Remove the first row as it will be NaN due to the shift operation
    return_rate_matrix = return_rate_matrix.iloc[1:]
    return_rate_matrix.insert(loc=0,column = 'time',value = stock_data['time'].tolist()[1:])
    #return_rate_matrix.drop(['DOW','PYPL','KHC','GOOG'],axis=1,inplace=True)
    #return_rate_matrix.drop(drop_list,axis=1,inplace=True)
    #return_rate_matrix.dropna(inplace=True)
    return_rate_matrix.reset_index(inplace=True,drop=True)
    return_rate_matrix  = return_rate_matrix[return_rate_matrix.columns[:-35].tolist()]
    return_rate_matrix.drop([601633,601012,601336,601360,601901,601238],axis=1,inplace=True)
    return_rate_matrix = return_rate_matrix.drop(return_rate_matrix.index[:5])
    return_rate_matrix = return_rate_matrix.fillna(method='ffill')

    idx_matching = predicting[['Index','time']]
    idx_matching.drop_duplicates(inplace=True)
    #predicting = predicting.merge(originalmatch,how='inner',left_on=['time'],right_on=['date'])
    predicting = predicting[['Volume', 'call_put_ratio_200',  'SQZ',
        'MACD', 'vix_fix_gauge',  'Greater_than_MA99',#'mkt_cap',
        'Greater_than_MA125','Index','Ticker_label','Close','Ticker',
        #'tresuary_bills_60', 'mkt_return'
        ]]
    return_rate_matrix = return_rate_matrix.merge(idx_matching,how='left',left_on='time',right_on='time')
    return_rate_matrix = return_rate_matrix.rename(columns={'time':'date'})

    return return_rate_matrix

# generate random trading date
def generate_random_day(return_rate_matrix,training_window=1000, testing_window=300):
    data_size = len(return_rate_matrix)
    random_date = random.choice(return_rate_matrix['Index'])
    #must fit the training and testing window
    while (random_date<training_window) or (random_date>(data_size-testing_window)):
        random_date = random.choice(return_rate_matrix['Index'])
    #print('training:%d - %d'%(random_date-training_window,random_date))
    #print('testing:%d - %d'%(random_date,random_date+testing_window)) 
    return random_date

# generate dataset
def generate_dataset(return_rate_matrix,training_window=1000, 
                     testing_window=300,rdm_day = None,sample_size=100,selected_columns=None):

    # random date
    if not rdm_day:
        random_date = generate_random_day(return_rate_matrix,training_window,testing_window)
    else:
        random_date = rdm_day

    #print('training:%d - %d'%(random_date-training_window,random_date))
    #print('testing:%d - %d'%(random_date,random_date+testing_window)) 
    # generate training data
    training_data = return_rate_matrix[return_rate_matrix['Index']<=random_date]
    training_data = training_data[training_data['Index']>(random_date-training_window)]
    training_data.reset_index(drop =True, inplace=True)
    training_data = training_data.drop(['Index','date'],axis=1)
    # generate testing data
    testing_data = return_rate_matrix[return_rate_matrix['Index']>random_date]
    testing_data = testing_data[testing_data['Index']<=(random_date+testing_window)]
    testing_data.reset_index(drop =True, inplace=True)
    testing_data = testing_data.drop(['Index','date'],axis=1)

    # Randomly select 100 tickers
    columns = [col for col in return_rate_matrix.columns if col not in ["Index", "date"]]
    if selected_columns:
        training_data = training_data.loc[:, selected_columns]
        testing_data = testing_data.loc[:, selected_columns]
    else:
        random_tickers = random.sample(columns, sample_size)
        training_data = training_data.loc[:, random_tickers]
        testing_data = testing_data.loc[:, random_tickers]

    return training_data, testing_data


# verify postive definite and full rank
def test_full_and_pd(matrix):
    cov_matrix = matrix.cov()
    rk = np.linalg.matrix_rank(cov_matrix)
    W = correlation_from_covariance(cov_matrix)
    # Calculate eigenvalues
    eigenvalues = np.linalg.eigvals(W)
    # Check if all eigenvalues are positive
    is_positive_definite = np.all(eigenvalues > 0)
    print('full rank:',rk==len(cov_matrix))
    print("Is Positive Definite:", is_positive_definite)

# function to plot out the weight
def weight_dis(weight,bin=1000):
    # distribution of weights
    data = pd.Series(np.array(weight))
    data.plot(kind='hist', bins=bin, alpha=0.7, title='distribution of weights(xx data)')
    plt.xlabel('weight')
    plt.ylabel('Frequency')
    plt.show()

def avg_variance_for_return(target_return, data):
    """
    find the average variance for a given return
    """

    # Sort the data by return
    data.sort(key=lambda x: x[1])

    for i in range(1, len(data)):
        if data[i-1][1] <= target_return <= data[i][1]:
            return (data[i-1][0] + data[i][0]) / 2

    # If no suitable return rate is found in the data, return None
    return None
