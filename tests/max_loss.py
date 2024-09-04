import sys
import os
import numpy as np
import matplotlib.pyplot as plt
# import package(from parent directory)
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, parent_dir)
from out_of_sample_test import generate_random_list,mfcf_test
from set_up import *
from markowirtz_networks import *

def log_transform_sharpe_ratios_2d(sharpe_ratios_2d):

    # Convert input to a numpy array if it is not already
    sharpe_ratios_2d = np.array(sharpe_ratios_2d)
    
    # Apply the log transformation with sign preservation element-wise
    transformed_sharpe_ratios_2d = np.sign(sharpe_ratios_2d) * np.log1p(np.abs(sharpe_ratios_2d))
    
    return transformed_sharpe_ratios_2d

def max_drawdown_det(rd_date, in_sample_return_matrix, 
                      out_sample_return_matrix, model_input,
                        lmda,training_window = 30,
                        ct_control = None,                 
                        in_sample=False,
                        method='mean'):
    # define sum up list

    print(rd_date)
    in_sample_mean,cov,adj_matrix = generate_cov_mfcf(in_sample_return_matrix, model_input,
                                                                    rd_date, training_window,
                                                                    ct_control=ct_control,
                                                                    method=method)
    

    mean = np.array(in_sample_mean)
    a = np.ones(len(adj_matrix)).T.dot(adj_matrix.dot(np.ones(len(adj_matrix))))
    b = np.ones(len(adj_matrix)).T.dot(adj_matrix.dot(mean))
    #lmda = (mean.mean()-b/a)/(mean.T.dot(adj_matrix.dot(mean))-b**2/a)
    #lmda = lambda
    gamma = (1-lmda*b)/a
    optmised_weight = adj_matrix.dot(mean*lmda+gamma*np.ones(len(mean)))
    normalised_weights = optmised_weight / np.sum(optmised_weight)
    weights = np.array(normalised_weights)

    if in_sample==True:
    # out of sample mean and cov(therefore ret and var)
        df = in_sample_return_matrix
    else:
        df = out_sample_return_matrix

    max_drawdown = np.inf
    # Calculate portfolio returns for each timestamp
    portfolio_returns = np.dot(df, weights)
    max_drawdown = portfolio_returns.min()
    return portfolio_returns


training_window_list = [30]
testing_window_list = [21]
num_iter=100
return_rate_matrix,model_input = generate_nasdaq()

in_sample_return_matrix_list, out_sample_return_matrix_list, random_date_list = generate_random_list(return_rate_matrix, model_input, training_window_list, testing_window_list, num_iter=num_iter,sample_size=20)

method = 'std'
lmda=0.1
max_clique_size_list = [2,5,7,10,15,20]#,50]
random_date_assigned = np.arange(model_input['Index'].min(),model_input['Index'].max(),20)
#iteration_range_mfcf = np.arange(-0.05,0.05,0.001) # for small scale
iteration_range_mfcf = np.arange(-1,3,0.05) # for large scale
averages = {}
ct_control = {
'max_clique_size': 2,
'min_clique_size': 1,
'threshold': 0.00,
'coordination_num':np.inf,
'drop_sep': False
}

raw = {}
for method in method_list:
    for max_clique_size in max_clique_size_list:
        results = {}
        ct_control['max_clique_size'] = max_clique_size
        print('=======================',max_clique_size)

        for _ in range(n): # number of trials
            # generate data for this test
            in_sample_return_matrix_list, out_sample_return_matrix_list, random_date_list = generate_random_list(return_rate_matrix, model_input, training_window_list, testing_window_list, num_iter=num_iter,sample_size=20,random_date_assigned=random_date_assigned)

            for rd_date in random_date_assigned:
                if (training_window,rd_date) in in_sample_return_matrix_list:
                    in_sample_data = in_sample_return_matrix_list[(training_window,rd_date)]
                    out_sample_data = out_sample_return_matrix_list[(testing_window,rd_date)]
                    #try:
                    max_drawdown = max_drawdown_det(rd_date, in_sample_data, 
                                        out_sample_data, model_input,
                                            lmda,
                                            ct_control = ct_control,                 
                                            in_sample=False,
                                            method=method)
                
                    if (method,training_window,testing_window, rd_date) not in results:
                        results[(method,training_window,testing_window, rd_date)] =  []
                    results[(method,training_window,testing_window, rd_date)].append(max_drawdown)  
                    #except:
                    #    pass
        raw[(method,max_clique_size)]=results
        for key, values in results.items():
            averages[key[0],key[2],max_clique_size] = {
                'avg_max_drawdown': sum(values['max_drawdown']) / n,
            }  

for max_clique_size in max_clique_size_list:
    print('=======================',max_clique_size)
    ret_result = []
    var_result = []
    date_list = []
    for rd_date in random_date_assigned:
        if (rd_date,max_clique_size) in averages:
            ret_result.append(averages[(rd_date,max_clique_size)]['avg_ret'])
            var_result.append(averages[(rd_date,max_clique_size)]['avg_var'])
            date_list.append(rd_date)
    sharpe_ratio_result = [ret_result[j]/var_result[j] for j in range(len(ret_result))]
    normalized_sr = log_transform_sharpe_ratios_2d(sharpe_ratio_result)
    label = 'max clique size=%d'%max_clique_size
    plt.plot(date_list,normalized_sr,label=label)

title = 'max loss, sharpe vs time, out, market=nasdaq'
plt.title(title)
plt.xlabel('time')
plt.ylabel('sharpe ratio')
plt.legend()
plt.show()