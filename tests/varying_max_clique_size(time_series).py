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


def performance_daily(rd_date, in_sample_return_matrix, 
                      out_sample_return_matrix, model_input,
                        lmda,
                        ct_control = None,                 
                        in_sample=False,
                        method='mean'):

    print(rd_date)
    in_sample_mean,cov,adj_matrix = generate_cov_mfcf(in_sample_return_matrix, model_input,
                                                                    rd_date, training_window,
                                                                    ct_control=ct_control,
                                                                    method=method)
    

    mean = np.array(in_sample_mean)
    a = np.ones(len(adj_matrix)).T.dot(adj_matrix.dot(np.ones(len(adj_matrix))))
    b = np.ones(len(adj_matrix)).T.dot(adj_matrix.dot(mean))
    gamma = (1-lmda*b)/a
    optmised_weight = adj_matrix.dot(mean*lmda+gamma*np.ones(len(mean)))
    normalised_weights = optmised_weight / np.sum(optmised_weight)
    weights = np.array(normalised_weights)

    if in_sample==True:
    # out of sample mean and cov(therefore ret and var)
        df = in_sample_return_matrix
    else:
        df = out_sample_return_matrix

    mean =df.mean()*252
    cov = df.cov()*252
    ret = mean.dot(weights)
    vol = np.sqrt(weights.T.dot(cov.dot(weights)))
    return ret, vol

# daily_performance ——> input(rd_date, in, out, mi, lmda), output(var,ret)
# generate_random_list ——> input(assigned_date, return_m, mi,tw,)
#                          output(in_sample_dict, out_of_sample_dict)
training_window_list = [30]
testing_window_list = [21]
num_iter=100
return_rate_matrix,model_input = generate_nasdaq() 
method = 'std'
n=100
lmda=0.1
max_clique_size_list = [2,3,5,7,10,15,20]#,50]
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
for max_clique_size in max_clique_size_list:
    results = {}
    ct_control['max_clique_size'] = max_clique_size
    print('=======================',max_clique_size)

    for _ in range(n): # number of trials
        # generate data for this test
        in_sample_return_matrix_list, out_sample_return_matrix_list, random_date_list = generate_random_list(return_rate_matrix, model_input, training_window_list, testing_window_list, num_iter=num_iter,sample_size=30,random_date_assigned=random_date_assigned)

        for rd_date in random_date_assigned:
            if (training_window,rd_date) in in_sample_return_matrix_list:
                in_sample_data = in_sample_return_matrix_list[(training_window,rd_date)]
                out_sample_data = out_sample_return_matrix_list[(testing_window,rd_date)]
                try:
                    ret, var = performance_daily(rd_date, in_sample_data, 
                            out_sample_data, model_input,
                                lmda,
                                ct_control = ct_control,                 
                                in_sample=False,
                                method=method)
                
                    if (training_window,testing_window, rd_date) not in results:
                        results[(training_window,testing_window, rd_date)] = {'var': [], 'ret': []}
                    results[(training_window,testing_window, rd_date)]['var'].append(var)
                    results[(training_window,testing_window, rd_date)]['ret'].append(ret)     
                except:
                    pass
    
    for key, values in results.items():
        averages[key[2],max_clique_size] = {
            'avg_var': sum(values['var']) / n,
            'avg_ret': sum(values['ret']) / n
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

    label = 'max clique size=%d'%max_clique_size
    plt.plot(date_list,var_result,label=label)


title = 'sharpe vs time, OUT, ,market=nasdaq'
plt.title(title)
plt.xlabel('time')
#plt.ylabel('var')
plt.ylabel('sharpe ratio')
plt.legend()
#plt.savefig("%s/%s %s.png"%(directory_path,title,datetime.now().strftime("%Y%m%d_%H%M%S"))) # You can specify the format by changing the file extension (e.g., .pdf, .jpg, .svg)
plt.show()