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

# function to plot out the weight
def weight_dis(weight,bin=1000,title=None,xrange=None):
    # distribution of weights
    data = pd.Series(np.array(weight))
    if title:
        data.plot(kind='hist', bins=bin, alpha=0.7, title=title)
    else:
        data.plot(kind='hist', bins=bin, alpha=0.7,)# title='distribution of weights(xx data)')
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    if xrange:
        plt.xlim(xrange[0], xrange[1])
    plt.show()

def weight_test(random_date_list, in_sample_return_matrix_list, 
                model_input,
              ct_control = None, 
              training_window =1000, 
            method='mean',lmda = 0.1):
    # define sum up list
    weights_sum = []
    for rd_index in range(len(random_date_list)):
        rd_date = random_date_list[rd_index]

        in_sample_return_matrix = in_sample_return_matrix_list[(training_window,rd_date)]

        in_sample_mean,cov,adj_matrix = generate_cov_mfcf(in_sample_return_matrix, model_input,
                                                                     rd_date, training_window,
                                                                     ct_control=ct_control,
                                                                     method=method)
        

        mean = np.array(in_sample_mean)
        a = np.ones(len(adj_matrix)).T.dot(adj_matrix.dot(np.ones(len(adj_matrix))))
        b = np.ones(len(adj_matrix)).T.dot(adj_matrix.dot(mean))
        #lmda = (mean.mean()-b/a)/(mean.T.dot(adj_matrix.dot(mean))-b**2/a)
        #lmda = i
        gamma = (1-lmda*b)/a
        optmised_weight = adj_matrix.dot(mean*lmda+gamma*np.ones(len(mean)))
        normalised_weights = optmised_weight / np.sum(optmised_weight)
        weights = np.array(normalised_weights)
        weights_sum.append(weights)
  
    return weights_sum


# setting up parameters
ct_control = {
'max_clique_size': 2,
'min_clique_size': 1,
'threshold': 0.00,
'coordination_num':np.inf,
'drop_sep': False
}
all_weights = []
training_window = 30
method = 'mean'
lmda = 0.1
max_clique_size_list = [2,3,5,7,10,15,20]#,50]
sample_size = 20
num_iter=100
return_rate_matrix,model_input = generate_nasdaq() 
in_sample_return_matrix_list, out_sample_return_matrix_list, random_date_list = generate_random_list(return_rate_matrix, model_input, training_window_list, testing_window_list, num_iter=num_iter,sample_size=sample_size)
weights_sum = []


for max_clique_size in max_clique_size_list:
    ct_control['max_clique_size'] = max_clique_size
    print(ct_control['max_clique_size'])

    weight_sum = weight_test(random_date_list, in_sample_return_matrix_list, 
                model_input,
              ct_control = ct_control, 
              training_window =training_window, 
            method=method,lmda = lmda)
    
    all_weights.append(np.array(weight_sum).flatten())


# Create box plot
plt.figure(figsize=(8, 6))
plt.boxplot(all_weights, flierprops=dict(marker=''))
plt.xlabel('Max Clique size')
plt.ylabel('Weights')
plt.ylim(-7.5, 7.5)
plt.xticks([1, 2, 3, 4,5], ['2', '5', '7', '10', '15'])
plt.show()
