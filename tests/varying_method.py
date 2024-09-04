import sys
import os
import numpy as np
import matplotlib.pyplot as plt
# import package(from parent directory)
# Get the path to the previous directory
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, parent_dir)
from out_of_sample_test import generate_random_list,mfcf_test
from set_up import *
from markowirtz_networks import *


# setting parameter
training_window_list = [30]
testing_window_list = [21]
method_list = ['mean','std','ema','std_o','capm',]
num_iter=100
sample_size = 20
max_clique_size_list = [2,3,5,7,10,15,20]#,50]
#iteration_range_mfcf = np.arange(-0.05,0.05,0.001) # for small scale
iteration_range_mfcf = np.arange(-1,3,0.05) # for large scale
save_result = False
# generate dataset
return_rate_matrix,model_input = generate_nasdaq() 
in_sample_return_matrix_list, out_sample_return_matrix_list, random_date_list = generate_random_list(return_rate_matrix, model_input, training_window_list, testing_window_list, num_iter=num_iter,sample_size=sample_size)
ct_control = {
'max_clique_size': 2,
'min_clique_size': 1,
'threshold': 0.00,
'coordination_num':np.inf,
'drop_sep': False
}

for method in method_list:
    for testing_window in testing_window_list:
        sum_var = [] # result list of variance
        sum_ret = [] # result list of return
        for training_window in training_window_list:

            variance_list_sum_mfcf_for_different_clique,return_list_sum_mfcf_for_different_clique=[],[] # result list for this combination of training date, testing date and method
            for max_clique_size in max_clique_size_list:

                ct_control['max_clique_size'] = max_clique_size
                print(ct_control['max_clique_size'])

                variance_list_sum_mfcf,return_list_sum_mfcf = mfcf_test(random_date_list,in_sample_return_matrix_list, 
                                                                    out_sample_return_matrix_list,
                                                                    model_input,iteration_range_mfcf,
                                                                    ct_control=ct_control,training_window =training_window, 
                                                                    testing_window=testing_window,given_return=False,in_sample=False,method=method)
                variance_list_sum_mfcf_for_different_clique.append(variance_list_sum_mfcf)
                return_list_sum_mfcf_for_different_clique.append(return_list_sum_mfcf)
                print('=======================',max_clique_size)

            sum_var.append(variance_list_sum_mfcf_for_different_clique)
            sum_ret.append(return_list_sum_mfcf_for_different_clique)

        sum_var = np.array(sum_var)
        sum_ret = np.array(sum_ret)

        # plotting
        colors = plt.cm.viridis(np.linspace(0, 1, len(max_clique_size_list)))
        lambda_range = iteration_range_mfcf # for small scale # 100 points
        cliuqe_size_list = max_clique_size_list

        for training_index in range(len(sum_var)):
            variance_list_sum_mfcf_for_different_clique, return_list_sum_mfcf_for_different_clique=sum_var[training_index],sum_ret[training_index]
            for i in range(len(variance_list_sum_mfcf_for_different_clique)):
                variance_list_sum_ranged_mfcf = np.mean(variance_list_sum_mfcf_for_different_clique[i], axis=0)
                return_list_sum_ranged_mfcf = np.mean(return_list_sum_mfcf_for_different_clique[i], axis=0)
                sharpe_ratio_sum_ranged_mfcf = [return_list_sum_ranged_mfcf[j]/variance_list_sum_ranged_mfcf[j] for j in range(len(return_list_sum_ranged_mfcf))]
                label = 'max clique size=%d'%cliuqe_size_list[i]
                plt.plot(lambda_range,return_list_sum_ranged_mfcf,label=label,color=colors[i])
            title = 'λ vs var, OUT, training= %s,testing=%d, method=%s,market=nasdaq'% (training_window_list[training_index],testing_window,method)
            plt.title(title)
            plt.xlabel('lambda')
            plt.ylabel('sharpe ratio')
            plt.legend()
            #plt.savefig("%s/%s %s.png"%(directory_path,title,datetime.now().strftime("%Y%m%d_%H%M%S"))) # You can specify the format by changing the file extension (e.g., .pdf, .jpg, .svg)
            plt.show()

        if save_result==True:
            #make directory
            directory_path = 'method=%s,λ vs SR,OUT, testing = %d'%(method,testing_window)
            if not os.path.isdir(directory_path):
                os.makedirs(directory_path)
            # name of file
            title = 'λ vs ret, out of sample_training size = %s,testing size=%d,method=%s'% (str(training_window_list),testing_window,method)
            # Flatten the 4D array to convert it into a 2D array
            flattened_array = sum_var.reshape(-1, sum_var.shape[-1])
            # Save the flattened array to a text file
            np.savetxt('%s/%s sum_var %s.txt'%(directory_path,title,str(sum_var.shape)), flattened_array)
            flattened_array = sum_ret.reshape(-1, sum_ret.shape[-1])
            np.savetxt('%s/%s sum_ret %s.txt'%(directory_path,title,str(sum_ret.shape)), flattened_array)