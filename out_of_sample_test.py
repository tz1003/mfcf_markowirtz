import numpy as np
from markowirtz_networks import *
from set_up import *

def check_pdf(matrix):
    cov_matrix = matrix.cov()
    np.linalg.matrix_rank(cov_matrix)
    W = correlation_from_covariance(cov_matrix)
    #cov_matrix.columns[128]
    #W = correlation_from_covariance(cov_matrix)
    #W.iloc[128,128]

    #for i in range(len(W)):
    #    for j in range(len(W.columns)):
    #        if math.isclose(W.iloc[i,j], 1, abs_tol=1e-2):
    #            print(i,j)
    # Calculate eigenvalues
    eigenvalues = np.linalg.eigvals(W)

    # Check if all eigenvalues are positive
    return np.all(eigenvalues > 0)

def check_nan(matrix):
    if len(matrix.columns)  == len(matrix.dropna(axis=1).columns):
        return True
    else:
        return False

def check_prediction(model_input, rd_date,selected_columns):
    for col in selected_columns:
        temp = model_input[model_input['Ticker']==col]
        temp = temp[temp['Index']==rd_date] 
        if len(temp)==0:
            return False
    return True


def generate_random_list_largest_var(return_rate_matrix, model_input, training_window_list, testing_window_list, num_iter=10,sample_size=30):
    ct = 0
    training_window = max(training_window_list)
    testing_window = max(testing_window_list)
    random_date_list = []
    in_sample_return_matrix_list = {}
    out_sample_return_matrix_list = {}
    excluding_list = [1198,1219,1449,1471,1699,1723,1950,2200,2225,2455,2475]
    while True:
        # generate random date
        rd_date = generate_random_day(return_rate_matrix,training_window=training_window, testing_window=testing_window)
        if rd_date > 1108+training_window and rd_date <2488-testing_window and rd_date not in excluding_list:
            passing_label = 1
            # Randomly select 20 columns
            columns = [col for col in return_rate_matrix.columns if col not in ["Index", "date"]]

            for training_window in training_window_list:
                for testing_window in testing_window_list:
                    in_sample_return_matrix, out_sample_return_matrix = generate_dataset(
                        return_rate_matrix,
                        training_window=training_window, 
                        testing_window=testing_window,
                        rdm_day=rd_date,
                        selected_columns=columns
                    )
                    
                    # Calculate variance for each column and select top columns with highest variance
                    variance = in_sample_return_matrix.var()
                    selected_columns = variance.nlargest(sample_size).index.tolist()                   
                    # Generate datasets again with selected columns
                    in_sample_return_matrix, out_sample_return_matrix = generate_dataset(
                        return_rate_matrix,
                        training_window=training_window, 
                        testing_window=testing_window,
                        rdm_day=rd_date,
                        selected_columns=selected_columns
                    )
                    #print(check_pdf(in_sample_return_matrix)==True and check_pdf(out_sample_return_matrix)==True,training_window,testing_window)
                    if not (check_pdf(in_sample_return_matrix)==True and check_pdf(out_sample_return_matrix)==True and check_nan(in_sample_return_matrix)==True and check_nan(out_sample_return_matrix)==True and check_prediction(model_input,rd_date,selected_columns)==True):
                        passing_label=0
            #print(passing_label)
            if passing_label==1:
                for training_window in training_window_list:
                    for testing_window in testing_window_list:
                        in_sample_return_matrix, out_sample_return_matrix = generate_dataset(return_rate_matrix,
                                                training_window=training_window, 
                                                testing_window=testing_window,
                                                rdm_day = rd_date,selected_columns=selected_columns)
                        
                        in_sample_return_matrix_list[(training_window,rd_date)]=in_sample_return_matrix
                        out_sample_return_matrix_list[(testing_window,rd_date)]=out_sample_return_matrix
                random_date_list.append(rd_date)

                ct+=1
                if ct>=num_iter:
                    return in_sample_return_matrix_list, out_sample_return_matrix_list, random_date_list


    
def generate_random_list(return_rate_matrix, model_input, training_window_list, testing_window_list, num_iter=10,sample_size=30):
    ct = 0
    training_window = max(training_window_list)
    testing_window = max(testing_window_list)
    random_date_list = []
    in_sample_return_matrix_list = {}
    out_sample_return_matrix_list = {}
    excluding_list = [1198,1219,1449,1471,1699,1723,1950,2200,2225,2455,2475]
    while True:
        # generate random date
        rd_date = generate_random_day(return_rate_matrix,training_window=training_window, testing_window=testing_window)
        if rd_date > 1108+training_window and rd_date <2488-testing_window and rd_date not in excluding_list:
            passing_label = 1
            # Randomly select 20 columns
            columns = [col for col in return_rate_matrix.columns if col not in ["Index", "date"]]
            selected_columns = random.sample(columns, sample_size)
                        
            for training_window in training_window_list:
                for testing_window in testing_window_list:
                    in_sample_return_matrix, out_sample_return_matrix = generate_dataset(return_rate_matrix,
                                            training_window=training_window, 
                                            testing_window=testing_window,
                                            rdm_day = rd_date,selected_columns=selected_columns)
                    #print(check_pdf(in_sample_return_matrix)==True and check_pdf(out_sample_return_matrix)==True,training_window,testing_window)
                    if not (check_pdf(in_sample_return_matrix)==True and check_pdf(out_sample_return_matrix)==True and check_nan(in_sample_return_matrix)==True and check_nan(out_sample_return_matrix)==True and check_prediction(model_input,rd_date,selected_columns)==True):
                        passing_label=0
            #print(passing_label)
            if passing_label==1:
                for training_window in training_window_list:
                    for testing_window in testing_window_list:
                        in_sample_return_matrix, out_sample_return_matrix = generate_dataset(return_rate_matrix,
                                                training_window=training_window, 
                                                testing_window=testing_window,
                                                rdm_day = rd_date,selected_columns=selected_columns)
                        
                        in_sample_return_matrix_list[(training_window,rd_date)]=in_sample_return_matrix
                        out_sample_return_matrix_list[(testing_window,rd_date)]=out_sample_return_matrix
                random_date_list.append(rd_date)

                ct+=1
                if ct>=num_iter:
                    return in_sample_return_matrix_list, out_sample_return_matrix_list, random_date_list


def mfcf_test(random_date_list, in_sample_return_matrix_list, out_sample_return_matrix_list, model_input,
              iteration_range,
              ct_control = None, 
              training_window =1000, 
              testing_window=300,                       
              given_return=False,
            in_sample=False,
            method='mean'):
    # define sum up list
    return_list_sum = []
    variance_list_sum = []

    for rd_index in range(len(random_date_list)):
        rd_date = random_date_list[rd_index]

        in_sample_return_matrix = in_sample_return_matrix_list[(training_window,rd_date)]

        in_sample_mean,cov,adj_matrix = generate_cov_mfcf(in_sample_return_matrix, model_input,
                                                                     rd_date, training_window,
                                                                     ct_control=ct_control,
                                                                     method=method)
        

    
        return_list = []
        variance_list =[]
        for i in iteration_range:
            mean = np.array(in_sample_mean)
            a = np.ones(len(adj_matrix)).T.dot(adj_matrix.dot(np.ones(len(adj_matrix))))
            b = np.ones(len(adj_matrix)).T.dot(adj_matrix.dot(mean))
            #lmda = (mean.mean()-b/a)/(mean.T.dot(adj_matrix.dot(mean))-b**2/a)
            lmda = i
            gamma = (1-lmda*b)/a
            optmised_weight = adj_matrix.dot(mean*lmda+gamma*np.ones(len(mean)))
            normalised_weights = optmised_weight / np.sum(optmised_weight)
            weights = np.array(normalised_weights)

            if in_sample==True:
            # out of sample mean and cov(therefore ret and var)
                df = in_sample_return_matrix
            else:
                df = out_sample_return_matrix_list[(testing_window,rd_date)] 

            mean =df.mean()*252
            cov = df.cov()*252
            ret = mean.dot(weights)
            vol = np.sqrt(weights.T.dot(cov.dot(weights)))
            return_list.append(ret)
            variance_list.append(vol)
        
        return_list_sum.append(return_list)
        variance_list_sum.append(variance_list)

    if given_return == False:

        return variance_list_sum,return_list_sum

