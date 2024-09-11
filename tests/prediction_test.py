import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import os
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, f1_score

# load model and input
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
model_path = os.path.join(parent_dir, 'rf.sav')
loaded_model = pickle.load(open(model_path, 'rb'))

model_input_data = os.path.join(parent_dir, 'model_input_nasdaq.csv')
model_input = pd.read_csv(model_input_data)

X = model_input[['call_put_ratio_200', # call put raio over the past 200 days
                'SQZ', 
                'MACD',
                'vix_fix_gauge', # fear index
                'Index', # representing the time series
                'Greater_than_MA125', 
                'Ticker_label', # An arbitary label for ticker, as no categorial/string feature allowed when training
                'Close', # closing price
                'market_sum' # the sum of the close price for the whole market on the day
                                ]]
y = [True if model_input['if_profit_21'][i]==1 else False for i in range(len(model_input))]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# accuracy
y_pred = loaded_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# Calculate MCC
mcc = matthews_corrcoef(y_test, y_pred)
print(f'Matthews Correlation Coefficient: {mcc}')

# Calculate F1 Score
f1 = f1_score(y_test, y_pred)
print(f'F1 Score: {f1}')

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
# Calculate the True Positive Rate (TPR)
# cm[1,1] is the number of true positives
# cm[1,:].sum() is the number of actual positives (true positives + false negatives)
TPR = cm[1, 1] / cm[1, :].sum()
print(f'True Positive Rate: {TPR}')