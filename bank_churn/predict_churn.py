import numpy 
import pandas as pd
import os 

train_path = "https://raw.githubusercontent.com/dbloxham1/kaggle_projects/main/bank_churn/train.csv"
train = pd.read_csv(train_path, 
                    sep = ",")

