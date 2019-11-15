
import pandas as pd
from pathlib import Path
import os

def get_ori_data():
    
    current_path=os.getcwd()
    data_path=os.path.join(current_path+"/data")

    train_full_path=os.path.join(data_path+"/train.csv")
    train_labels_full_path=os.path.join(data_path+"/train_labels.csv")
    spec_full_path=os.path.join(data_path+'/specs.csv')
    test_full_path=os.path.join(data_path+'/test.csv')
    ss_full_path=os.path.join(data_path+'/sample_submission.csv')


    train=pd.read_csv(train_full_path,sep=',')
    train_label=pd.read_csv(train_labels_full_path,sep=',')
    spec=pd.read_csv(spec_full_path,sep=',')
    test=pd.read_csv(test_full_path,sep=',')
    ss=pd.read_csv(ss_full_path,sep=',')

    return train,train_label,spec,test,ss