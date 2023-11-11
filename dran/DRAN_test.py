import os
import torch
import pandas as pd
import numpy as np
from DRAN import DRAN


BASE_PATH = '../DPLAN/data.nosync/preprocessed/UNSW-NB15/'
TEST_PATH = os.path.join(BASE_PATH,'test_for_all.csv')

# you can play with the config
config = {
    'batch_size': 32,
    'lr': 1e-4, 
    'sad_lr': 1e-3,
    'validation_step' : 100,
    'update_step' : 1,
}
c = None # you need to pre compute the c 


if not os.path.exists('results_sad_no_score.csv'):
    with open('results_sad_no_score.csv','w') as f:
        f.write('version,dataset,subset,pr_mean,pr_std,roc_mean,roc_std\n')


data_list = ['Analysis','Backdoors','DoS','Exploits','Fuzzers','Generic','Reconnaissance']

dran = None
means=[]
num_runs = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for x in data_list:
    outs_pr = []
    outs_roc = []
    for i in range(num_runs):
        train_path = os.path.join(BASE_PATH,x+'_0.02_60.csv')
        test_path = os.path.join(BASE_PATH,'test_for_all.csv')
        train_labels = pd.read_csv(os.path.join(BASE_PATH,x+'_0.02_60.csv')).values
        X = pd.read_csv(train_path).values
        test = pd.read_csv(test_path).values
        dran = DRAN(
            train_set=X.copy(),
            test_set=test.copy(),
            config=config,
            device=device,
            c = c
        )
        dran.train(n_epochs=5,n_epochs_sad=20) # play with the epochs

        out,roc,ms = dran.test_final()
        means.append(ms)
        outs_pr.append(out)
        outs_roc.append(roc)
        print(out,roc)
        print(np.unique(np.array(dran.relabeling_accuracy),return_counts=True))
        print("-----------------------------")
    # print the mean of the results
    print(f'mean pr :{np.mean(outs_pr)}')
    print(f'mean roc: {np.mean(outs_roc)}')
    # print the std of the results
    print(f'std pr :{np.std(outs_pr)}')
    print(f'std roc: {np.std(outs_roc)}')
    # save to results.csv
    with open('results_sad_no_score.csv', 'a') as f:
        f.write(f'regression,UNSW-NB15,{x},{np.mean(outs_pr)},{np.std(outs_pr)},{np.mean(outs_roc)},{np.std(outs_roc)}\n')
    