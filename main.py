from util import hyper,write_results
from Env import ADEnv
from DPLAN import DPLAN
import torch
import os
import pandas as pd


BASE_PATH = './data.nosync/preprocessed'

subsets = {
    'UNSW-NB15' : ['Fuzzers','Analysis','Backdoors','DoS','Exploits','Generic','Reconnaissance'],
}
datasets = subsets.keys()
TEST_NAME = 'test_for_all.csv'
VALIDATION_NAME = 'validation_for_all.csv'
LABEL_NORMAL = 0
LABEL_ANOMALY = 1
CONTAMINATION_RATE  = hyper['contamination_rate']
NUM_ANOMALY_KNOWS = hyper['num_anomaly_knows']
NUM_RUNS = hyper['runs']

MODELS_PATH = 'models/'
RESULTS_PATH = 'results'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results_filename = os.path.join(RESULTS_PATH, 'results.csv')
if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)
with open(results_filename, 'w') as f:
        f.write('dataset,subset,pr_mean,pr_std,roc_mean,roc_std\n')


for dataset in datasets:
    test_path = os.path.join(BASE_PATH, dataset, TEST_NAME)
    test_set = pd.read_csv(test_path).values

    for subset in subsets[dataset]:
        data_path = os.path.join(BASE_PATH, dataset, subset)+f'_{CONTAMINATION_RATE}_{NUM_ANOMALY_KNOWS}.csv'
        training_set = pd.read_csv(data_path).values

        pr_auc_history = []
        roc_auc_history = []

        for i in range(NUM_RUNS):
            print(f'Running {dataset} {subset} {i}...')
            model_id = f'_{CONTAMINATION_RATE}_{NUM_ANOMALY_KNOWS}_run_{i}'
            
            env = ADEnv(
                dataset=training_set,
                sampling_Du=hyper['sampling_du'],
                prob_au=hyper['prob_au'],
                label_normal=LABEL_NORMAL,
                label_anomaly=LABEL_ANOMALY
            )

            dplan = DPLAN(
                env=env,
                validation_set=None,
                test_set=test_set,
                destination_path=MODELS_PATH,
                c = c,
                double_dqn=False
            )
            dplan.fit(reset_nets = True)
            dplan.show_results()
            roc,pr = dplan.model_performance(on_test_set=True)
            print(f'Finished run {i} with pr: {pr} and auc-roc: {roc}...')
            pr_auc_history.append(pr)
            roc_auc_history.append(roc)

            destination_filename =subset+'_'+model_id + '.pth'
            dplan.save_model(destination_filename)
            print()
            print('--------------------------------------------------\n')

        print(f'Finished {dataset} {subset}...')
        print('--------------------------------------------------\n')
        write_results(pr_auc_history,roc_auc_history,dataset,subset,results_filename)

