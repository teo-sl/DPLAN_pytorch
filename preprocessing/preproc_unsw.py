# IMPORT
import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split

# GLOBAL VARIABLES

DEST_PATH = 'pat/to/dest'
DATA_PATH = 'path/to/data'
CONTAMINATION_RATE = 0.02
NUM_KNOWS = 60
D_U_SIZE = 1900


ANOMALIES = ['Generic', 'Exploits', 'Fuzzers', 'DoS', 'Reconnaissance', 'Analysis', 'Backdoors']
AT_DICT = {
                            'Normal' : 0,     
                            'Generic' : 1,     
                            'Exploits' : 2,   
                            'Fuzzers'  : 3,
                            'DoS'    : 4,
                            'Reconnaissance'  : 5,
                            'Analysis'   : 6,
                            'Backdoors'  : 7,
                        }

# PREPROCESSING

# read the features file
features_df = pd.read_csv(os.path.join(DATA_PATH,'NUSW-NB15_features.csv'),encoding='cp1252')

# get features names
features_name = features_df['Name'].tolist()


# types management
type_conversion = {}
for x in features_df.values:
    type_conversion[x[1]] = x[2]

for x in type_conversion.keys():
    value = type_conversion[x]
    if value == 'nominal':
        dst_value = 'str'
    elif value == 'integer' or value == 'Timestamp':
        dst_value = 'int64'
    elif value == 'Float':
        dst_value = 'float64'
    elif value == 'Binary' or value == 'binary':
        dst_value = 'bool'
    type_conversion[x]=dst_value



# read the data
files = glob.glob(DATA_PATH+'/*_[1-4].csv')
dfs = []
for f in files:
    dfs.append(pd.read_csv(f,names=features_name))
df = pd.concat(dfs, ignore_index=True)

# drop the useless columns
columns_to_drop = ['srcip', 'sport', 'dstip', 'dsport']
df = df.drop(columns_to_drop, axis=1)


# trim every element in attack_cat column
df['attack_cat'] = df['attack_cat'].str.strip()
# rename Backdoor to Backdoors 
df['attack_cat'] = df['attack_cat'].replace('Backdoor','Backdoors')


# get the unique values of attack_cat column || optional
# types_of_attack = df['attack_cat'].unique()

df.loc[df['Label'] == 0, 'attack_cat'] = 'Normal'


# delete unused anomalies
df_dst = df[df['attack_cat'] != 'Worms']
df_dst= df_dst[df_dst['attack_cat'] != 'Shellcode']


# this anomalies are overrepresented
to_be_sampled = ['DoS','Exploits','Fuzzers','Generic','Reconnaissance','Normal']
df_sampled = []
for attack in to_be_sampled:
    if attack == 'Normal':
        df_sampled.append(df_dst[df_dst['attack_cat'] == attack].sample(n=93_000, random_state=42))
    else:
        df_sampled.append(df_dst[df_dst['attack_cat'] == attack].sample(n=3000, random_state=42))


# get all the rows that are not in to_be_sampled list
df_not_sampled = df_dst[~df_dst['attack_cat'].isin(to_be_sampled)]

# concat all the sampled dataframes
df_sampled = pd.concat(df_sampled, ignore_index=True)

# concat the sampled and not sampled dataframes
df_sampled = pd.concat([df_sampled, df_not_sampled], ignore_index=True)


# convert to numeric values
df_sampled['ct_ftp_cmd'] = pd.to_numeric(df_sampled['ct_ftp_cmd'],errors='coerce')

# replace the missing values with the mean of the target column
df_sampled['ct_ftp_cmd'] = df_sampled['ct_ftp_cmd'].fillna(df_sampled['ct_ftp_cmd'].mean())
df_sampled['ct_flw_http_mthd'] = df_sampled['ct_flw_http_mthd'].fillna(df_sampled['ct_flw_http_mthd'].mean())
df_sampled['is_ftp_login'] = df_sampled['is_ftp_login'].fillna(df_sampled['is_ftp_login'].mean())


# one hot encoding
ohe_columns = ['proto','state','service']
df_ohe = pd.get_dummies(df_sampled, columns=ohe_columns)



# convert the attack_cat column to numeric values using the AT_DICT
df_ohe['attack_cat'] = df_ohe['attack_cat'].map(AT_DICT)
df_ohe['attack_cat'] = df_ohe['attack_cat'].astype(int)

# normalize the data between 0 and 1 except the attack_cat column
for col in df_ohe.columns:
    if col != 'attack_cat':
        df_ohe[col] = (df_ohe[col] - df_ohe[col].min()) / (df_ohe[col].max() - df_ohe[col].min())

# CREATE THE DATASET


df = df_ohe
train, test = train_test_split(df, test_size=0.2, random_state=42)

# use it for debug purpose
train['Label'].value_counts()


# TRAINING SET

df = train

df_normal = df[df['attack_cat'] == AT_DICT['Normal']]

# Adjust according to your needs
df_normal.loc[df_normal.sample(n=100, random_state=42).index, 'Label'] = 2
num_for_each_attack =int((df.shape[0]*0.02))//7

for attack_type in AT_DICT.keys():

    if attack_type == 'Normal':
        continue

    name = '{}_{}_{}.csv'.format(attack_type, CONTAMINATION_RATE, NUM_KNOWS)  
    file_name = os.path.join(DEST_PATH, name)
    
    df_attack_sample = df[df['attack_cat'] == AT_DICT[attack_type]].sample(n=NUM_KNOWS, random_state=42)
    # get all rows that are not Normal and not the attack type
    d_u = df[(df['attack_cat'] != AT_DICT['Normal'])]
    d_us = []
    for sub_attack in AT_DICT.keys():
        if sub_attack=='Normal':
            continue
        d_us_i = d_u[d_u['attack_cat'] == AT_DICT[sub_attack]].sample(n=num_for_each_attack, random_state=42)
        d_us.append(d_us_i)
    
    d_u = pd.concat(d_us, ignore_index=True)
    d_u['Label']=0.0
    
    df_attack = pd.concat([df_attack_sample, d_u, df_normal], ignore_index=True)
    print(df_attack['attack_cat'].value_counts())
    # remove the attack_cat column
    df_attack = df_attack.drop(columns=['attack_cat'])
    print(df_attack['Label'].value_counts())
    # move the Label column to the end
    cols = list(df_attack.columns.values)
    cols.pop(cols.index('Label'))
    df_attack = df_attack[cols+['Label']]
    # save the file
    df_attack.to_csv(file_name, index=False)

# TEST SET

# Adjust according to your needs
num_for_each_attack = int((test.shape[0]*0.03)//7)
df_test = test

# take the normal data
df_test_normal = df_test[df_test['Label'] == 0]
# take the anomaly data
df_test_anomaly = df_test[df_test['Label'] == 1]
attack_sampled = []
for anomaly_type in ANOMALIES:
    df_test_anomaly_i = df_test_anomaly[df_test_anomaly['attack_cat'] == AT_DICT[anomaly_type]].sample(n=num_for_each_attack, random_state=42)
    attack_sampled.append(df_test_anomaly_i)
    
df_test_anomaly = pd.concat(attack_sampled, ignore_index=True)

df_test_final = pd.concat([df_test_normal, df_test_anomaly],ignore_index=True)

print(df_test_final['Label'].value_counts())

# save attack_cat column to csv
df_test_final['attack_cat'].to_csv(os.path.join(DEST_PATH, 'y_test_for_all.csv'), index=False)
# drop the attack column
df_test_final = df_test_final.drop(['attack_cat'], axis=1)
# pt Label column at the end
cols = list(df_test_final.columns.values)
cols.pop(cols.index('Label'))
df_attack = df_test_final[cols+['Label']]

file_name = os.path.join(DEST_PATH, 'test_for_all.csv')

df_attack.to_csv(file_name, index=False)


