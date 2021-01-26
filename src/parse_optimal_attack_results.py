import os
import numpy as np
import pandas as pd
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
args = vars(parser.parse_args())
base = args['path']
folders = sorted([f for f in os.listdir(base)])
# print(folders)
D = {'exp_id':[], 'reward_mean':[], 'reward_std':[]}
for f in folders:
    filename = os.path.join(base, f, 'attack_scan', 'optatk_deterministic.log')
    if not os.path.exists(filename):
        continue 
    with open(filename, 'r') as l:
        for line in l:
            pass
        last_line = line
    
    try:
        print('got mean:', float(last_line.split()[1][:-1]))
        print('got std:', float(last_line.split()[2][4:-1]))
    except:
        continue
    D['exp_id'].append(f)
    D['reward_mean'].append(float(last_line.split()[1][:-1]))
    D['reward_std'].append(float(last_line.split()[2][4:-1]))

df = pd.DataFrame(D)
print('\nchosing the agent with the best (lowest) reward...')
best_df = df[df['reward_mean'] == df['reward_mean'].min()]
print('best exp id: {}\nbest reward (mean+-std): {}+-{}'.format(best_df['exp_id'].to_numpy()[0], best_df['reward_mean'].to_numpy()[0], best_df['reward_std'].to_numpy()[0]))

save_filename = 'optatk_deter_reward.csv'
df.to_csv(save_filename)
print('full result saved at: ', save_filename)

