import numpy as np

std = np.zeros((1, 7))
seed = np.zeros((10, 7))
for i in range(10):
    name = 'gcn_result_con/128_128_multi_5fold_drug_miRNA_seed' + str(i) + '.csv'
    seed[i, :] = np.loadtxt(name, delimiter=',', dtype=float)
for i in range(7):
    std[0, i] = np.std(seed[:, i], ddof=1)
fname = 'gcn_result_con/std_128_128_multi_5fold.csv'
np.savetxt(fname, std, delimiter=',')
