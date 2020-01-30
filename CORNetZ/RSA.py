import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import scipy

def load_mats(folder='v1_processed'):
    mats = []
    f_names = list()
    for f in os.listdir(folder):
        if ('mat' in f):
            f_names.append(f)
            print('loading...', f)
            #mats.append(scipy.io.loadmat(folder + '/' + f, squeeze_me = True))
            mats.append(pd.read_pickle(folder + '/' + f))
    return mats, f_names

def cosine_dist(act1, act2):
    return scipy.spatial.distance.cosine(act1, act2)

def RSA(act):
    n_neurons = act.shape[1]
    rsa_list = [] 
    for i in range(act.shape[0]):
        print('img1: ' +str(i))
        act1 = act_mat[i]
        for j in range(act.shape[0]):
            if(i!=j):
                act2 = act_mat[j]
                rsa = cosine_dist(act1, act2) 
                rsa_list.append(rsa)
    return rsa_list

def create_dummy_df(n, fname):
    df = pd.DataFrame(np.zeros((n*n - n, 2)), columns=['img1', 'img2'])
    row = 0 
    for i in range(n):
        print('img1 ' + str(i))
        for j in range(n):
            if(i!=j):
                df['img1'].ix[row] = i
                df['img2'].ix[row] = j
                row += 1
    df.to_csv(fname + '.csv')
            
if __name__ == '__main__':
    mats, f_names = load_mats()
    for i, mat in enumerate(mats):
        print('Running ' + str(f_names[i]))
        act_mat = mat['activity'].T
        rsa_mat = RSA(act_mat)
        rsa_df = pd.DataFrame(rsa_mat)
        rsa_df.to_csv(f_names[i] + '_rsa.csv')
    
