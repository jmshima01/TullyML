import pandas as pd
import json
import numpy as np
from matplotlib import pyplot as plt
import scipy.io

'''
1.       Bench, sFoLqgI5afw6UcfrmxZX
2.       Barbell Curl, bkziJY2HV6G0NF0zUUqU
3.       Back Squat, lCcTdGNrsSUCUcWFpdHh
4.       Overhead Press, 9PJxOlCdxcbFBvZw7gk9
5.       Deadlift, fZXPgJEMjxIMjz5ZxXqZ

'''

def dc_block(dataset, time, init_coeff=None, fc=0.001):
    Ts = np.mean(np.diff(time))
    fs = 1/Ts
    #Ns = len(Ts)
    #time = time-time[0] #already did in main...

    x_diff = np.diff(dataset)
    y = np.zeros(len(dataset))
    
    wc = 2*np.pi*fc/fs

    ACC_ALPHA = np.cos(wc) - np.sqrt(np.cos(wc)*np.cos(wc) - 8*np.cos(wc) + 7)

    if init_coeff != None:
        y[0] = dataset[0]
    
    for i in range(1,len(dataset)):
        y[i] = ACC_ALPHA*y[i-1] + x_diff[i-1]
    
    return y



def simpsons_rule(x, h, c=None, leaky_coeff=1):
    Nstart = 2
    y = np.zeros(len(x))
    
    if c != None:
        y[0] = c

    for i in range(Nstart,len(x)):
        y[i] = leaky_coeff*y[i-2] + 1/3*(x[i] + 4*x[i-1] + x[i-2]) * h

    return y




# ========== main() ====================
if __name__ == "__main__":
    # READ DATA FROM data.json...
    with open("data.json") as f:

        data = json.load(f)

        # Overhead Press, 9PJxOlCdxcbFBvZw7gk9
        overhead_xacc = np.array(list(data['data']["__collections__"]["history"]['9PJxOlCdxcbFBvZw7gk9']['__collections__']['raw_acc']['rawU5UtIv8orYCEkkOpJ']['xacc']))
        overhead_yacc = np.array(list(data['data']["__collections__"]["history"]['9PJxOlCdxcbFBvZw7gk9']['__collections__']['raw_acc']['rawU5UtIv8orYCEkkOpJ']['yacc']))
        overhead_zacc = np.array(list(data['data']["__collections__"]["history"]['9PJxOlCdxcbFBvZw7gk9']['__collections__']['raw_acc']['rawU5UtIv8orYCEkkOpJ']['zacc']))
        overhead_time = np.array(list(data['data']["__collections__"]["history"]['9PJxOlCdxcbFBvZw7gk9']['__collections__']['raw_time']['gu5M31wNhf8D8Z85y7j9']['time']))
        overhead_time = overhead_time - overhead_time[0] 


        # Barbell Curl, bkziJY2HV6G0NF0zUUqU
        curl_xacc = np.array(list(data['data']["__collections__"]["history"]['bkziJY2HV6G0NF0zUUqU']['__collections__']['raw_acc']['snji2pTvZmPpcpogOzAX']['xacc']))
        curl_yacc = np.array(list(data['data']["__collections__"]["history"]['bkziJY2HV6G0NF0zUUqU']['__collections__']['raw_acc']['snji2pTvZmPpcpogOzAX']['yacc']))
        curl_zacc = np.array(list(data['data']["__collections__"]["history"]['bkziJY2HV6G0NF0zUUqU']['__collections__']['raw_acc']['snji2pTvZmPpcpogOzAX']['zacc']))
        curl_time = np.array(list(data['data']["__collections__"]["history"]['bkziJY2HV6G0NF0zUUqU']['__collections__']['raw_time']['lmVhWYwzYvQiVD1zGUMv']['time']))
        curl_time = curl_time - curl_time[0] 

        # Deadlift, fZXPgJEMjxIMjz5ZxXqZ
        deadlift_xacc = np.array(list(data['data']["__collections__"]["history"]['fZXPgJEMjxIMjz5ZxXqZ']['__collections__']['raw_acc']['JctsFDC1KjwU8qXO07SY']['xacc']))
        deadlift_yacc = np.array(list(data['data']["__collections__"]["history"]['fZXPgJEMjxIMjz5ZxXqZ']['__collections__']['raw_acc']['JctsFDC1KjwU8qXO07SY']['yacc']))
        deadlift_zacc = np.array(list(data['data']["__collections__"]["history"]['fZXPgJEMjxIMjz5ZxXqZ']['__collections__']['raw_acc']['JctsFDC1KjwU8qXO07SY']['zacc']))
        deadlift_time = np.array(list(data['data']["__collections__"]["history"]['fZXPgJEMjxIMjz5ZxXqZ']['__collections__']['raw_time']['aHWqtDacLbu2GfV68fYI']['time']))
        deadlift_time = deadlift_time - deadlift_time[0] 
        
        # Back Squat, lCcTdGNrsSUCUcWFpdHh
        squat_xacc = np.array(list(data['data']["__collections__"]["history"]['lCcTdGNrsSUCUcWFpdHh']['__collections__']['raw_acc']['ef2E8hh7CdHGEU5HEsX9']['xacc']))
        squat_yacc = np.array(list(data['data']["__collections__"]["history"]['lCcTdGNrsSUCUcWFpdHh']['__collections__']['raw_acc']['ef2E8hh7CdHGEU5HEsX9']['yacc']))
        squat_zacc = np.array(list(data['data']["__collections__"]["history"]['lCcTdGNrsSUCUcWFpdHh']['__collections__']['raw_acc']['ef2E8hh7CdHGEU5HEsX9']['zacc']))
        squat_time = np.array(list(data['data']["__collections__"]["history"]['lCcTdGNrsSUCUcWFpdHh']['__collections__']['raw_time']['aFWme3iCwy1vX7K4cTot']['time']))
        squat_time = squat_time - squat_time[0] 

        # Bench, sFoLqgI5afw6UcfrmxZX
        bench_xacc = np.array(list(data['data']["__collections__"]["history"]['sFoLqgI5afw6UcfrmxZX']['__collections__']['raw_acc']['l1TfF4zou5vJchT2iukX']['xacc']))
        bench_yacc = np.array(list(data['data']["__collections__"]["history"]['sFoLqgI5afw6UcfrmxZX']['__collections__']['raw_acc']['l1TfF4zou5vJchT2iukX']['yacc']))
        bench_zacc = np.array(list(data['data']["__collections__"]["history"]['sFoLqgI5afw6UcfrmxZX']['__collections__']['raw_acc']['l1TfF4zou5vJchT2iukX']['zacc']))
        bench_time = np.array(list(data['data']["__collections__"]["history"]['sFoLqgI5afw6UcfrmxZX']['__collections__']['raw_time']['Dxu4MWwfRL0l8d4nU07Q']['time']))
        print(bench_time)
        #bench_time = bench_time - bench_time[0] 


        print("done reading...")


    # test against matlab dataset
    #mat = scipy.io.loadmat('dataset_1.mat')
    #ax_dyn = dc_block(mat['ax'][0],mat['t'][0])

    curl_ax_dyn = dc_block(curl_xacc,curl_time)
    curl_ay_dyn = dc_block(curl_yacc,curl_time)
    curl_az_dyn = dc_block(curl_zacc,curl_time)

    bench_ax_dyn = dc_block(bench_xacc,bench_time)
    bench_ay_dyn = dc_block(bench_yacc,bench_time)
    bench_az_dyn = dc_block(bench_zacc,bench_time)

    overhead_ax_dyn = dc_block(overhead_xacc,overhead_time)
    overhead_ay_dyn = dc_block(overhead_yacc,overhead_time)
    overhead_az_dyn = dc_block(overhead_zacc,overhead_time)

    deadlift_ax_dyn = dc_block(deadlift_xacc,deadlift_time)
    deadlift_ay_dyn = dc_block(deadlift_yacc,deadlift_time)
    deadlift_az_dyn = dc_block(deadlift_zacc,deadlift_time)

    squat_ax_dyn = dc_block(squat_xacc,squat_time)
    squat_ay_dyn = dc_block(squat_yacc,squat_time)
    squat_az_dyn = dc_block(squat_zacc,squat_time)

    # PLOTTING...
    plt.plot(curl_time,curl_ax_dyn)
    plt.plot(curl_time,curl_ay_dyn)
    plt.plot(curl_time,curl_az_dyn)
    plt.legend(['ax','ay','az'])
    plt.title("Curl Dynamic Acceleration")
    plt.show()

    plt.plot(bench_time,bench_ax_dyn)
    plt.plot(bench_time,bench_ay_dyn)
    plt.plot(bench_time,bench_az_dyn)
    plt.legend(['ax','ay','az'])
    plt.title("Bench Dynamic Acceleration")
    plt.show()

    plt.plot(squat_time,squat_ax_dyn)
    plt.plot(squat_time,squat_ay_dyn)
    plt.plot(squat_time,squat_az_dyn)
    plt.legend(['ax','ay','az'])
    plt.title("Squat Dynamic Acceleration")
    plt.show()

    plt.plot(deadlift_time,deadlift_ax_dyn)
    plt.plot(deadlift_time,deadlift_ay_dyn)
    plt.plot(deadlift_time,deadlift_az_dyn)
    plt.legend(['ax','ay','az'])
    plt.title("Deadlift Acceleration")
    plt.show()

    plt.plot(overhead_time,overhead_ax_dyn)
    plt.plot(overhead_time,overhead_ay_dyn)
    plt.plot(overhead_time,overhead_az_dyn)
    plt.legend(['ax','ay','az'])
    plt.title("Overhead Press Acceleration")
    plt.show()

        


        
        

