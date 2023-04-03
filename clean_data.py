#Libraries:
import pandas as pd
import json
import numpy as np
from matplotlib import pyplot as plt
import scipy.io
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten,Conv2D,MaxPooling2D

'''
Helo ML Classification (CNN)
@author James Shima 
@author Jim Shima

CLASSES:
1.       Bench, sFoLqgI5afw6UcfrmxZX
2.       Barbell Curl, bkziJY2HV6G0NF0zUUqU
3.       Back Squat, lCcTdGNrsSUCUcWFpdHh
4.       Overhead Press, 9PJxOlCdxcbFBvZw7gk9
5.       Deadlift, fZXPgJEMjxIMjz5ZxXqZ

'''
class_labels = {0:"Bench", 1:"Curl", 2:"Squat", 3:"Overhead Press", 4:"Deadlift"}

# ============== Helper Functions ==================
def dc_block(dataset, time, init_coeff=None, fc=0.1):
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



def simpsons_rule(x, time, c=None, leaky_coeff=0.99):
    h = np.mean(np.diff(time))
    #Ns = len(Ts)
    #time = time-time[0] #already did in main...

    Nstart = 2
    y = np.zeros(len(x))
    
    if c != None:
        y[0] = c

    for i in range(Nstart,len(x)):
        y[i] = leaky_coeff*y[i-2] + 1/3*(x[i] + 4*x[i-1] + x[i-2]) * h

    return y


def make_matrix(x_slice, y_slice, z_slice):
    return np.vstack((x_slice, y_slice, z_slice))


def parse_data(x_data, y_data, z_data, class_ind, parse_size=20):
    parsed = []
    skip = 1
    
    length = x_data.size
    while(length%parse_size!=0):
        length-=1
    print(length)
    labels = []
    for i in range(0,length-parse_size,parse_size):
        #if(skip == 1):
        parsed.append(make_matrix(x_data[i:i+parse_size],y_data[i:i+parse_size],z_data[i:i+parse_size]))
        labels.append(class_ind)

        skip ^= 1
    
    return np.array(parsed),np.array(labels)
    
# ========== MAIN ====================
if __name__ == "__main__":
    # READ DATA FROM data.json...
    with open("data.json") as f:

        data = json.load(f)

        # Overhead Press, 9PJxOlCdxcbFBvZw7gk9
        overhead_xacc = np.array(list(data['data']["__collections__"]["history"]['9PJxOlCdxcbFBvZw7gk9']['__collections__']['raw_acc']['rawU5UtIv8orYCEkkOpJ']['xacc']))
        overhead_yacc = np.array(list(data['data']["__collections__"]["history"]['9PJxOlCdxcbFBvZw7gk9']['__collections__']['raw_acc']['rawU5UtIv8orYCEkkOpJ']['yacc']))
        overhead_zacc = np.array(list(data['data']["__collections__"]["history"]['9PJxOlCdxcbFBvZw7gk9']['__collections__']['raw_acc']['rawU5UtIv8orYCEkkOpJ']['zacc']))
        overhead_time = np.array(list(data['data']["__collections__"]["history"]['9PJxOlCdxcbFBvZw7gk9']['__collections__']['raw_time']['gu5M31wNhf8D8Z85y7j9']['time']))
        overhead_time -= overhead_time[0] 

        # Barbell Curl, bkziJY2HV6G0NF0zUUqU
        curl_xacc = np.array(list(data['data']["__collections__"]["history"]['bkziJY2HV6G0NF0zUUqU']['__collections__']['raw_acc']['snji2pTvZmPpcpogOzAX']['xacc']))
        curl_yacc = np.array(list(data['data']["__collections__"]["history"]['bkziJY2HV6G0NF0zUUqU']['__collections__']['raw_acc']['snji2pTvZmPpcpogOzAX']['yacc']))
        curl_zacc = np.array(list(data['data']["__collections__"]["history"]['bkziJY2HV6G0NF0zUUqU']['__collections__']['raw_acc']['snji2pTvZmPpcpogOzAX']['zacc']))
        curl_time = np.array(list(data['data']["__collections__"]["history"]['bkziJY2HV6G0NF0zUUqU']['__collections__']['raw_time']['lmVhWYwzYvQiVD1zGUMv']['time']))
        curl_time -= curl_time[0] 

        # Deadlift, fZXPgJEMjxIMjz5ZxXqZ
        deadlift_xacc = np.array(list(data['data']["__collections__"]["history"]['fZXPgJEMjxIMjz5ZxXqZ']['__collections__']['raw_acc']['JctsFDC1KjwU8qXO07SY']['xacc']))
        deadlift_yacc = np.array(list(data['data']["__collections__"]["history"]['fZXPgJEMjxIMjz5ZxXqZ']['__collections__']['raw_acc']['JctsFDC1KjwU8qXO07SY']['yacc']))
        deadlift_zacc = np.array(list(data['data']["__collections__"]["history"]['fZXPgJEMjxIMjz5ZxXqZ']['__collections__']['raw_acc']['JctsFDC1KjwU8qXO07SY']['zacc']))
        deadlift_time = np.array(list(data['data']["__collections__"]["history"]['fZXPgJEMjxIMjz5ZxXqZ']['__collections__']['raw_time']['aHWqtDacLbu2GfV68fYI']['time']))
        deadlift_time -= deadlift_time[0] 
        
        # Back Squat, lCcTdGNrsSUCUcWFpdHh
        squat_xacc = np.array(list(data['data']["__collections__"]["history"]['lCcTdGNrsSUCUcWFpdHh']['__collections__']['raw_acc']['ef2E8hh7CdHGEU5HEsX9']['xacc']))
        squat_yacc = np.array(list(data['data']["__collections__"]["history"]['lCcTdGNrsSUCUcWFpdHh']['__collections__']['raw_acc']['ef2E8hh7CdHGEU5HEsX9']['yacc']))
        squat_zacc = np.array(list(data['data']["__collections__"]["history"]['lCcTdGNrsSUCUcWFpdHh']['__collections__']['raw_acc']['ef2E8hh7CdHGEU5HEsX9']['zacc']))
        squat_time = np.array(list(data['data']["__collections__"]["history"]['lCcTdGNrsSUCUcWFpdHh']['__collections__']['raw_time']['aFWme3iCwy1vX7K4cTot']['time']))
        squat_time -= squat_time[0] 

        # Bench, sFoLqgI5afw6UcfrmxZX
        bench_xacc = np.array(list(data['data']["__collections__"]["history"]['sFoLqgI5afw6UcfrmxZX']['__collections__']['raw_acc']['l1TfF4zou5vJchT2iukX']['xacc']))
        bench_yacc = np.array(list(data['data']["__collections__"]["history"]['sFoLqgI5afw6UcfrmxZX']['__collections__']['raw_acc']['l1TfF4zou5vJchT2iukX']['yacc']))
        bench_zacc = np.array(list(data['data']["__collections__"]["history"]['sFoLqgI5afw6UcfrmxZX']['__collections__']['raw_acc']['l1TfF4zou5vJchT2iukX']['zacc']))
        bench_time = np.array(list(data['data']["__collections__"]["history"]['sFoLqgI5afw6UcfrmxZX']['__collections__']['raw_time']['Dxu4MWwfRL0l8d4nU07Q']['time']))
        bench_time -= bench_time[0] 


        print("done reading...")


    #test against matlab dataset
    # mat = scipy.io.loadmat('dataset_1.mat')
    # ax_dyn = dc_block(mat['ax'][0],mat['t'][0])
    # ay_dyn = dc_block(mat['ay'][0],mat['t'][0])
    # az_dyn = dc_block(mat['az'][0],mat['t'][0])

    # vx_dyn = simpsons_rule(ax_dyn,mat['t'][0],leaky_coeff=0.95)
    # vy_dyn = simpsons_rule(ay_dyn,mat['t'][0],leaky_coeff=0.95)
    # vz_dyn = simpsons_rule(az_dyn,mat['t'][0],leaky_coeff=0.95)

    # px_dyn = simpsons_rule(vx_dyn,mat['t'][0],leaky_coeff=0.95)
    # py_dyn = simpsons_rule(vy_dyn,mat['t'][0],leaky_coeff=0.95)
    # pz_dyn = simpsons_rule(vz_dyn,mat['t'][0],leaky_coeff=0.95)

    # plt.plot(mat['t'][0],px_dyn)
    # plt.plot(mat['t'][0],py_dyn)
    # plt.plot(mat['t'][0],pz_dyn)
    # plt.show()

    # ::::::::::: ACCELERATION :::::::::::::

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


    #PLOTTING...
    # plt.plot(curl_time,curl_ax_dyn)
    # plt.plot(curl_time,curl_ay_dyn)
    # plt.plot(curl_time,curl_az_dyn)
    # plt.legend(['ax','ay','az'])
    # plt.title("Curl Dynamic Acceleration")
    # plt.show()

    # plt.plot(bench_time,bench_ax_dyn)
    # plt.plot(bench_time,bench_ay_dyn)
    # plt.plot(bench_time,bench_az_dyn)
    # plt.legend(['ax','ay','az'])
    # plt.title("Bench Dynamic Acceleration")
    # plt.show()

    # plt.plot(squat_time,squat_ax_dyn)
    # plt.plot(squat_time,squat_ay_dyn)
    # plt.plot(squat_time,squat_az_dyn)
    # plt.legend(['ax','ay','az'])
    # plt.title("Squat Dynamic Acceleration")
    # plt.show()

    # plt.plot(deadlift_time,deadlift_ax_dyn)
    # plt.plot(deadlift_time,deadlift_ay_dyn)
    # plt.plot(deadlift_time,deadlift_az_dyn)
    # plt.legend(['ax','ay','az'])
    # plt.title("Deadlift Acceleration")
    # plt.show()

    # plt.plot(overhead_time,overhead_ax_dyn)
    # plt.plot(overhead_time,overhead_ay_dyn)
    # plt.plot(overhead_time,overhead_az_dyn)
    # plt.legend(['ax','ay','az'])
    # plt.title("Overhead Press Acceleration")
    # plt.show()


    # :::::::: VELOCITY ::::::::
    alpha = 0.97
        
    curl_vx_dyn = simpsons_rule(curl_ax_dyn,curl_time,leaky_coeff=alpha)
    curl_vy_dyn = simpsons_rule(curl_ay_dyn,curl_time,leaky_coeff=alpha)
    curl_vz_dyn = simpsons_rule(curl_az_dyn,curl_time,leaky_coeff=alpha)

    deadlift_vx_dyn = simpsons_rule(deadlift_ax_dyn,curl_time,leaky_coeff=alpha)
    deadlift_vy_dyn = simpsons_rule(deadlift_ay_dyn,curl_time,leaky_coeff=alpha)
    deadlift_vz_dyn = simpsons_rule(deadlift_az_dyn,curl_time,leaky_coeff=alpha)


    squat_vx_dyn = simpsons_rule(squat_ax_dyn,curl_time,leaky_coeff=alpha)
    squat_vy_dyn = simpsons_rule(squat_ay_dyn,curl_time,leaky_coeff=alpha)
    squat_vz_dyn = simpsons_rule(squat_az_dyn,curl_time,leaky_coeff=alpha)

    overhead_vx_dyn = simpsons_rule(overhead_ax_dyn,curl_time,leaky_coeff=alpha)
    overhead_vy_dyn = simpsons_rule(overhead_ay_dyn,curl_time,leaky_coeff=alpha)
    overhead_vz_dyn = simpsons_rule(overhead_az_dyn,curl_time,leaky_coeff=alpha)

    bench_vx_dyn = simpsons_rule(bench_ax_dyn,curl_time,leaky_coeff=alpha)
    bench_vy_dyn = simpsons_rule(bench_ay_dyn,curl_time,leaky_coeff=alpha)
    bench_vz_dyn = simpsons_rule(bench_az_dyn,curl_time,leaky_coeff=alpha)


    print(bench_vx_dyn.size)
    print(squat_vx_dyn.size)
    print(overhead_vx_dyn.size)
    print(curl_vx_dyn.size)
    print(deadlift_vx_dyn.size)
    print()

    # class_labels = {0:"Bench", 1:"Curl", 2:"Squat", 3:"Overhead Press", 4:"Deadlift"}
    # print(parse_data(bench_vx_dyn,bench_vy_dyn,bench_vz_dyn,class_ind=0)[1].shape)
    # print(parse_data(bench_vx_dyn,bench_vy_dyn,bench_vz_dyn,class_ind=0)[0].shape)

    # print(parse_data(deadlift_vx_dyn,deadlift_vy_dyn,deadlift_vz_dyn,class_ind=4)[1].shape)
    # print(parse_data(deadlift_vx_dyn,deadlift_vy_dyn,deadlift_vz_dyn,class_ind=4)[0].shape)

    # print(parse_data(squat_vx_dyn,squat_vy_dyn,squat_vz_dyn,class_ind=4)[1].shape)
    # print(parse_data(squat_vx_dyn,squat_vy_dyn,squat_vz_dyn,class_ind=4)[0].shape)

    # print(parse_data(curl_vx_dyn,curl_vy_dyn,curl_vz_dyn,class_ind=4)[1].shape)
    # print(parse_data(curl_vx_dyn,curl_vy_dyn,curl_vz_dyn,class_ind=4)[0].shape)

    # print(parse_data(overhead_vx_dyn,overhead_vy_dyn,overhead_vz_dyn,class_ind=4)[1].shape)
    # print(parse_data(overhead_vx_dyn,overhead_vy_dyn,overhead_vz_dyn,class_ind=4)[0].shape)

    y_labels = []
    
    y_labels.append(parse_data(bench_vx_dyn,bench_vy_dyn,bench_vz_dyn,class_ind=0)[1])
    bench_vtrain = parse_data(bench_vx_dyn,bench_vy_dyn,bench_vz_dyn,class_ind=0)[0]

    y_labels.append(parse_data(curl_vx_dyn,curl_vy_dyn,curl_vz_dyn,class_ind=1)[1])
    curl_vtrain = parse_data(curl_vx_dyn,curl_vy_dyn,curl_vz_dyn,class_ind=1)[0]

    y_labels.append(parse_data(squat_vx_dyn,squat_vy_dyn,squat_vz_dyn,class_ind=2)[1])
    squat_vtrain = parse_data(squat_vx_dyn,squat_vy_dyn,squat_vz_dyn,class_ind=2)[0]

    y_labels.append(parse_data(overhead_vx_dyn,overhead_vy_dyn,overhead_vz_dyn,class_ind=3)[1])
    overhead_vtrain = parse_data(overhead_vx_dyn,overhead_vy_dyn,overhead_vz_dyn,class_ind=3)[0]

    y_labels.append(parse_data(deadlift_vx_dyn,deadlift_vy_dyn,deadlift_vz_dyn,class_ind=4)[1])
    deadlift_vtrain = parse_data(deadlift_vx_dyn,deadlift_vy_dyn,deadlift_vz_dyn,class_ind=4)[0]

    flatten = []
    for arr in y_labels:
        for i in arr:
            flatten.append(i)
    
    X_tensor = np.vstack((bench_vtrain, curl_vtrain, squat_vtrain, overhead_vtrain, deadlift_vtrain))
    y_labels = np.array(flatten)
    print(y_labels.shape)
    print(X_tensor.shape)

    
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_labels, test_size=0.3)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)

    print(X_train.shape)
    print(y_train.shape)

    CNN = Sequential()

    CNN.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(3, 20)))
    CNN.add(MaxPooling1D(pool_size=5))
    CNN.add(Flatten())
    CNN.add(Dense(32, activation='relu'))
    CNN.add(Dense(units=5, activation='softmax'))
    CNN.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    CNN.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))
    score = CNN.evaluate(X_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])





        

    # :::::::: POSITION ::::::::::::

    # curl_vx_dyn = simpsons_rule(curl_ax_dyn,curl_time,leaky_coeff=alpha)
    # curl_vy_dyn = simpsons_rule(curl_ay_dyn,curl_time,leaky_coeff=alpha)
    # curl_vz_dyn = simpsons_rule(curl_az_dyn,curl_time,leaky_coeff=alpha)

    # deadlift_vx_dyn = simpsons_rule(deadlift_ax_dyn,curl_time,leaky_coeff=alpha)
    # deadlift_vy_dyn = simpsons_rule(deadlift_ay_dyn,curl_time,leaky_coeff=alpha)
    # deadlift_vx_dyn = simpsons_rule(deadlift_az_dyn,curl_time,leaky_coeff=alpha)


    # squat_vx_dyn = simpsons_rule(squat_ax_dyn,curl_time,leaky_coeff=alpha)
    # squat_vy_dyn = simpsons_rule(squat_ay_dyn,curl_time,leaky_coeff=alpha)
    # squat_vz_dyn = simpsons_rule(squat_az_dyn,curl_time,leaky_coeff=alpha)

    # overhead_vx_dyn = simpsons_rule(overhead_ax_dyn,curl_time,leaky_coeff=alpha)
    # overhead_vy_dyn = simpsons_rule(overhead_ay_dyn,curl_time,leaky_coeff=alpha)
    # overhead_vz_dyn = simpsons_rule(overhead_az_dyn,curl_time,leaky_coeff=alpha)

    # bench_vx_dyn = simpsons_rule(bench_ax_dyn,curl_time,leaky_coeff=alpha)
    # bench_vy_dyn = simpsons_rule(bench_ay_dyn,curl_time,leaky_coeff=alpha)
    # bench_vz_dyn = simpsons_rule(bench_az_dyn,curl_time,leaky_coeff=alpha)
