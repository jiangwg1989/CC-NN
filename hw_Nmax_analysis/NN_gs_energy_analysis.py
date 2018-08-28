import keras
import numpy as np
import re
import os
import matplotlib.pyplot as plt

from math import log
from math import e

from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Input, Dense, Dropout, Activation
from keras.models import Model
from keras import optimizers
from keras.optimizers import RMSprop
from scipy import interpolate
from keras.models import load_model

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import layers
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers.core import Lambda




def input_file(file_path,raw_data,gs_energy_line,nmax_line,hw_line):
    count = len(open(file_path,'rU').readlines())
    with open(file_path,'r') as f_1:
        data =  f_1.readlines()
        loop2 = 0
        loop1 = 0
        wtf = re.match('#', 'abc',flags=0)
        while loop1 < count:
            if ( re.match('#', data[loop1],flags=0) == wtf):
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop1]) 
                raw_data[loop2][0] = float(temp_1[gs_energy_line])
                raw_data[loop2][1] = int(temp_1[nmax_line])
                raw_data[loop2][2] = float(temp_1[hw_line])
                loop2 = loop2 + 1
            loop1 = loop1 + 1
        print loop2  

def input_raw_data_count(file_path):
    count = len(open(file_path,'rU').readlines())
    with open(file_path,'r') as f_1:
        data =  f_1.readlines()
        loop2 = 0
        loop1 = 0
        wtf = re.match('#', 'abc',flags=0)
        while loop1 < count:
            if ( re.match('#', data[loop1],flags=0) == wtf):
                loop2 = loop2 + 1
            loop1 = loop1 + 1
    return loop2  



def NN_all(file_path,data_num,monitor,min_delta,patience,epochs,input_dim,output_dim,interpol_count):
     
    raw_data = np.zeros((data_num,3),dtype = np.float)

    
    input_file(file_path,raw_data,gs_energy_line,nmax_line,hw_line)
   

 
    #
    # To get more data, we do interpolation for the data
    #
    # interpolation for second colum
    # kind can be 'slinear', 'quadratic' and 'cubic' refer to a spline interpolation     of first, second or third order)  
    kind = "quadratic"
    nmax_max = int(np.max(raw_data[:,1]))
    nmax_min = int(np.min(raw_data[:,1]))
    nmax_count = (nmax_max-nmax_min)/2 + 1
    x_new = np.zeros((nmax_count,interpol_count))
    y_new = np.zeros((nmax_count,interpol_count))
    interpol_count_tot = 0   
   
    # interpolation for each namx 
    for loop1 in range(0,nmax_count):
        raw_data_new = raw_data[np.where(raw_data[:,1]==(loop1*2+nmax_min))]
        x = raw_data_new[:,2]
        y = raw_data_new[:,0]
        x_new[loop1,:] = np.linspace(np.min(x),np.max(x),interpol_count)
        f=interpolate.interp1d(x,y,kind=kind)
        y_new[loop1,:] = f(x_new[loop1,:])
        x_new_count = len(x_new[loop1,:])
        interpol_count_tot = interpol_count_tot + x_new_count
        #
        # plot interplation point
        #
        fig1 = plt.figure('fig1')
        l1=plt.scatter(x,y,color='k',linestyle='--',s = 10, marker = 'o', label='CC_calculation')
    
    l2=plt.scatter(x_new,y_new,color='r',linestyle='--',s = 10, marker = 'x', label='CC_calculation')
    
    #plt.ylim((-0.05, 0.01))
    fig1.show()

 
    # joint all initerpoled data    
    data_interpolation= np.zeros((interpol_count_tot,3))
    count = 0
    for loop2 in range(0,nmax_count):
        for loop3 in range(0,interpol_count):
            data_interpolation[count,1] = loop2*2 + nmax_min
            data_interpolation[count,2] = x_new[loop2,loop3]
            data_interpolation[count,0] = y_new[loop2,loop3]
            count = count +1

    print data_interpolation 
    
    
#    #
#    # shuffle the data
#    #
#    np.random.shuffle(data_interpolation)
#    np.random.shuffle(raw_data)
#    
#    print len(raw_data)
#    print raw_data
#    
#    #batch_size = data_num
#    
#    input_shape = (input_dim,)
#    input_data = Input(shape = input_shape)
#    
#    #raw_data_new  = raw_data[np.where(raw_data[:,1]<11)]
#    #raw_data_new = data_interpolation[np.where(data_interpolation[:,1]<25)]
#    raw_data_new = data_interpolation
#    print "raw_data_new="+str(raw_data_new)
#    
#    x_train = raw_data_new[:,1:3]
#    y_train = raw_data_new[:,0]
#    
#    print "x_train = "+str(x_train)
#    print "y_train = "+str(y_train)
#    
#    #
#    # NN Model
#    # 
#    x = Dense(8, activation = 'sigmoid')(input_data)
#    predictions = Dense(output_dim)(x)
#    model = Model(inputs= input_data, outputs = predictions)
#    
#    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#    
#    model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
#    
#    early_stopping = EarlyStopping(monitor=monitor,min_delta = min_delta , patience=patience, verbose=0, mode='min')
#    
#    model.fit(x_train,y_train, epochs = epochs, validation_split = 0.15 , shuffle = 1, callbacks=[early_stopping])
#    
#    model.save('He8_gs.h5')
    
    #
    # load model
    #
    #Li6_model = load_model('Li6_gs.h5') 
    
    
    
#    count = len(range(4,204,1))*len(range(5,121,1))
#    x_test = np.zeros((count,2),dtype = np.float)
#    
#    loop3 = 0
#    
#    
#    for loop1 in range(4,204,1):
#        for loop2 in range(5,121,1):
#            x_test[loop3][0] = loop1 
#            x_test[loop3][1] = loop2 
#            loop3 = loop3 + 1
#    
#    print x_test
#    
#    y_test = model.predict(x_test)
#    
#    raw_predic_data = np.concatenate((y_test,x_test),axis=1)
#    
#    print "raw_predic_data="+str(raw_predic_data)
#    
#    #fig,(ax0,ax1) = plt.subplots(nrows = 2, figsize=(9,9))
#    
#    x_list_1 = raw_data[:,2] 
#    y_list_1 = raw_data[:,0]
#    
#    raw_predic_data_4 = raw_predic_data[np.where(raw_predic_data[:,1]==4)]
#    raw_predic_data_8 = raw_predic_data[np.where(raw_predic_data[:,1]==8)]
#    raw_predic_data_12 = raw_predic_data[np.where(raw_predic_data[:,1]==12)]
#    raw_predic_data_20 = raw_predic_data[np.where(raw_predic_data[:,1]==20)]
#    raw_predic_data_40 = raw_predic_data[np.where(raw_predic_data[:,1]==40)]
#    raw_predic_data_60 = raw_predic_data[np.where(raw_predic_data[:,1]==60)]
#    
#    
#    x_list_2 = raw_predic_data_4[:,2]
#    y_list_2 = raw_predic_data_4[:,0]
#    
#    x_list_3 = raw_predic_data_8[:,2]
#    y_list_3 = raw_predic_data_8[:,0]
#    
#    x_list_4 = raw_predic_data_12[:,2]
#    y_list_4 = raw_predic_data_12[:,0]
#    
#    x_list_5 = raw_predic_data_20[:,2]
#    y_list_5 = raw_predic_data_20[:,0]
#    
#    
#    x_list_6 = raw_predic_data_40[:,2]
#    y_list_6 = raw_predic_data_40[:,0]
#    
#    x_list_7 = raw_predic_data_60[:,2]
#    y_list_7 = raw_predic_data_60[:,0]
#    
#    
#    fig1 = plt.figure('fig1')
#    l1=plt.scatter(x_list_1,y_list_1,color='k',linestyle='--',s = 10, marker = 'x', label='CC_calculation')
#    l2=plt.plot(x_list_2,y_list_2,color='y',linestyle='--',label='NN_Nmax_4')
#    l3=plt.plot(x_list_3,y_list_3,color='r',linestyle='--',label='NN_Nmax_8')
#    l4=plt.plot(x_list_4,y_list_4,color='g',linestyle='--',label='NN_Nmax_12')
#    l5=plt.plot(x_list_5,y_list_5,color='c',linestyle='--',label='NN_Nmax_20')
#    
#    l6=plt.plot(x_list_6,y_list_6,color='m',linestyle='--',label='NN_Nmax_40')
#    l7=plt.plot(x_list_7,y_list_7,color='b',linestyle='--',label='NN_Nmax_60')
#    #l4=fig1.scatter(x_list_2,y_list_2,color='y',linestyle='--',marker=',')
#    #l5=fig1.scatter(x_list_2,y_list_2,color='r',linestyle='--',marker=',')
#    #l6=fig1.scatter(x_list_2,y_list_2,color='c',linestyle='--',marker=',')
#    #fig1.scatter(x_list_2,y_list_2,color='m',linestyle='--',marker=',')
#    plt.legend(loc = 'upper left') 
#    
#    plt.savefig('He8_gs.jpg')
#    fig1.show()
#    
#    
#    
#    #ax0.scatter(x_list_1, y_list_1, c='r', s=20, alpha=0.5)
#    #ax0.set_title('calculation')
#    #ax1.scatter(x_list_2, y_list_2, c='b', s=20, alpha=0.5)
#    #ax1.set_title('NN_prediction')
#    
#    #fig.subplots_adjust(hspace =0.4)
#    #plt.show()
#    
#    #x_list = range()
#    #y_list = 2/np.exp(-2*x_list)
#    #
#    #plt.figure('Scatter fig')
#    #ax = plt.gca()
#    #
#    #ax.set_xlabel('x')
#    #ax.set_ylabel('y')
#    #
#    #ax.scatter(x_list, y_list, c='r', s=20, alpha=0.5)
#    #
#    #plt.show()
#    file_path = "He8_gs_NN_prediction.txt"
#    with open(file_path,'w') as f_1:
#        for loop1 in range(1,count):
#            f_1.write('{:>-10.5f}'.format(y_test[loop1,0]))
#            f_1.write('{:>10}'.format(x_test[loop1,0]))
#            f_1.write('{:>10}'.format(x_test[loop1,1])+'\n')
    hw_set = [35,37,39,41,43,45]
    color = ['r','g','b','y','m','c','g']
    hw_set_count = len(hw_set)
    x_list = np.zeros((hw_set_count,nmax_count))   
    y_list = np.zeros((hw_set_count,nmax_count))  
     
    #for loop1 in range (0,hw_set_count):
    #    for loop2 in range (0,nmax_count):
    #        #x_list[loop1,loop2] = abs((x_new[loop2,:] - hw_set[loop1])) 
    #        #y_list[loop1,loop2] =
    #        #print "x_new="+str(x_new[loop2,:])
    #        #print "hw_set[loop1]= "+str(hw_set[loop1]) 
    #        locate_temp = np.where(abs( (x_new[loop2,:] - hw_set[loop1])) == min(abs((x_new[loop2,:] - hw_set[loop1]))) )
    #        x_list[loop1,loop2] = loop2*2 + nmax_min
    #        y_list[loop1,loop2] = y_new[loop2,locate_temp]
    #    
    #    y_list_log = np.log10(y_list - gs_converge)        
    #    fig_3 = plt.figure('fig_3')
    #    l = plt.plot(x_list[loop1,:],y_list_log[loop1,:],color=color[loop1],linestyle='--', marker = 'x', label='raw_data')
    #    plt.title("E(converge)="+str(gs_converge))
    #    plt.ylabel("lg(E(infinte)-E(converge))")
    #    plt.legend(loc = 'lower left')
    ##plt.ylim((1.2,2.8))
    ##plt.savefig('Li6_radius_NN_prediction.jpg')
    #    fig_3.show()
            
    #print "x_list = "+str(x_list)
    #print "y_list = "+str(y_list)


    #
    # plot for lowest point of each Nmax
    #
    x_list_1 = np.zeros(nmax_count) 
    y_list_1 = np.zeros(nmax_count)
    y_list_1_log_slope = np.zeros(nmax_count-1)
    gs_converge = np.min(y_new)- 0.00001
    for loop in range (0,1000):
        for loop1 in range (0,nmax_count):
            x_list_1[loop1] = loop1*2 + nmax_min
            y_list_1[loop1] = min(y_new[loop1,:])
           
        gs_converge = gs_converge - loop*0.001 
        y_list_1_log = np.log10(y_list_1 - gs_converge) 
        
        for loop2 in range (0,nmax_count-1):
            y_list_1_log_slope[loop2] = y_list_1_log[loop2+1]-y_list_1_log[loop2] 
        print y_list_1_log_slope
        if y_list_1_log_slope[nmax_count-2] > np.mean(y_list_1_log_slope[0:nmax_count-2]):
        #if y_list_1_log_slope[nmax_count-2] > y_list_1_log_slope[nmax_count-3]:
            break

       
    fig_2 = plt.figure('fig_2')
    l = plt.scatter(x_list_1,y_list_1_log,color='k',linestyle='--',s = 10, marker = 'x', label='raw_data')
    
    plt.title("E(converge)="+str(gs_converge))
    
    plt.ylabel("lg(E(infinte)-E(converge))")
    plt.legend(loc = 'lower left')
    #plt.ylim((1.2,2.8))
    #plt.savefig('Li6_radius_NN_prediction.jpg')
    fig_2.show()

 
     


#    raw_data_new_4 = raw_data[np.where(raw_data[:,2] == 50 )]
#    raw_data_new_3 = raw_data[np.where(raw_data[:,2] == 40 )]
#    raw_data_new_2 = raw_data[np.where(raw_data[:,2] == 30 )]
#    raw_data_new_1 = raw_data[np.where(raw_data[:,2] == 20 )]
#    
#    
#    #raw_data_new_5 = raw_data[np.where(raw_data[:,1] == 200)]
#    temp_1 = raw_data_new_5[:,0]
#    
#    #gs_converge = np.min(temp_1)
#    
#    
#    raw_data_new = np.zeros((200,2),dtype = np.float)
#    for loop in range(0,200):
#        raw_data_new[loop,1] = loop+4
#        raw_data_new[loop,0] = np.min(raw_data[np.where(raw_data[:,1]==loop+4)])
#    
#    x_list = raw_data_new[:,1]
#    y_list = np.log10(raw_data_new[:,0] - gs_converge)
#    
#    x_list_1 = raw_data_new_1 [:,1]
#    y_list_1 = np.log10(raw_data_new_1[:,0] - gs_converge)
#    
#    x_list_2 = raw_data_new_2 [:,1]
#    y_list_2 = np.log10(raw_data_new_2[:,0] - gs_converge)
#    
#    x_list_3 = raw_data_new_3 [:,1]
#    y_list_3 = np.log10(raw_data_new_3[:,0] - gs_converge)
#    
#    x_list_4 = raw_data_new_4 [:,1]
#    y_list_4 = np.log10(raw_data_new_4[:,0] - gs_converge)
#    
#    
#    
#    
#    fig_1 = plt.figure('fig_1')
#    l1 = plt.scatter(x_list_1,y_list_1,color='k',linestyle='--',s = 10, marker = 'x    ', label='NN_prediction_hw=20')
#    l2 = plt.scatter(x_list_2,y_list_2,color='r',linestyle='--',s = 10, marker = 'x    ', label='NN_prediction_hw=30')
#    l3 = plt.scatter(x_list_3,y_list_3,color='g',linestyle='--',s = 10, marker = 'x    ', label='NN_prediction_hw=40')
#    l4 = plt.scatter(x_list_4,y_list_4,color='b',linestyle='--',s = 10, marker = 'x    ', label='NN_prediction_hw=50')
#    
#    plt.title("E(converge)="+str(gs_converge))
#    
#    plt.ylabel("lg(E(infinte)-E(converge))")
#    plt.legend(loc = 'lower left')
#    #plt.ylim((1.2,2.8))
#    #plt.savefig('Li6_radius_NN_prediction.jpg')
#    fig_1.show()
#    
#    fig_2 = plt.figure('fig_2')

     


#
# all NN parameters
#
file_path = 'He8E_NNLOopt.txt'
data_num = input_raw_data_count(file_path)
print 'data_num='+str(data_num)
# earlystopping parameters
monitor  = 'val_loss'
min_delta = 0.0001
patience = 10
epochs = 10000
input_dim = 2 
output_dim = 1
# interpolation setting
interpol_count = 10000
hw_line = 2
nmax_line = 1
gs_energy_line = 0

#gs_converge = -27.592

NN_all(file_path=file_path,data_num=data_num,monitor=monitor,min_delta=min_delta,patience=patience,epochs=epochs,input_dim=input_dim,output_dim=output_dim,interpol_count=interpol_count)

input()
