#
#  with corelate loss, each interpolation point is correlate with the near by points
#



import keras
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Input, Dense, Dropout, Activation, Merge, Multiply
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

sess = tf.InteractiveSession()

#A = [[1,2,3,4,5]]
#AT= np.transpose(A)
#AT= AT.astype(np.int32)
#B = tf.ones_like(A)
#C = tf.multiply(A,A)
#Xi = tf.matmul(tf.transpose(B),C)
#Xj = tf.matmul(tf.transpose(C),B)
#Xij= tf.matmul(tf.transpose(A),A)
#Y  = Xi+Xj-tf.multiply(Xij,2)
#zero = tf.zeros_like(Y)
#one  = tf.ones_like(Y)
#Y  = tf.where(Y<2,one,zero)
##Y  = tf.where(Y<2,one,zero)
##print(sess.run(tf.fill([2,1],[1,2,3])))
#print(sess.run(Xi+Xj-tf.multiply(Xij,2)))
#print(sess.run(Y))
#input()

#A = [[1,2,3,4,5]]
#AT= np.transpose(A)
#AT= AT.astype(np.int32)
#B = tf.ones_like(A)
#C = tf.multiply(A,A)
#Xi = tf.matmul(tf.transpose(B),A)
#Xj = tf.transpose(Xi)
#Xij= tf.matmul(tf.transpose(A),A)
#Y = Xi-Xj
#zero = tf.zeros_like(Xi,tf.float32)
##zero = tf.cast(zero,tf.float32)
#one  = tf.ones_like(Xi,tf.float32)
##one  = tf.cast(one ,tf.float32)
#Y  = tf.where(Y<0,-Y,Y)
#BB = tf.ones_like([1,2,3,4,5],tf.float32)
#W1 = tf.diag(BB*(1-0.5))
#W2 = tf.where(Y<1,0.5*one,zero)
##print(sess.run(tf.fill([2,1],[1,2,3])))
#print(sess.run(Xi-Xj))
#print(sess.run(tf.add(W1,W2)))
#
#input()











def normfun(x,mu,sigma):
    pdf = np.exp(-((x - mu)**2)/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return pdf


def input_file_1(file_path,raw_data,gs_energy_line,nmax_line,hw_line):
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
                raw_data[loop2][4] = 1
                loop2 = loop2 + 1
            loop1 = loop1 + 1
        print loop2  

def input_file_2(file_path,raw_data):
    count = len(open(file_path,'rU').readlines())
    with open(file_path,'r') as f_1:
        data =  f_1.readlines()
        loop2 = 0
        loop1 = 0
        wtf = re.match('#', 'abc',flags=0)
        while loop1 < count:
            if ( re.match('#', data[loop1],flags=0) == wtf):
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
                raw_data[loop2][0] = float(temp_1[0])
                raw_data[loop2][1] = float(temp_1[1])
                raw_data[loop2][2] = float(temp_1[2])
                loop2 = loop2 + 1
            loop1 = loop1 + 1
    #    print loop2





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



def NN_all(input_path,output_path,data_num,monitor,min_delta,patience,epochs,batch_size,input_dim,output_dim,interpol_count,max_nmax_fit,sigma):
    
    #
    # Take in all the input data from file
    # 
    raw_data = np.zeros((data_num,5),dtype = np.float)
    input_file_1(input_path,raw_data,gs_energy_line,nmax_line,hw_line)
   

 
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
        x_new_count = len(x_new[loop1,:])-2 # ignore the first and last points
        interpol_count_tot = interpol_count_tot + x_new_count
 
    # joint all initerpoled data    
    data_interpolation = np.zeros((interpol_count_tot,5))
    min_position = np.zeros(nmax_count)
    count = 0
    for loop2 in range(0,nmax_count):
        for loop3 in range(1,interpol_count-1):
            data_interpolation[count,1] = loop2*2 + nmax_min
            data_interpolation[count,2] = x_new[loop2,loop3]
            data_interpolation[count,0] = y_new[loop2,loop3]
            data_interpolation[count,4] = 0
            #data_interpolation[count,3] = normfun(x_new[loop2,loop3],mu,sigma)
            count = count +1
        #print "max_y="+str(x_new[loop2,np.where(y_new[loop2,:]==np.min(y_new[loop2,:]))])
        min_position[loop2] = x_new[loop2,np.where(y_new[loop2,:]==np.min(y_new[loop2,:]))]

    #
    # set up gussian like sample weights for interpolation points and raw_data
    #
 
    # sort the raw_data with nmax(4->20)
    #raw_data = raw_data[raw_data[:,2].argsort()]
    raw_data = raw_data[raw_data[:,1].argsort()]

    count_1 = 0
    count_2 = 0
    for loop2 in range(0,nmax_count):
         for loop3 in range(1,interpol_count-1):
             if(sample_weight_switch   == 'on'):
                 data_interpolation[count_1,3] = normfun(x_new[loop2,loop3],min_position[loop2],sigma) 
             elif(sample_weight_switch == 'off'):
                 data_interpolation[count_1,3] = 1
             else:
                 print('sample_weight_switch error!') 
             count_1 = count_1 +1
         for loop4 in range(0,len(raw_data[np.where(raw_data[:,1]==(loop2*2+nmax_min))])):
             if(sample_weight_switch   == 'on'):
                 raw_data[count_2,3] = normfun(raw_data[count_2,2],min_position[loop2],sigma)
             elif(sample_weight_switch == 'off'):
                 raw_data[count_2,3] = 1
             else:
                 print('sample_weight_switch error!') 
             count_2 = count_2 +1
 
    #fig6 = plt.figure('fig6')
    #plt.plot(data_interpolation[30000:40000,2],data_interpolation[30000:40000,3],color='y',linestyle='--')
    #plt.ylim((-29,-23))  
    #plt.xlim((10,50))
    #fig6.show()
    #plt.close('all')


    #
    # data_new = raw_data + data_interpolation   [[raw_data][data_interpolation]] determine to add raw data in the fiting or not.
    #
    #data_new = np.vstack((raw_data,data_interpolation))
    data_new = data_interpolation

    #
    # setup correlation info   (different nmax position are different by +interpol_count)
    #
    correlation_position = np.zeros((len(data_new),1))
    
    loop3 = 0 
    for loop1 in range(0,nmax_count):
        for loop2 in range(1,interpol_count-1):
            correlation_position[loop3][0] = loop2 + 2*interpol_count*loop1
            loop3 = loop3+1
    data_new   = np.hstack((data_new,correlation_position))

    
    #
    # take part of the data for train (below certain nmax)
    #
    data_new   =  data_new[np.where((data_new[:,1]<=max_nmax_fit))]

    #print data_new[144,:]
    #input()

    #
    # shuffle the data
    #
    np.random.shuffle(data_new)
    #np.random.shuffle(data_interpolation)
    #np.random.shuffle(raw_data)
    
       
    
 
    #raw_data_new  = data_interpolation[np.where((data_interpolation[:,1]<=max_nmax_fit))]
    #raw_data_new = data_interpolation[np.where((data_interpolation[:,1]<=max_nmax_fit)&(data_interpolation[:,2]<=60))]
    #raw_data_new = data_interpolation
    #print "raw_data_new="+str(raw_data_new)
    
    x_train = data_new[:,1:3]
    y_train = data_new[:,0]
    c_position = data_new[:,5]

    #print "x_train = "+str(x_train)
    #print "y_train = "+str(y_train)
    #print "c_position="+str(c_position)    
    #print "c_position="+str(c_relation)   
    #input() 
    
    #
    # test
    #var_position = K.variable([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]])
    #var_relation = K.variable([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0.4,0.6,0.2,0.8],[0.3,0.5,0.9,0.1]])
    #
    #A= K.dot(var_relation,K.transpose(var_position))
    #B= K.dot(var_position,K.transpose(var_relation))
    #C= A+B
    ##A = K.dot(var,K.transpose(var))
    #D = K.sum(C)
    #print K.eval(C)
    #print K.eval(D)

    #input()

    #
    # NN Model
    #
    
    # initialize the input structure for NN
    input_shape = (input_dim,)
    input_data  = Input(shape = input_shape)
    y_in        = Input(shape = (1,))
    input_position = Input(shape = (1,))
 
    # get the residual matrix R_ij 
    def R_ij(k):
        R = k[0]-k[1]
        R_matrix = K.dot(R,K.transpose(R))
        #Rx = K.relu(R)
        A  = tf.constant(1.0)
        #Rxx= tf.cond(K.sum(Rx)<A,lambda:Rx,lambda:-Rx)
        return R_matrix

    def C_ij(k):
        B = tf.ones_like(k)                      # creat a [1][1][1]... array
        Xi = tf.matmul(B,tf.transpose(k))        # creat a [1,2,3,4...][1,2,3,4...] square matrix
        Xj = tf.transpose(Xi)                    # creat a [1,1,1,1...][2,2,2,2,..] square matrix
        Xij= tf.matmul(k,tf.transpose(k))        # get Xij matrix
        Y = Xi-Xj                                # get Xi-Xj matrix
        zero = tf.zeros_like(Xi,tf.float32)         
        one  = tf.ones_like(Xi,tf.float32)        
        Y  = tf.where(Y<0,-Y,Y)                  # matrix element is now all positive
        C_matrix  = tf.where(Y<corr_num,corr_weight*one,zero)  # if matrix element is smaller than 
        return C_matrix         

    def correlated_loss(k):
        D = K.sum(k)
        #A  = K.mean(K.dot(K.transpose(R),R)
        #A = K.dot(K.transpose(R),R)
        #A = K.mean(K.square(R))
        return D/batch_size
    
    def test(k):
        E = K.sum(k)
        #print(k)
        #input()
        #E = E*0+6*140 
        #B = k.ones_like()
        #return E/batch_size
        return E

    # set up NN structure
    x                 =  Dense(8, activation = 'sigmoid')(input_data)
    predictions       =  Dense(output_dim)(x)
    residual_layer    =  Lambda(R_ij, name ='R_ij')([predictions,y_in])
    correlation_layer =  Lambda(C_ij, name ='C_ij')(input_position)
    mul_layer         =  Multiply()([residual_layer,correlation_layer])
    loss_layer        =  Lambda(correlated_loss,name='correlated_loss')(mul_layer)
    test_layer        =  Lambda(test,name='test')(correlation_layer)

    model = Model(inputs= [input_data,y_in,input_position], outputs = loss_layer)
    
#    model = Model(inputs= [input_data,y_in,input_position], outputs = test_layer)
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    


    #
    # define my own loss function
    #    
    #def correlate_loss(y_true,y_pred):
    #    #A=K.mean(K.square(y_pred - y_true), axis=-1)
    #    #B=y_pred-y_true
    #    B=K.zeros(shape=(32,32))
    #    C=K.dot(K.transpose(y_pred-y_true),y_pred-y_true)
    #    D=K.dot(y_pred-y_true,C)
    #    E=K.dot(C,K.transpose(y_pred-y_true))
    #    if (y_pred[1,1]==1): 
    #        F=K.sum(C)*3
    #    else:
    #        F=K.sum(C)*2 
    #    #G=tf.mul(C,C)
    #    #H=K.sum(G) 
    #    return F
#K.dot(y_true,y_pred)
 

    model.compile(optimizer='adam',loss=lambda y_true,y_pred: y_pred)
    
    early_stopping = EarlyStopping(monitor=monitor,min_delta = min_delta , patience=patience, verbose=0, mode='min')
    
    history = model.fit([x_train,y_train,c_position],y_train, epochs = epochs,batch_size=batch_size, validation_split = 0.01 , shuffle = 1, callbacks=[early_stopping], sample_weight = data_new[:,3])
    loss = history.history['loss'][len(history.history['loss'])-1]
    val_loss = history.history['val_loss'][len(history.history['val_loss'])-1]
   
    #val_loss = 0 
    #fig4 = plt.figure('fig4')
    #plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    #plt.savefig('loss_val_loss.eps')
    plt.close('all')


    model_path = 'gs.h5' 
    model.save(model_path)
    predic_model = Model(inputs =input_data, outputs = predictions)
    #
    # load model
    #
    #Li6_model = load_model('Li6_gs.h5') 
    
    
    
    count = len(range(4,204,1))*len(range(5,121,1))
    x_test = np.zeros((count,2),dtype = np.float)
    
    loop3 = 0
    
    
    for loop1 in range(4,204,1):
        for loop2 in range(5,121,1):
            x_test[loop3][0] = loop1 
            x_test[loop3][1] = loop2 
            loop3 = loop3 + 1
    
    #print x_test
    
    y_test = predic_model.predict(x_test)
    
    raw_predic_data = np.concatenate((y_test,x_test),axis=1)
    
    #print "raw_predic_data="+str(raw_predic_data)
    
    #fig,(ax0,ax1) = plt.subplots(nrows = 2, figsize=(9,9))
    
    x_list_1 = raw_data[:,2] 
    y_list_1 = raw_data[:,0]
    
    raw_predic_data_4 = raw_predic_data[np.where(raw_predic_data[:,1]==4)]
    raw_predic_data_8 = raw_predic_data[np.where(raw_predic_data[:,1]==8)]
    raw_predic_data_12 = raw_predic_data[np.where(raw_predic_data[:,1]==12)]
    raw_predic_data_20 = raw_predic_data[np.where(raw_predic_data[:,1]==20)]
    raw_predic_data_40 = raw_predic_data[np.where(raw_predic_data[:,1]==40)]
    raw_predic_data_60 = raw_predic_data[np.where(raw_predic_data[:,1]==100)]
    
    temp = (raw_predic_data[np.where(raw_predic_data[:,1]== 200)])
    gs_converge = np.min(temp[:,0])
   
 
    
    x_list_2 = raw_predic_data_4[:,2]
    y_list_2 = raw_predic_data_4[:,0]
    
    x_list_3 = raw_predic_data_8[:,2]
    y_list_3 = raw_predic_data_8[:,0]
    
    x_list_4 = raw_predic_data_12[:,2]
    y_list_4 = raw_predic_data_12[:,0]
    
    x_list_5 = raw_predic_data_20[:,2]
    y_list_5 = raw_predic_data_20[:,0]
    
    
    x_list_6 = raw_predic_data_40[:,2]
    y_list_6 = raw_predic_data_40[:,0]
    
    x_list_7 = raw_predic_data_60[:,2]
    y_list_7 = raw_predic_data_60[:,0]
    
    
    fig1 = plt.figure('fig1')
    ax = plt.subplot(111)
    l1=plt.scatter(x_list_1,y_list_1,color='k',linestyle='--',s = 10, marker = 'x', label='CC_calculation')
    l2=plt.plot(x_list_2,y_list_2,color='y',linestyle='--',label='NN_Nmax_4')
    l3=plt.plot(x_list_3,y_list_3,color='r',linestyle='--',label='NN_Nmax_8')
    l4=plt.plot(x_list_4,y_list_4,color='g',linestyle='--',label='NN_Nmax_12')
    l5=plt.plot(x_list_5,y_list_5,color='c',linestyle='--',label='NN_Nmax_20')
    
    l6=plt.plot(x_list_6,y_list_6,color='m',linestyle='--',label='NN_Nmax_40')
    l7=plt.plot(x_list_7,y_list_7,color='b',linestyle='--',label='NN_Nmax_100')
    #l4=fig1.scatter(x_list_2,y_list_2,color='y',linestyle='--',marker=',')
    #l5=fig1.scatter(x_list_2,y_list_2,color='r',linestyle='--',marker=',')
    #l6=fig1.scatter(x_list_2,y_list_2,color='c',linestyle='--',marker=',')
    #fig1.scatter(x_list_2,y_list_2,color='m',linestyle='--',marker=',')
    #plt.legend(loc = 'upper left')

    xmajorLocator   = MultipleLocator(10)
    #xmajorFormatter = FormatStrFormatter('%5f')
    xminorLocator   = MultipleLocator(2)
    
    
    ymajorLocator   = MultipleLocator(5)
    #ymajorFormatter = FormatStrFormatter('%1.1d')
    yminorLocator   = MultipleLocator(1)

    ax.xaxis.set_major_locator(xmajorLocator)
    #ax.xaxis.set_major_formatter(xmajorFormatter)
    ax.yaxis.set_major_locator(ymajorLocator)
    #ax.yaxis.set_major_formatter(ymajorFormatter)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.yaxis.set_minor_locator(yminorLocator)
    ax.xaxis.grid(True, which='major') 
    ax.yaxis.grid(True, which='major')
    

    plt.legend(loc='upper right', bbox_to_anchor=(1.5,0.75),ncol=1,fancybox=True,shadow=True,borderaxespad = 0.)
    plt.title("gs(infinite)="+str(gs_converge))
    plot_path = 'gs.eps'
    plt.ylim((-29,-23))  
    plt.xlim((10,50))
    plt.subplots_adjust(right = 0.7)
    plt.savefig(plot_path)
    plt.close('all')
    #fig1.show()
    
    
    file_path = "gs_NN_prediction.txt"
    with open(file_path,'w') as f_1:
        for loop1 in range(1,count):
            f_1.write('{:>-10.5f}'.format(y_test[loop1,0]))
            f_1.write('{:>10}'.format(x_test[loop1,0]))
            f_1.write('{:>10}'.format(x_test[loop1,1])+'\n')
    os.system('cp '+file_path+' '+output_path)        
    os.system('cp '+model_path+' '+output_path) 
    os.system('cp '+'gs.eps'+' '+output_path) 
    os.system('cp '+'loss_val_loss.eps'+' '+output_path) 
#
# plot different_hw.eps and lowest_each_Nmax.eps
#
    data_num = len(open(file_path,'rU').readlines())
    raw_data = np.zeros((data_num,3),dtype = np.float)
    input_file_2(file_path,raw_data)
    
    raw_data_new_4 = raw_data[np.where(raw_data[:,2] == 50 )]
    raw_data_new_3 = raw_data[np.where(raw_data[:,2] == 40 )]
    raw_data_new_2 = raw_data[np.where(raw_data[:,2] == 30 )]
    raw_data_new_1 = raw_data[np.where(raw_data[:,2] == 20 )]
    
    raw_data_new_5 = raw_data[np.where(raw_data[:,1] == 200)]
    temp_1 = raw_data_new_5[:,0]
    gs_converge = np.min(temp_1)
    
    raw_data_new = np.zeros((200,2),dtype = np.float)
    for loop in range(0,200):
        raw_data_new[loop,1] = loop+4
        raw_data_new[loop,0] = np.min(raw_data[np.where(raw_data[:,1]==loop+4)])
    
    x_list = raw_data_new[:,1]
    y_list = np.log10(raw_data_new[:,0] - gs_converge)
    
    x_list_1 = raw_data_new_1 [:,1]
    y_list_1 = np.log10(raw_data_new_1[:,0] - gs_converge)
    
    x_list_2 = raw_data_new_2 [:,1]
    y_list_2 = np.log10(raw_data_new_2[:,0] - gs_converge)
    
    x_list_3 = raw_data_new_3 [:,1]
    y_list_3 = np.log10(raw_data_new_3[:,0] - gs_converge)
    
    x_list_4 = raw_data_new_4 [:,1]
    y_list_4 = np.log10(raw_data_new_4[:,0] - gs_converge)
    fig_1 = plt.figure('fig_1')
    l1 = plt.scatter(x_list_1,y_list_1,color='k',linestyle='--',s = 10, marker = 'x', label    ='NN_prediction_hw=20')
    l2 = plt.scatter(x_list_2,y_list_2,color='r',linestyle='--',s = 10, marker = 'x', label    ='NN_prediction_hw=30')
    l3 = plt.scatter(x_list_3,y_list_3,color='g',linestyle='--',s = 10, marker = 'x', label    ='NN_prediction_hw=40')
    l4 = plt.scatter(x_list_4,y_list_4,color='b',linestyle='--',s = 10, marker = 'x', label    ='NN_prediction_hw=50')
    
    plt.title("E(converge)="+str(gs_converge))
    
    plt.ylabel("lg(E(infinte)-E(converge))")
    plt.legend(loc = 'lower left')
    #plt.ylim((1.2,2.8))
    #plt.savefig('Li6_radius_NN_prediction.jpg')
    plot_path = 'different_hw.eps'
    plt.savefig(plot_path)
    #fig_1.show()
    fig_2 = plt.figure('fig_2')
    l = plt.scatter(x_list,y_list,color='k',linestyle='--',s = 10, marker = 'x', label='E(infinite)')
    
    
    plt.title("E(converge)="+str(gs_converge))
    plt.ylabel("lg(E(infinte)-E(converge))")
    plt.legend(loc = 'lower left')
    #plt.ylim((1.2,2.8))
    #plt.savefig('Li6_radius_NN_prediction.jpg')
    plot_path = 'lowest_each_Nmax.eps'
    plt.savefig(plot_path)
    #fig_2.show()
    plt.close('all') 
   # import plot_gs as plot
    os.system('cp '+'different_hw.eps'+' '+output_path) 
    os.system('cp '+'lowest_each_Nmax.eps'+' '+output_path) 
    return gs_converge,loss,val_loss



#
# all NN parameters
#
nuclei = 'He4'
target_option = 'gs'
input_path = 'He4E_NNLOopt.txt'
#output_path = './result/gs/'
data_num = input_raw_data_count(input_path)
print 'data_num='+str(data_num)
# earlystopping parameters
monitor  = 'loss'
min_delta = 0.0001
patience = 30
epochs = 10000
batch_size = 32
input_dim = 2 
output_dim = 1
# interpolation setting
interpol_count = 10000
hw_line = 2
nmax_line = 1
gs_energy_line = 0
run_times_start = 1 
run_times_end   = 100
#parameter for gaussian distribution of sample_weight
sample_weight_switch = 'off'
FWHM = 100
sigma = FWHM/2.355 
#correlate parameters
corr_num   = 1
corr_weight= 1.


gs_converge_all = np.zeros(run_times_end)
loss_all = np.zeros(run_times_end)
val_loss_all = np.zeros(run_times_end)

os.system('mkdir '+nuclei)
os.system('mkdir '+nuclei+'/'+target_option)        


for max_nmax_fit in range(10,11,2):
    os.system('mkdir '+nuclei+'/'+target_option+'/gs-nmax4-'+str(max_nmax_fit))
    with open(nuclei+'/'+target_option+'/gs-nmax4-'+str(max_nmax_fit)+'/'+'gs_NN_info.txt','a') as f_3:
        #f_3.read()
        f_3.write('################################################################'+'\n')
        f_3.write('# loop   gs_energy            loss                  val_loss'+'\n')
    for loop_all in range(run_times_start-1,run_times_end):
        os.system('mkdir '+nuclei+'/'+target_option+'/gs-nmax4-'+str(max_nmax_fit)+'/'+str(loop_all+1))
        output_path = nuclei+'/'+target_option+'/gs-nmax4-'+str(max_nmax_fit)+'/'+str(loop_all+1)
        gs_converge_all[loop_all],loss_all[loop_all],val_loss_all[loop_all] = NN_all(input_path=input_path,output_path=output_path,data_num=data_num,monitor=monitor,min_delta=min_delta,patience=patience,epochs=epochs,batch_size=batch_size,input_dim=input_dim,output_dim=output_dim,interpol_count=interpol_count,max_nmax_fit=max_nmax_fit,sigma=sigma)
        with open(nuclei+'/'+target_option+'/gs-nmax4-'+str(max_nmax_fit)+'/'+'gs_NN_info.txt','a') as f_3:
            #f_3.read()
            f_3.write('{:>5}'.format(loop_all+1)+'   ')
            f_3.write('{:>-10.5f}'.format(gs_converge_all[loop_all])+'   ')
            f_3.write('{:>-20.15f}'.format(loss_all[loop_all])+'   ')
            f_3.write('{:>-20.15f}'.format(val_loss_all[loop_all])+'\n')


    
print 'gs_converge_all='+str(gs_converge_all)


#input()G
