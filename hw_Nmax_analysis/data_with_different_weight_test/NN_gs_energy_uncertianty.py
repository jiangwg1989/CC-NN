import keras
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import tensorflow as tf


from matplotlib.ticker import MultipleLocator, FormatStrFormatter
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
from compiler.ast import flatten
from random import gauss



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
                loop2 = loop2 + 1
            loop1 = loop1 + 1

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

#
# unflattern the weights
#
def unflattern_weights(model_str,weights_new,weights):
    temp = 0
    for loop1 in range(model_str[0]):
        for loop2 in range(model_str[1]):
            weights[0][loop1][loop2]  = weights_new[temp]  
            temp = temp + 1
    for loop3 in range(model_str[1]):
        weights[1][loop3] = weights_new[temp]  
        temp = temp + 1
    for loop4 in range(model_str[1]):
        for loop5 in range(model_str[2]):
            weights[2][loop4][loop5] = weights_new[temp] 
            temp = temp + 1
    for loop6 in range(model_str[2]):
        weights[3][loop6] = weights_new[temp]  
        temp = temp + 1
    return weights



#
# 
#
def output_loss(my_model,x_in,y_in,sample_weights):
    y_pre = my_model.predict(x_in)
#    print("y_pre"+str(y_pre))
#    print("y_in"+str(y_in))
    mse_loss = 1
    y_pre = y_pre.T
    D     = y_in-y_pre
#    print(D)
    D_2   = D**2
    D_2   = D_2 * sample_weights
    mse_loss = D_2.mean(axis=1)
    return mse_loss

def make_random_vec(dims):
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return [x/mag for x in vec]

#
# walk with random_vec, find out how far could we go the make the loss double. 
#
def random_vec_walk_2timeloss(my_model,model_str,x_in,y_in,sample_weights,origin_flat_weights,origin_unflat_weights,origin_loss,random_vec):
    def new_loss_cal(a):
        weights_flat_temp = origin_flat_weights + random_vec*a
        weights_unflat_temp = unflattern_weights(model_str=model_str,weights_new=weights_flat_temp,weights=origin_unflat_weights)    
        my_model.set_weights(weights_unflat_temp)
        loss_new = output_loss(my_model,x_in,y_in,sample_weights)
        return(loss_new)

    #  try 0, 0.05, 0.1
#    x = [0,0.005,0.01]
#    y = np.zeros(len(x))
#    for loop1 in range(len(x)):
#        y[loop1] = new_loss_cal( a = x[loop1]) 
#    #print("output_loss_test="+str(x))
##    print("output_loss_test="+str(y))
#
#    f1 = np.polyfit(y,x,2)
#    p1 = np.poly1d(f1) 
#    x_new = p1(10*y[0]) 
#    k = new_loss_cal( a = x_new)    
   
#    print('x_new='+str(x_new)+"  loss_new="+str(k))
    
    a_old = 0
    a_new = 0.005
    y_old = origin_loss
    y_new = new_loss_cal( a= a_new)
    y_target = 2* origin_loss    # two time loss
    target_precision = 0.1

    loop_max = 0
    while (np.abs( y_new-y_target ) > (target_precision*origin_loss) ) :

        if (y_new > y_target ):
            a_temp = (a_new+a_old)/2
            y_temp = new_loss_cal( a = a_temp )
            if (y_temp > y_target):
                a_new = a_temp
            else:
                a_old = a_temp
        else:
            a_temp = a_new + (a_new-a_old)/2
            y_temp = new_loss_cal( a = a_temp)
            a_new  = a_temp

        y_new = y_temp

        loop_max = loop_max + 1
        if (loop_max == 100):
            break

#
#  calculate predict gs value at new weights (double the loss)
#
    weights_flat_temp = origin_flat_weights + random_vec*a_new
    weights_unflat_temp = unflattern_weights(model_str=model_str,weights_new=weights_flat_temp,weights=origin_unflat_weights)    
    my_model.set_weights(weights_unflat_temp)
    gs_new = model_predict_gs(my_model)

    #print("a_new ="+str(a_new)+"  loss_new="+str(y_new)+"  gs_new="+str(gs_new))
    return  gs_new
    
def model_predict_gs(my_model):
    #
    # get model prediction
    #
    count = len(range(4,204,1))*len(range(5,121,1))
    x_test = np.zeros((count,2),dtype = np.float)
    
    loop3 = 0
    for loop1 in range(4,204,1):
        for loop2 in range(5,121,1):
            x_test[loop3][0] = loop1 
            x_test[loop3][1] = loop2 
            loop3 = loop3 + 1
    
    y_test = my_model.predict(x_test)
    raw_predic_data = np.concatenate((y_test,x_test),axis=1)
    temp = (raw_predic_data[np.where(raw_predic_data[:,1]== 200)])
    gs_converge = np.min(temp[:,0])

    return gs_converge




#def NN_all(input_path,output_path,data_num,monitor,min_delta,patience,epochs,input_dim,output_dim,interpol_count,max_nmax_fit,sigma):
def NN_uncertainty(model_h5file_path,max_nmax_fit,input_dim):   
    #
    # Take in all the input data from file
    # 
    raw_data = np.zeros((data_num,3),dtype = np.float)
    input_file_1(raw_data_path,raw_data,gs_energy_line,nmax_line,hw_line)
   

 
    #
    # To get more data, we do interpolation for the data
    #
    # interpolation for second colum
    # kind can be 'slinear', 'quadratic' and 'cubic' refer to a spline interpolation     of first, second or third order)  
    kind = "quadratic"
    nmax_max = int(np.max(raw_data[:,1]))
    nmax_min = int(np.min(raw_data[:,1]))
    nmax_count = int((nmax_max-nmax_min)/2 + 1)
    print("nmax_count="+str(nmax_count))
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
 
    # joint all initerpoled data    
    data_interpolation = np.zeros((interpol_count_tot,4))
    min_position = np.zeros(nmax_count)
    count = 0
    for loop2 in range(0,nmax_count):
        for loop3 in range(0,interpol_count):
            data_interpolation[count,1] = loop2*2 + nmax_min
            data_interpolation[count,2] = x_new[loop2,loop3]
            data_interpolation[count,0] = y_new[loop2,loop3]
            #data_interpolation[count,3] = normfun(x_new[loop2,loop3],mu,sigma)
            count = count +1
        #print "max_y="+str(x_new[loop2,np.where(y_new[loop2,:]==np.min(y_new[loop2,:]))])
        min_position[loop2] = x_new[loop2,np.where(y_new[loop2,:]==np.min(y_new[loop2,:]))]


    count = 0
    for loop2 in range(0,nmax_count):
         for loop3 in range(0,interpol_count):
             if(sample_weight_switch   == 'on'):
                 data_interpolation[count,3] = normfun(x_new[loop2,loop3],min_position[loop2],sigma) 
             elif(sample_weight_switch == 'off'):
                  data_interpolation[count,3] = 1 
             else:
                 print('sample_weight_switch error!') 
             count = count +1 
#
#
#    
#    #fig6 = plt.figure('fig6')
#    #plt.plot(data_interpolation[30000:40000,2],data_interpolation[30000:40000,3],color='y',linestyle='--')
#    #plt.ylim((-29,-23))  
#    #plt.xlim((10,50))
#    #fig6.show()
#    #plt.close('all')
#
#   
#    #print min_position
#    #input()
#
#         
#    #
#    # shuffle the data
#    #
#    np.random.shuffle(data_interpolation)
#    np.random.shuffle(raw_data)
    
    #print len(raw_data)
    #print raw_data
    
    #batch_size = data_num
    
    input_shape = (input_dim,)
    input_data = Input(shape = input_shape)
    
    raw_data_new  = data_interpolation[np.where((data_interpolation[:,1]<=max_nmax_fit))]
#    #raw_data_new = data_interpolation[np.where((data_interpolation[:,1]<=max_nmax_fit)&(data_interpolation[:,2]<=60))]
#    #raw_data_new = data_interpolation
#    
    x_train = raw_data_new[:,1:3]
    y_train = raw_data_new[:,0]
##    y_train = y_train*0 
##    print(y_train)

    #print "x_train = "+str(x_train)
    #print "y_train = "+str(y_train)

    def test_f(k):
        A = tf.zeros_like(k)
        D = tf.ones_like(k)
        B = k*A  # now B is [[0][0][0]...]
        C = B+D  #
        print(C)
        #input()    
        return C*2

    
    #
    # NN Model
    # 
    x = Dense(8, activation = 'sigmoid')(input_data)
    predictions = Dense(output_dim)(x)
#    predictions = Lambda(test_f,name='test_f')(predictions)

    model = Model(inputs= input_data, outputs = predictions)
    
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
 

#    model.compile(optimizer='adam',loss= 'mse')# ,metrics=['accuracy'])
#    
#    early_stopping = EarlyStopping(monitor=monitor,min_delta = min_delta , patience=patience, verbose=0, mode='min')
#    
#    history = model.fit(x_train,y_train, epochs = epochs, validation_split = 0.01 , shuffle = 1,batch_size =32 , callbacks=[early_stopping], sample_weight = raw_data_new[:,3])
#    loss = history.history['loss'][len(history.history['loss'])-1]
#    val_loss = history.history['val_loss'][len(history.history['val_loss'])-1]
#    
#    fig4 = plt.figure('fig4')
#    plt.plot(history.history['loss'])
#    plt.plot(history.history['val_loss'])
#    #plt.savefig('loss_val_loss.eps')
#    plt.close('all')
#
#
#    model_path = 'gs.h5' 
#    model.save(model_path)
   
    #
    # load model
    #
    model_str=[2,8,1]
    total_parameter_num = model_str[0]*model_str[1]+model_str[1]+ model_str[1]*model_str[2]+model_str[2]
    print(total_parameter_num)
    my_model = load_model(model_h5file_path)

    # get the weights from the model (its a list)
    weights_unflat_origin  = my_model.get_weights() 

    #
    # flattern the weights
    #
    weights_flat_origin = np.zeros(total_parameter_num) 
    temp = 0
    for loop1 in range(model_str[0]):
        for loop2 in range(model_str[1]):
            weights_flat_origin[temp] = weights_unflat_origin[0][loop1][loop2] 
            temp = temp + 1
    for loop3 in range(model_str[1]):
        weights_flat_origin[temp] = weights_unflat_origin[1][loop3] 
        temp = temp + 1
    for loop4 in range(model_str[1]):
        for loop5 in range(model_str[2]):
            weights_flat_origin[temp] = weights_unflat_origin[2][loop4][loop5] 
            temp = temp + 1
    for loop6 in range(model_str[2]):
        weights_flat_origin[temp] = weights_unflat_origin[3][loop6] 
        temp = temp + 1
 
#    print(weights)
#    print(weights_flat_origin)



#    #
#    # unflattern the weights
#    #
#    temp = 0
#    for loop1 in range(model_str[0]):
#        for loop2 in range(model_str[1]):
#            weights[0][loop1][loop2]  = weights_flat_origin[temp]  
#            temp = temp + 1
#    for loop3 in range(model_str[1]):
#        weights[1][loop3] = weights_flat_origin[temp]  
#        temp = temp + 1
#    for loop4 in range(model_str[1]):
#        for loop5 in range(model_str[2]):
#            weights[2][loop4][loop5] = weights_flat_origin[temp] 
#            temp = temp + 1
#    for loop6 in range(model_str[2]):
#        weights[3][loop6] = weights_flat_origin[temp]  
#        temp = temp + 1

#    weights_flat_origin[1] = weights_flat_origin[1]-1




    weights_unflat_temp = unflattern_weights(model_str=model_str,weights_new=weights_flat_origin,weights=weights_unflat_origin)    
    my_model.set_weights(weights_unflat_temp)

    origin_loss = output_loss(my_model=my_model,x_in=x_train,y_in=y_train,sample_weights=raw_data_new[:,3])
 
    origin_gs   = model_predict_gs(my_model)

    print("origin_loss="+str(origin_loss)) 

    print("origin_gs="+str(origin_gs))
##
##  first derivative
##   
#    f_x = np.zeros(len(weights_new))
#    h = 0.0001
#
#    for loop1 in range(len(weights_new)):
#        weights_temp = weights_new
#        weights_temp[loop1] = weights_new[loop1]+h
#        my_model.set_weights(unflattern_weights(model_str=model_str,weights_new=weights_temp,weights=weights))        
#        loss_add_h = output_loss(my_model=my_model,x_in=x_train,y_in=y_train,sample_weights=raw_data_new[:,3])           
#        
#        weights_temp = weights_new
#        weights_temp[loop1] = weights_new[loop1]-h
#        my_model.set_weights(unflattern_weights(model_str=model_str,weights_new=weights_temp,weights=weights))        
#        loss_minus_h = output_loss(my_model=my_model,x_in=x_train,y_in=y_train,sample_weights=raw_data_new[:,3])           
#        f_x[loop1] = (loss_add_h - loss_minus_h )/(2*h) 
#
#    print(f_x) 

#
# generate random vec 
#
    random_vec_num = 100
    random_vec = np.zeros((len(weights_flat_origin),random_vec_num)) 
    for loop1 in range(random_vec_num):
        random_vec[:,loop1]=make_random_vec(len(weights_flat_origin))
    #print (random_vec)

#
# calculate double loss walk along random vec direction.
#
    
    #print("lalal="+str(random_vec[:,1]))

    model_pre = np.zeros(random_vec_num)
    for loop1 in range(random_vec_num):
        model_pre[loop1] = random_vec_walk_2timeloss(my_model=my_model,model_str=model_str,x_in=x_train,y_in=y_train,sample_weights=raw_data_new[:,3],origin_flat_weights=weights_flat_origin,origin_unflat_weights=weights_unflat_origin,origin_loss=origin_loss,random_vec=random_vec[:,loop1])
    print ("mean_model_pre_gs_error = "+str(np.mean(model_pre)))


    # calculate the variant of the new predict gs with two times loss
    variant = 0
    for loop1 in range(random_vec_num):
        variant = variant + np.square(model_pre[loop1]-origin_gs)        
    variant = variant / random_vec_num
    variant = np.sqrt(variant)
    print ("new_pre_gs_variant = "+str(variant))

#    #
#    # get model prediction
#    #
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
#    #print x_test
#    
#    y_test = my_model.predict(x_test)
#    
#    raw_predic_data = np.concatenate((y_test,x_test),axis=1)
#    
#    #print "raw_predic_data="+str(raw_predic_data)
#    
#    #fig,(ax0,ax1) = plt.subplots(nrows = 2, figsize=(9,9))
#    
##    x_list_1 = raw_data[:,2] 
##    y_list_1 = raw_data[:,0]
##    
##    raw_predic_data_4 = raw_predic_data[np.where(raw_predic_data[:,1]==4)]
##    raw_predic_data_8 = raw_predic_data[np.where(raw_predic_data[:,1]==8)]
##    raw_predic_data_12 = raw_predic_data[np.where(raw_predic_data[:,1]==12)]
##    raw_predic_data_20 = raw_predic_data[np.where(raw_predic_data[:,1]==20)]
##    raw_predic_data_40 = raw_predic_data[np.where(raw_predic_data[:,1]==40)]
##    raw_predic_data_60 = raw_predic_data[np.where(raw_predic_data[:,1]==100)]
#    
#    temp = (raw_predic_data[np.where(raw_predic_data[:,1]== 200)])
#    gs_converge = np.min(temp[:,0])
 
    
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
#    ax = plt.subplot(111)
#    l1=plt.scatter(x_list_1,y_list_1,color='k',linestyle='--',s = 10, marker = 'x', label='CC_calculation')
#    l2=plt.plot(x_list_2,y_list_2,color='y',linestyle='--',label='NN_Nmax_4')
#    l3=plt.plot(x_list_3,y_list_3,color='r',linestyle='--',label='NN_Nmax_8')
#    l4=plt.plot(x_list_4,y_list_4,color='g',linestyle='--',label='NN_Nmax_12')
#    l5=plt.plot(x_list_5,y_list_5,color='c',linestyle='--',label='NN_Nmax_20')
#    
#    l6=plt.plot(x_list_6,y_list_6,color='m',linestyle='--',label='NN_Nmax_40')
#    l7=plt.plot(x_list_7,y_list_7,color='b',linestyle='--',label='NN_Nmax_100')
#    #l4=fig1.scatter(x_list_2,y_list_2,color='y',linestyle='--',marker=',')
#    #l5=fig1.scatter(x_list_2,y_list_2,color='r',linestyle='--',marker=',')
#    #l6=fig1.scatter(x_list_2,y_list_2,color='c',linestyle='--',marker=',')
#    #fig1.scatter(x_list_2,y_list_2,color='m',linestyle='--',marker=',')
#    #plt.legend(loc = 'upper left')
#
#    xmajorLocator   = MultipleLocator(10)
#    #xmajorFormatter = FormatStrFormatter('%5f')
#    xminorLocator   = MultipleLocator(2)
#    
#    
#    ymajorLocator   = MultipleLocator(5)
#    #ymajorFormatter = FormatStrFormatter('%1.1d')
#    yminorLocator   = MultipleLocator(1)
#
#    ax.xaxis.set_major_locator(xmajorLocator)
#    #ax.xaxis.set_major_formatter(xmajorFormatter)
#    ax.yaxis.set_major_locator(ymajorLocator)
#    #ax.yaxis.set_major_formatter(ymajorFormatter)
#    ax.xaxis.set_minor_locator(xminorLocator)
#    ax.yaxis.set_minor_locator(yminorLocator)
#    ax.xaxis.grid(True, which='major') 
#    ax.yaxis.grid(True, which='major')
#    
#
#    plt.legend(loc='upper right', bbox_to_anchor=(1.5,0.75),ncol=1,fancybox=True,shadow=True,borderaxespad = 0.)
#    plt.title("gs(infinite)="+str(gs_converge))
#    plot_path = 'gs.eps'
#    plt.ylim((-29,-23))  
#    plt.xlim((10,50))
#    plt.subplots_adjust(right = 0.7)
#    plt.savefig(plot_path)
#    plt.close('all')
#    #fig1.show()
#    
#    
#    file_path = "gs_NN_prediction.txt"
#    with open(file_path,'w') as f_1:
#        for loop1 in range(1,count):
#            f_1.write('{:>-10.5f}'.format(y_test[loop1,0]))
#            f_1.write('{:>10}'.format(x_test[loop1,0]))
#            f_1.write('{:>10}'.format(x_test[loop1,1])+'\n')
#    os.system('cp '+file_path+' '+output_path)        
#    os.system('cp '+model_path+' '+output_path) 
#    os.system('cp '+'gs.eps'+' '+output_path) 
#    os.system('cp '+'loss_val_loss.eps'+' '+output_path) 
##
## plot different_hw.eps and lowest_each_Nmax.eps
##
#    data_num = len(open(file_path,'rU').readlines())
#    raw_data = np.zeros((data_num,3),dtype = np.float)
#    input_file_2(file_path,raw_data)
#    
#    raw_data_new_4 = raw_data[np.where(raw_data[:,2] == 50 )]
#    raw_data_new_3 = raw_data[np.where(raw_data[:,2] == 40 )]
#    raw_data_new_2 = raw_data[np.where(raw_data[:,2] == 30 )]
#    raw_data_new_1 = raw_data[np.where(raw_data[:,2] == 20 )]
#    
#    raw_data_new_5 = raw_data[np.where(raw_data[:,1] == 200)]
#    temp_1 = raw_data_new_5[:,0]
#    gs_converge = np.min(temp_1)
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
#    fig_1 = plt.figure('fig_1')
#    l1 = plt.scatter(x_list_1,y_list_1,color='k',linestyle='--',s = 10, marker = 'x', label    ='NN_prediction_hw=20')
#    l2 = plt.scatter(x_list_2,y_list_2,color='r',linestyle='--',s = 10, marker = 'x', label    ='NN_prediction_hw=30')
#    l3 = plt.scatter(x_list_3,y_list_3,color='g',linestyle='--',s = 10, marker = 'x', label    ='NN_prediction_hw=40')
#    l4 = plt.scatter(x_list_4,y_list_4,color='b',linestyle='--',s = 10, marker = 'x', label    ='NN_prediction_hw=50')
#    
#    plt.title("E(converge)="+str(gs_converge))
#    
#    plt.ylabel("lg(E(infinte)-E(converge))")
#    plt.legend(loc = 'lower left')
#    #plt.ylim((1.2,2.8))
#    #plt.savefig('Li6_radius_NN_prediction.jpg')
#    plot_path = 'different_hw.eps'
#    plt.savefig(plot_path)
#    #fig_1.show()
#    fig_2 = plt.figure('fig_2')
#    l = plt.scatter(x_list,y_list,color='k',linestyle='--',s = 10, marker = 'x', label='E(infinite)')
#    
#    
#    plt.title("E(converge)="+str(gs_converge))
#    plt.ylabel("lg(E(infinte)-E(converge))")
#    plt.legend(loc = 'lower left')
#    #plt.ylim((1.2,2.8))
#    #plt.savefig('Li6_radius_NN_prediction.jpg')
#    plot_path = 'lowest_each_Nmax.eps'
#    plt.savefig(plot_path)
#    #fig_2.show()
#    plt.close('all') 
#   # import plot_gs as plot
#    os.system('cp '+'different_hw.eps'+' '+output_path) 
#    os.system('cp '+'lowest_each_Nmax.eps'+' '+output_path) 

    #print(gs_converge)
    return variant



#
# all NN parameters
#
nuclei = 'Li6'
target_option = 'gs'
raw_data_path = 'Li6E_NNLOopt.txt'
#output_path = './result/gs/'
data_num = input_raw_data_count(raw_data_path)
# earlystopping parameters
monitor  = 'loss'
min_delta = 0.0001
patience = 30
epochs = 10000
input_dim = 2 
output_dim = 1
# interpolation setting
interpol_count = 100
hw_line = 2
nmax_line = 1
gs_energy_line = 0
run_times_start = 1 
run_times_end   = 150
#parameter for gaussian distribution of sample_weight
sample_weight_switch = 'on'
FWHM = 20
sigma = FWHM/2.355
 

gs_converge_all = np.zeros(run_times_end)
loss_all = np.zeros(run_times_end)
val_loss_all = np.zeros(run_times_end)




#for loop1 in range(100):
loop1 = 3
max_nmax_fit = 22
folder_num   = 5
new_pre_gs_variant = np.zeros(folder_num)


for loop1 in range (folder_num):

    model_h5file_path = "/home/slime/work/CC/hw_Nmax_analysis/data_with_different_weight_test/"+nuclei+"/"+target_option+"/gs-nmax4-"+str(max_nmax_fit)+"_new_balance/"+str(loop1+1)+"/gs.h5"
    active = os.path.exists(model_h5file_path)
    print active
    if (active == True ):
        new_pre_gs_variant[loop1] = NN_uncertainty(model_h5file_path=model_h5file_path,max_nmax_fit=max_nmax_fit,input_dim=input_dim)
    else:
        new_pre_gs_variant[loop1] = 0

mean_variant = 0
mean_variant_count = 0

for loop1 in range (folder_num):
    if (new_pre_gs_variant[loop1] != 0):
        mean_variant = mean_variant +  new_pre_gs_variant[loop1]
        mean_variant_count = mean_variant_count + 1
mean_variant = mean_variant / mean_variant_count


with open('uncertainty_analysis.txt','w') as f_1:
    f_1.write('#############################################'+'\n')
    f_1.write('# loop    new_pre_gs_variant  '+'\n')
    for loop1 in range (folder_num):
        f_1.write("   "+str(loop1+1)+'    '+str(new_pre_gs_variant[loop1])+'\n')
    f_1.write('#############################################'+'\n')
    f_1.write('mean_variant = '+str(mean_variant))




#os.system('mkdir '+nuclei)
#os.system('mkdir '+nuclei+'/'+target_option)        


#for max_nmax_fit in range(12,7,-2):
#    os.system('mkdir '+nuclei+'/'+target_option+'/gs-nmax4-'+str(max_nmax_fit))
#    with open(nuclei+'/'+target_option+'/gs-nmax4-'+str(max_nmax_fit)+'/'+'gs_NN_info.txt','a') as f_3:
#        #f_3.read()
#        f_3.write('#############################################'+'\n')
#        f_3.write('# loop   gs_energy       loss       val_loss'+'\n')
#    for loop_all in range(run_times_start-1,run_times_end):
#        os.system('mkdir '+nuclei+'/'+target_option+'/gs-nmax4-'+str(max_nmax_fit)+'/'+str(loop_all+1))
#        output_path = nuclei+'/'+target_option+'/gs-nmax4-'+str(max_nmax_fit)+'/'+str(loop_all+1)
#        gs_converge_all[loop_all],loss_all[loop_all],val_loss_all[loop_all] = NN_all(input_path=input_path,output_path=output_path,data_num=data_num,monitor=monitor,min_delta=min_delta,patience=patience,epochs=epochs,input_dim=input_dim,output_dim=output_dim,interpol_count=interpol_count,max_nmax_fit=max_nmax_fit,sigma=sigma)
#        with open(nuclei+'/'+target_option+'/gs-nmax4-'+str(max_nmax_fit)+'/'+'gs_NN_info.txt','a') as f_3:
#            #f_3.read()
#            f_3.write('{:>5}'.format(loop_all+1)+'   ')
#            f_3.write('{:>-10.5f}'.format(gs_converge_all[loop_all])+'   ')
#            f_3.write('{:>-20.15f}'.format(loss_all[loop_all])+'   ')
#            f_3.write('{:>-20.15f}'.format(val_loss_all[loop_all])+'\n')


    


#input()
