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
                raw_data[loop2][4] = 1  # marker : raw_data = 1 , interpolation = 0
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
                #raw_data[loop2][4] = 1  # marker : raw_data = 1 , interpolation = 0
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



def NN_all(input_path,output_path,data_num,monitor,min_delta,patience,epochs,input_dim,output_dim,interpol_count,max_nmax_fit,sigma):
    
    #
    # Take in all the input data from file
    # 
    raw_data = np.zeros((data_num,5),dtype = np.float)  #the last column store the sample weight for raw_data
    input_file_1(input_path,raw_data,gs_energy_line,nmax_line,hw_line)
    #print raw_data
    #input() 

 
    #
    # To get more data, we do interpolation for the data
    #
    # interpolation for second colum
    # kind can be 'slinear', 'quadratic' and 'cubic' refer to a spline interpolation of first, second or third order)  
    kind = "cubic"
    nmax_max = int(np.max(raw_data[:,1]))
    nmax_min = int(np.min(raw_data[:,1]))
    nmax_count = (nmax_max-nmax_min)/2 + 1
    x_new = np.zeros((nmax_count,interpol_count))
    y_new = np.zeros((nmax_count,interpol_count))
   
    # interpolation for each namx 
    interpol_count_tot = 0   
    for loop1 in range(0,nmax_count):
        raw_data_new = raw_data[np.where(raw_data[:,1]==(loop1*2+nmax_min))]
        x = raw_data_new[:,2]
        y = raw_data_new[:,0]
        x_new[loop1,:] = np.linspace(np.min(x),np.max(x),interpol_count)
        #f=interpolate.interp1d(x,y,kind=kind)
        f=interpolate.CubicSpline(x,y,bc_type='not-a-knot', extrapolate=None)
        y_new[loop1,:] = f(x_new[loop1,:])
        x_new_count = len(x_new[loop1,:])-2
        interpol_count_tot = interpol_count_tot + x_new_count


    # joint all initerpoled data    
    data_interpolation = np.zeros((interpol_count_tot,5))
    min_position = np.zeros(nmax_count)
    count = 0
    for loop2 in range(0,nmax_count):
        for loop3 in range(1,interpol_count-1):  # here we ignore the first and last data points since these two is from the origin data.    
            data_interpolation[count,1] = loop2*2 + nmax_min
            data_interpolation[count,2] = x_new[loop2,loop3]
            data_interpolation[count,0] = y_new[loop2,loop3]
            data_interpolation[count,4] = 0   # marker raw_data = 1, interpolation =0
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
             data_interpolation[count_1,3] = normfun(x_new[loop2,loop3],min_position[loop2],sigma) 
             count_1 = count_1 +1 
         for loop4 in range(0,len(raw_data[np.where(raw_data[:,1]==(loop2*2+nmax_min))])):
             raw_data[count_2,3] = normfun(raw_data[count_2,2],min_position[loop2],sigma)
             count_2 = count_2 +1

    #print "raw_data_x="+str(raw_data[:,2])+"\n raw_data_y"+str(raw_data[:,0])
    #print "interpol_x="+str(data_interpolation[:,2])+"\n interpol_y"+str(data_interpolation[:,0])
    #input()
    #print "raw_data_x="+str(raw_data[:,1])+"  raw_data_sample_weithts="+str(raw_data[:,3])
    #input()

    #fig6 = plt.figure('fig6')
    #plt.scatter(raw_data[:,2],raw_data[:,0],color='b',s = 50, marker = '.', label='CC_calculation')
    #plt.scatter(data_interpolation[:,2],data_interpolation[:,0],color='y',s =10, marker = '.')
    #plt.plot(data_interpolation[30000:40000,2],data_interpolation[30000:40000,3],color='y',linestyle='--')
    #A = raw_data[np.where(raw_data[:,1]==10)]   
    #plt.scatter(A[:,2],A[:,3],color='y',s = 30, marker = '.', label='CC_calculation')
    #plt.ylim((0,0.2))  
    #plt.xlim((20,50))
    #fig6.show()
    #plt.close('all')
    #input()
   
    #
    # data_new = raw_data + data_interpolation
    #
    data_new_1 = np.vstack((raw_data,data_interpolation))
    data_new =  data_new_1[np.where((data_new_1[:,1]<=max_nmax_fit))]
 
    #raw_data_new = data_interpolation[np.where((data_interpolation[:,1]<=max_nmax_fit))]
    #print "raw_data="+str(raw_data)
    #print "data_interpolation="+str(data_interpolation)
    #print "data_new="+str(data_new)
    #input()
 
    #
    # shuffle the data
    #
    #np.random.shuffle(data_new)
    #np.random.shuffle(data_interpolation)
    #np.random.shuffle(raw_data)


    #print len(raw_data)
    #print raw_data    
    #batch_size = data_num
    input_shape = (input_dim,)
    input_data  = Input(shape = input_shape)
    c_matrix    = Input(shape = (1,)) 
    y_test      = Input(shape = (1,))

    #raw_data_new = data_interpolation[np.where((data_interpolation[:,1]<=max_nmax_fit)&(data_interpolation[:,2]<=60))]
    #raw_data_new = data_interpolation
    #print "raw_data_new="+str(raw_data_new)
    
    x_train = data_new[:,1:3]
    y_train = data_new[:,0]
    
    #print "x_shape"+str(x_train.shape)
    #print "x_train = "+str(x_train)
    #print "y_train = "+str(y_train)
    #input()

    #
    # Correlate matrix setup
    #
    
    correlate_matrix = np.zeros((len(data_new),1))    
    #correlate_matrix = np.zeros((len(data_new),len(data_new)))    
    #correlate_matrix = np.eye(90000,dtype=int)    
#    for nmax_count in range(0,9):
#        for i in range(nmax_count*interpol_count,90000):
#            for j in range(0,90000):
#                correlate_matrix(i,j)=
#    
#    print "data_new="+str(len(data_new))
#    input()

    #def correlate_ab(a,b):
    #    w_ab = 0
    #    if ((a[0]==b[0])&(a[1]==b[1])&(a[2]==b[2])):
    #        w_ab = 1
    #    if (a[1] == b[1]):
    #        if(a[4]+b[4] == 1):
    #            w_ab = 1
    #    return w_ab
    #
    #for loop1 in range(0,len(data_new)):
    #    for loop2 in range(0,len(data_new)):
    #        correlate_matrix[loop1,loop2] = correlate_ab(data_new[loop1,:],data_new[loop2,:])
    #number_count_1=1
    #number_count_2=145
    #print "a="+str(data_new[number_count_1])
    #print "b="+str(data_new[number_count_2])
    #print "c_ab="+str(correlate_matrix[number_count_1,number_count_2])    
    ##input()


    #
    # NN Model
    # 
    x = Dense(8, activation = 'sigmoid')(input_data)
    predictions = Dense(output_dim)(x)
    #loss = Lambda(lambda k: K.dot(K.transpose(k[0]),k[1]),k[0])([predictions,c_matrix])
    #loss_layer = Lambda(lambda k: K.sum(K.dot(k[0],k[1])))([c_matrix,predictions])
    def correlated_loss(k): 
        #c_loss = K.dot(K.transpose(k[1]-k[2]),(k[1]-k[2]))
        c_loss = K.sum(K.batch_dot((k[1]-k[2]),(k[1]-k[2])))
        #A = np.zeros((45,45))
        #for i in range(0,44)
        #    A
        #B = K.dot(K.transpose(k[1]),A)
        #c_loss = K.dot(B,k[2])
        return K.sum(k[1]*0) 



    #loss_layer = Lambda(lambda k: K.dot(K.transpose(k[1]-k[2]),k[1]-k[2]))([c_matrix,predictions,y_test])
    loss_layer = Lambda(correlated_loss,name='correlated_loss')([c_matrix,predictions,y_test])
    model = Model(inputs=[input_data,c_matrix,y_test], outputs = loss_layer)
    
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
    #    F=K.sum(C)
    #    #G=tf.mul(C,C)
    #    #H=K.sum(G) 
    #    return y_pred
#K.dot(y_true,y_pred)

    # 
    # use y_pred as loss, just minimize the y_pred as we already set the correct form of loss in loss_layer
    # 

    model.compile(optimizer='adam',loss=lambda y_true,y_pred: y_pred)
    #model.compile(optimizer='adam',loss=correlate_loss)
    
    early_stopping = EarlyStopping(monitor=monitor,min_delta = min_delta , patience=patience, verbose=0, mode='min')
    
    #history = model.fit(x_train,y_train, epochs = epochs, validation_split = 0.01 , shuffle = 1, callbacks=[early_stopping], sample_weight = raw_data_new[:,3])
    #history = model.fit([x_train,correlate_matrix], y_train, epochs = epochs,batch_size=len(y_train),  callbacks=[early_stopping], sample_weight = raw_data_new[:,3])
    history = model.fit([x_train,correlate_matrix,y_train], y_train, epochs = epochs, shuffle =0, callbacks=[early_stopping], sample_weight = data_new[:,3])
    
    loss = history.history['loss'][len(history.history['loss'])-1]
    # temp change
    #val_loss = history.history['val_loss'][len(history.history['val_loss'])-1]
    val_loss = 0 

    #print "y_train_len="+str(len(y_train))
    #input()
    #fig4 = plt.figure('fig4')
    #plt.plot(history.history['loss'])
    #plt.plot(history.history['val_loss'])
    #plt.savefig('loss_val_loss.eps')
    plt.close('all')


    model_path = 'gs.h5' 
    model.save(model_path)
  
    predict_model = Model(inputs= input_data, outputs = predictions)

 
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
    
    y_test = predict_model.predict(x_test)
    
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
    #os.system('cp '+'loss_val_loss.eps'+' '+output_path) 
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
patience = 50
epochs = 100
input_dim = 2 
output_dim = 1
# interpolation setting
interpol_count =1000
hw_line = 2
nmax_line = 1
gs_energy_line = 0
run_times_start = 1 
run_times_end   = 100
#parameter for gaussian distribution of sample_weight
FWHM = 15
sigma = FWHM/2.355 

gs_converge_all = np.zeros(run_times_end)
loss_all = np.zeros(run_times_end)
val_loss_all = np.zeros(run_times_end)

os.system('mkdir '+nuclei)
os.system('mkdir '+nuclei+'/'+target_option)        


for max_nmax_fit in range(20,21,2):
    os.system('mkdir '+nuclei+'/'+target_option+'/gs-nmax4-'+str(max_nmax_fit))
    with open(nuclei+'/'+target_option+'/gs-nmax4-'+str(max_nmax_fit)+'/'+'gs_NN_info.txt','a') as f_3:
        #f_3.read()
        f_3.write('#######################################################'+'\n')
        f_3.write('# loop   gs_energy          loss             val_loss  '+'\n')
    for loop_all in range(run_times_start-1,run_times_end):
        os.system('mkdir '+nuclei+'/'+target_option+'/gs-nmax4-'+str(max_nmax_fit)+'/'+str(loop_all+1))
        output_path = nuclei+'/'+target_option+'/gs-nmax4-'+str(max_nmax_fit)+'/'+str(loop_all+1)
        gs_converge_all[loop_all],loss_all[loop_all],val_loss_all[loop_all] = NN_all(input_path=input_path,output_path=output_path,data_num=data_num,monitor=monitor,min_delta=min_delta,patience=patience,epochs=epochs,input_dim=input_dim,output_dim=output_dim,interpol_count=interpol_count,max_nmax_fit=max_nmax_fit,sigma=sigma)
        with open(nuclei+'/'+target_option+'/gs-nmax4-'+str(max_nmax_fit)+'/'+'gs_NN_info.txt','a') as f_3:
            #f_3.read()
            f_3.write('{:>5}'.format(loop_all+1)+'   ')
            f_3.write('{:>-10.5f}'.format(gs_converge_all[loop_all])+'   ')
            f_3.write('{:>-15.10f}'.format(loss_all[loop_all])+'   ')
            f_3.write('{:>-15.10f}'.format(val_loss_all[loop_all])+'\n')


    
print 'gs_converge_all='+str(gs_converge_all)


#input()
