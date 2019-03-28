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



def NN_all(input_path,output_path,data_num,monitor,min_delta,patience,epochs,input_dim,output_dim,interpol_count,max_nmax_fit,FWHM_percent):
    
    #
    # Take in all the input data from file
    # 
    raw_data = np.zeros((data_num,3),dtype = np.float)
    input_file_1(input_path,raw_data,gs_energy_line,nmax_line,hw_line)
   

 
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


##
##  second way of doing gussian sample weights, according to the cross point with last Nmax
##
# find the cross point of largest nmax with the last but one Nmax
    x_cross_point = 0 
    y_cross_point = 0 
    
    raw_data_new = raw_data[np.where(raw_data[:,1]==(nmax_max))]
    line_1_x = raw_data_new[:,2]
    line_1_y = raw_data_new[:,0]
    raw_data_new = raw_data[np.where(raw_data[:,1]==(nmax_max-2))]
    line_2_x = raw_data_new[:,2]
    line_2_y = raw_data_new[:,0]

#   radius_range is the FWHM 
    radius_range =np.abs((np.max(line_1_y)-np.min(line_1_y)+np.max(line_1_y)-np.min(line_1_y))/2.)

#  balance x and y
    regulator = (np.max(line_1_x)-np.min(line_1_x)) / (np.max(line_1_y)-np.min(line_1_y))
#    print(regulator)
    temp_min = pow((line_1_x[0]/regulator-line_2_x[0]/regulator),2)+pow(line_1_y[0] - line_2_y[0],2)
    for loop1 in range(0,len(line_1_x)):
        for loop2 in range(0,len(line_1_x)):
            temp = pow((line_1_x[loop1]/regulator-line_2_x[loop2]/regulator),2)+pow(line_1_y[loop1] - line_2_y[loop2],2) 
            #print("temp="+str(temp))
            if(temp < temp_min):
                temp_min = temp
                x_cross_point = line_1_x[loop1]#(line_1_x[loop1] + line_2_x[loop2])/2.
                y_cross_point = line_1_y[loop1]#(line_1_y[loop1] + line_2_y[loop2])/2. 

    print ("x_cross_point="+str(x_cross_point))
    print ("y_cross_point="+str(y_cross_point))



    count = 0
    for loop2 in range(0,nmax_count):
         for loop3 in range(0,interpol_count):
             if(sample_weight_switch   == 'on'):
                 data_interpolation[count,3] = normfun(y_new[loop2,loop3],y_cross_point, radius_range*FWHM_percent ) # gaussian distribute accroding to y_cross_point value
             elif(sample_weight_switch == 'off'):
                 data_interpolation[count,3] = 1
             else:
                 print('sample_weight_switch error!')
             count = count +1



#    for loop2 in range(0,nmax_count):
#         for loop3 in range(0,interpol_count):
#             data_interpolation[count,3] = normfun(x_new[loop2,loop3],min_position[loop2],sigma) 
#             count = count +1 
    
    #fig6 = plt.figure('fig6')
    #plt.plot(data_interpolation[30000:40000,2],data_interpolation[30000:40000,3],color='y',linestyle='--')
    #plt.ylim((-29,-23))  
    #plt.xlim((10,50))
    #fig6.show()
    #plt.close('all')

   
    #print min_position
    #input()

         
    #
    # shuffle the data
    #
    np.random.shuffle(data_interpolation)
    np.random.shuffle(raw_data)
    
    #print len(raw_data)
    #print raw_data
    
    #batch_size = data_num
    
    input_shape = (input_dim,)
    input_data = Input(shape = input_shape)
    
    raw_data_new  = data_interpolation[np.where((data_interpolation[:,1]<=max_nmax_fit))]
    #raw_data_new = data_interpolation[np.where((data_interpolation[:,1]<=max_nmax_fit)&(data_interpolation[:,2]<=60))]
    #raw_data_new = data_interpolation
    
    x_train = raw_data_new[:,1:3]
    y_train = raw_data_new[:,0]
#    y_train = y_train*0 
#    print(y_train)

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
 

    model.compile(optimizer='adam',loss= 'mse')# ,metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor=monitor,min_delta = min_delta , patience=patience, verbose=0, mode='min')
    
    history = model.fit(x_train,y_train, epochs = epochs, validation_split = 0.01 , shuffle = 1,batch_size =32 , callbacks=[early_stopping], sample_weight = raw_data_new[:,3],verbose=0)
    loss = history.history['loss'][len(history.history['loss'])-1]
    val_loss = history.history['val_loss'][len(history.history['val_loss'])-1]
    
    fig4 = plt.figure('fig4')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    #plt.savefig('loss_val_loss.eps')
    plt.close('all')


    model_path = 'gs.h5' 
    model.save(model_path)
   
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
    
    y_test = model.predict(x_test)
    
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
    radius_converge = np.min(temp[:,0])
    temp2= temp[np.where(:,2)==40]
    radius_converge2=  temp2[0,1]
    temp3= temp[np.where(:,2)==50]
    radius_converge3=  temp3[0,1]


    
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
    plt.title("radius(infinite)="+str(radius_converge))
    plot_path = 'radius.eps'
    plt.ylim((1.38,1.46))  
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
    os.system('cp '+'radius.eps'+' '+output_path) 
#    os.system('cp '+'loss_val_loss.eps'+' '+output_path) 
#
# plot different_hw.eps and lowest_each_Nmax.eps
#
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
#    radius_converge = np.min(temp_1)
#    
#    raw_data_new = np.zeros((200,2),dtype = np.float)
#    for loop in range(0,200):
#        raw_data_new[loop,1] = loop+4
#        raw_data_new[loop,0] = np.min(raw_data[np.where(raw_data[:,1]==loop+4)])
#    
#    x_list = raw_data_new[:,1]
#    y_list = np.log10(raw_data_new[:,0] - radius_converge)
#    
#    x_list_1 = raw_data_new_1 [:,1]
#    y_list_1 = np.log10(raw_data_new_1[:,0] - radius_converge)
#    
#    x_list_2 = raw_data_new_2 [:,1]
#    y_list_2 = np.log10(raw_data_new_2[:,0] - radius_converge)
#    
#    x_list_3 = raw_data_new_3 [:,1]
#    y_list_3 = np.log10(raw_data_new_3[:,0] - radius_converge)
#    
#    x_list_4 = raw_data_new_4 [:,1]
#    y_list_4 = np.log10(raw_data_new_4[:,0] - radius_converge)
#    fig_1 = plt.figure('fig_1')
#    l1 = plt.scatter(x_list_1,y_list_1,color='k',linestyle='--',s = 10, marker = 'x', label    ='NN_prediction_hw=20')
#    l2 = plt.scatter(x_list_2,y_list_2,color='r',linestyle='--',s = 10, marker = 'x', label    ='NN_prediction_hw=30')
#    l3 = plt.scatter(x_list_3,y_list_3,color='g',linestyle='--',s = 10, marker = 'x', label    ='NN_prediction_hw=40')
#    l4 = plt.scatter(x_list_4,y_list_4,color='b',linestyle='--',s = 10, marker = 'x', label    ='NN_prediction_hw=50')
#    
#    plt.title("E(converge)="+str(radius_converge))
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
#    plt.title("E(converge)="+str(radius_converge))
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
    return radius_converge,radius_converge2,radius_converge3,loss,val_loss



#
# all NN parameters
#
nuclei = 'He4'
target_option = 'radius'
input_path = 'He4R_NNLOopt.txt'
#output_path = './result/gs/'
data_num = input_raw_data_count(input_path)
# earlystopping parameters
monitor  = 'loss'
min_delta = 0.00000001
patience = 30
epochs = 10000
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
sample_weight_switch = 'on'
FWHM_percent = 0.6

 

radius_converge_all  = np.zeros(run_times_end)
radius_converge_all2 = np.zeros(run_times_end)
radius_converge_all3 = np.zeros(run_times_end)
loss_all = np.zeros(run_times_end)
val_loss_all = np.zeros(run_times_end)

os.system('mkdir '+nuclei)
os.system('mkdir '+nuclei+'/'+target_option)        


for max_nmax_fit in range(20,21,2):
    os.system('mkdir '+nuclei+'/'+target_option+'/radius-nmax4-'+str(max_nmax_fit))
    with open(nuclei+'/'+target_option+'/radius-nmax4-'+str(max_nmax_fit)+'/'+'radius_NN_info.txt','a') as f_3:
        #f_3.read()
        f_3.write('#################################################################################'+'\n')
        f_3.write('# loop   radius    radius2(hw=40)   radius3(hw=50)      loss       val_loss'+'\n')
    for loop_all in range(run_times_start-1,run_times_end):
        os.system('mkdir '+nuclei+'/'+target_option+'/radius-nmax4-'+str(max_nmax_fit)+'/'+str(loop_all+1))
        output_path = nuclei+'/'+target_option+'/radius-nmax4-'+str(max_nmax_fit)+'/'+str(loop_all+1)
        radius_converge_all[loop_all],radius_converge_all2[loop_all],radius_converge_all3[loop_all],loss_all[loop_all],val_loss_all[loop_all] = NN_all(input_path=input_path,output_path=output_path,data_num=data_num,monitor=monitor,min_delta=min_delta,patience=patience,epochs=epochs,input_dim=input_dim,output_dim=output_dim,interpol_count=interpol_count,max_nmax_fit=max_nmax_fit,FWHM_percent=FWHM_percent)
        with open(nuclei+'/'+target_option+'/radius-nmax4-'+str(max_nmax_fit)+'/'+'radius_NN_info.txt','a') as f_3:
            #f_3.read()
            f_3.write('{:>5}'.format(loop_all+1)+'   ')
            f_3.write('{:>-10.5f}'.format(radius_converge_all[loop_all])+'   ')
            f_3.write('{:>-20.15f}'.format(loss_all[loop_all])+'   ')
            f_3.write('{:>-20.15f}'.format(loss_all2[loop_all])+'   ')
            f_3.write('{:>-20.15f}'.format(loss_all3[loop_all])+'   ')
            f_3.write('{:>-20.15f}'.format(val_loss_all[loop_all])+'\n')


    


#input()
