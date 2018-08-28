import keras
import numpy as np
import re
import os
import matplotlib.pyplot as plt

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

def tansig(x):
    """
    f(x) = max(relu(x),sigmoid(x)-0.5)
    """
    return (2*K.sigmoid(2*x)-1)


def input_file(file_path,raw_data):
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
                raw_data[loop2][1] = int(temp_1[1])
                raw_data[loop2][2] = float(temp_1[2])
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


#
# all NN parameters
#
file_path = 'He8R_NNLOopt.txt'
data_num = input_raw_data_count(file_path)
print 'data_num='+str(data_num)
# earlystopping parameters
monitor  = 'val_loss'
min_delta = 0.000001
patience = 100
epochs = 10000
input_dim = 2 
output_dim = 1
# interpolation setting
interpol_count = 10000



 
raw_data = np.zeros((data_num,3),dtype = np.float)

input_file(file_path,raw_data)

#
# To get more data, we do interpolation for the data
#
# interpolation for second colum
# kind can be 'slinear', 'quadratic' and 'cubic' refer to a spline interpolation     of first, second or third order)  
kind = "quadratic"
# for data of 4
raw_data_new = raw_data[np.where(raw_data[:,1]==4)]
x_1 = raw_data_new[:,2]
y_1 = raw_data_new[:,0]
x_new_1 = np.linspace(np.min(x_1),np.max(x_1),interpol_count)
f=interpolate.interp1d(x_1,y_1,kind=kind)
y_new_1 = f(x_new_1)
x_new_1_count = len(x_new_1)

#fig3 = plt.figure('fig3')
#l1=plt.scatter(x_1,y_1,color='k',linestyle='--',s = 10, marker = 'x', label='CC_calculation')
#
#l2=plt.scatter(x_new_1,y_new_1,color='r',linestyle='--',s = 10, marker = 'x', label='CC_calculation')
#
##plt.ylim((-0.05, 0.01))
#fig3.show()


# for data of 6
raw_data_new = raw_data[np.where(raw_data[:,1]==6)]
x_2 = raw_data_new[:,2]
y_2 = raw_data_new[:,0]
x_new_2 = np.linspace(np.min(x_2),np.max(x_2),interpol_count)
f=interpolate.interp1d(x_2,y_2,kind=kind)
y_new_2 = f(x_new_2)
x_new_2_count = len(x_new_2)

#fig4 = plt.figure('fig4')
#l1=plt.scatter(x_2,y_2,color='k',linestyle='--',s = 10, marker = 'x', label='CC_calculation')
#l2=plt.scatter(x_new_2,y_new_2,color='r',linestyle='--',s = 10, marker = 'x', label='CC_calculation')
##plt.ylim((-0.05, 0.01))
#fig4.show()


# for data of 8
raw_data_new = raw_data[np.where(raw_data[:,1]==8)]
x_3 = raw_data_new[:,2]
y_3 = raw_data_new[:,0]
x_new_3 = np.linspace(np.min(x_3),np.max(x_3),interpol_count)
f=interpolate.interp1d(x_3,y_3,kind=kind)
y_new_3 = f(x_new_3)
x_new_3_count = len(x_new_3)

#fig4 = plt.figure('fig4')
#l1=plt.scatter(x_3,y_3,color='k',linestyle='--',s = 10, marker = 'x', label='CC_calculation')
#l2=plt.scatter(x_new_3,y_new_3,color='r',linestyle='--',s = 10, marker = 'x', label='CC_calculation')
##plt.ylim((-0.05, 0.01))
#fig4.show()

# for data of 10
raw_data_new = raw_data[np.where(raw_data[:,1]==10)]
x_4 = raw_data_new[:,2]
y_4 = raw_data_new[:,0]
x_new_4 = np.linspace(np.min(x_4),np.max(x_4),interpol_count)
f=interpolate.interp1d(x_4,y_4,kind=kind)
y_new_4 = f(x_new_4)
x_new_4_count = len(x_new_4)

#fig4 = plt.figure('fig4')
#l1=plt.scatter(x_4,y_4,color='k',linestyle='--',s = 10, marker = 'x', label='CC_calculation')
#l2=plt.scatter(x_new_4,y_new_4,color='r',linestyle='--',s = 10, marker = 'x', label='CC_calculation')
#fig4.show()

# for data of 12
raw_data_new = raw_data[np.where(raw_data[:,1]==12)]
x_5 = raw_data_new[:,2]
y_5 = raw_data_new[:,0]
x_new_5 = np.linspace(np.min(x_5),np.max(x_5),interpol_count)
f=interpolate.interp1d(x_5,y_5,kind=kind)
y_new_5 = f(x_new_5)
x_new_5_count = len(x_new_5)

#fig4 = plt.figure('fig4')
#l1=plt.scatter(x_5,y_5,color='k',linestyle='--',s = 10, marker = 'x', label='CC_calculation')
#l2=plt.scatter(x_new_5,y_new_5,color='r',linestyle='--',s = 10, marker = 'x', label='CC_calculation')
#fig4.show()



# for data of 14
#raw_data_new = raw_data[np.where(raw_data[:,1]==14)]
#x_6 = raw_data_new[:,2]
#y_6 = raw_data_new[:,0]
#x_new_6 = np.linspace(np.min(x_6),np.max(x_6),interpol_count)
#f=interpolate.interp1d(x_6,y_6,kind=kind)
#y_new_6 = f(x_new_6)
#x_new_6_count = len(x_new_6)
#
##fig4 = plt.figure('fig4')
##l1=plt.scatter(x_6,y_6,color='k',linestyle='--',s = 10, marker = 'x', label='CC_calculation')
##l2=plt.scatter(x_new_6,y_new_6,color='r',linestyle='--',s = 10, marker = 'x', label='CC_calculation')
###plt.ylim((-0.05, 0.01))
##fig4.show()
#
#
## for data of 16
#raw_data_new = raw_data[np.where(raw_data[:,1]==16)]
#x_7 = raw_data_new[:,2]
#y_7 = raw_data_new[:,0]
#x_new_7 = np.linspace(np.min(x_7),np.max(x_7),interpol_count)
#f=interpolate.interp1d(x_7,y_7,kind=kind)
#y_new_7 = f(x_new_7)
#x_new_7_count = len(x_new_7)
#
##fig4 = plt.figure('fig4')
##l1=plt.scatter(x_7,y_7,color='k',linestyle='--',s = 10, marker = 'x', label='CC_calculation')
##l2=plt.scatter(x_new_7,y_new_7,color='r',linestyle='--',s = 10, marker = 'x', label='CC_calculation')
##fig4.show()
#
## for data of 18
#raw_data_new = raw_data[np.where(raw_data[:,1]==18)]
#x_8 = raw_data_new[:,2]
#y_8 = raw_data_new[:,0]
#x_new_8 = np.linspace(np.min(x_8),np.max(x_8),interpol_count)
#f=interpolate.interp1d(x_8,y_8,kind=kind)
#y_new_8 = f(x_new_8)
#x_new_8_count = len(x_new_8)

#fig4 = plt.figure('fig4')
#l1=plt.scatter(x_8,y_8,color='k',linestyle='--',s = 10, marker = 'x', label='CC_calculation')
#l2=plt.scatter(x_new_8,y_new_8,color='r',linestyle='--',s = 10, marker = 'x', label='CC_calculation')
#fig4.show()
#
#
## for data of 20
#raw_data_new = raw_data[np.where(raw_data[:,1]==20)]
#x_9 = raw_data_new[:,2]
#y_9 = raw_data_new[:,0]
#x_new_9 = np.linspace(9.77,39.07,interpol_count)
#f=interpolate.interp1d(x_9,y_9,kind=kind)
#y_new_9 = f(x_new_9)
#x_new_9_count = len(x_new_9)
#
##fig4 = plt.figure('fig4')
##l1=plt.scatter(x_9,y_9,color='k',linestyle='--',s = 10, marker = 'x', label='CC_calculation')
##l2=plt.scatter(x_new_9,y_new_9,color='r',linestyle='--',s = 10, marker = 'x', label='CC_calculation')
###plt.ylim((-0.05, 0.01))
##fig4.show()
#
## for data of 22
#raw_data_new = raw_data[np.where(raw_data[:,1]==22)]
#x_10 = raw_data_new[:,2]
#y_10 = raw_data_new[:,0]
#x_new_10 = np.linspace(9.15,36.62,interpol_count)
#f=interpolate.interp1d(x_10,y_10,kind=kind)
#y_new_10 = f(x_new_10)
#x_new_10_count = len(x_new_10)
#
##fig5 = plt.figure('fig5')
##l1=plt.scatter(x_10,y_10,color='k',linestyle='--',s = 10, marker = 'x', label='CC_calculation')
##l2=plt.scatter(x_new_10,y_new_10,color='r',linestyle='--',s = 10, marker = 'x', label='CC_calculation')
###plt.ylim((-0.05, 0.01))
##fig5.show()



interpol_count_tot = x_new_1_count+x_new_2_count+x_new_3_count+x_new_4_count+x_new_5_count
data_interpolation= np.zeros((interpol_count_tot,3))
count = 0
for loop in range(0,x_new_1_count):
    data_interpolation[count,1] = 4
    data_interpolation[count,2] = x_new_1[loop]
    data_interpolation[count,0] = y_new_1[loop]
    count = count +1
for loop in range(0,x_new_1_count):
    data_interpolation[count,1] = 6
    data_interpolation[count,2] = x_new_2[loop]
    data_interpolation[count,0] = y_new_2[loop]
    count = count +1
for loop in range(0,x_new_1_count):
    data_interpolation[count,1] = 8
    data_interpolation[count,2] = x_new_3[loop]
    data_interpolation[count,0] = y_new_3[loop]
    count = count +1
for loop in range(0,x_new_1_count):
    data_interpolation[count,1] = 10
    data_interpolation[count,2] = x_new_4[loop]
    data_interpolation[count,0] = y_new_4[loop]
    count = count +1
for loop in range(0,x_new_1_count):
    data_interpolation[count,1] = 12
    data_interpolation[count,2] = x_new_5[loop]
    data_interpolation[count,0] = y_new_5[loop]
    count = count +1
#for loop in range(0,x_new_1_count):
#    data_interpolation[count,1] = 14
#    data_interpolation[count,2] = x_new_6[loop]
#    data_interpolation[count,0] = y_new_6[loop]
#    count = count +1
#for loop in range(0,x_new_1_count):
#    data_interpolation[count,1] = 16
#    data_interpolation[count,2] = x_new_7[loop]
#    data_interpolation[count,0] = y_new_7[loop]
#    count = count +1
#for loop in range(0,x_new_1_count):
#    data_interpolation[count,1] = 18
#    data_interpolation[count,2] = x_new_8[loop]
#    data_interpolation[count,0] = y_new_8[loop]
#    count = count +1
#for loop in range(0,x_new_1_count):
#    data_interpolation[count,1] = 20
#    data_interpolation[count,2] = x_new_9[loop]
#    data_interpolation[count,0] = y_new_9[loop]
#    count = count +1
#for loop in range(0,x_new_1_count):
#    data_interpolation[count,1] = 22
#    data_interpolation[count,2] = x_new_10[loop]
#    data_interpolation[count,0] = y_new_10[loop]
#    count = count +1

print "data_interpolation="+str(data_interpolation)

#
# shuffle the data
#

np.random.shuffle(data_interpolation)
np.random.shuffle(raw_data)

print len(raw_data)
print raw_data

#batch_size = data_num

input_shape = (input_dim,)
input_data = Input(shape = input_shape)

#raw_data_new  = raw_data[np.where(raw_data[:,1]<11)]
raw_data_new = data_interpolation[np.where(data_interpolation[:,1]<25)]
print "raw_data_new="+str(raw_data_new)

x_train = raw_data_new[:,1:3]
y_train = raw_data_new[:,0]

print "x_train = "+str(x_train)

print "y_train = "+str(y_train)


#Li6_model = Sequential([
#Dense(8, input_shape=input_shape),
#Activation('sigmoid'),
#Dense(2)
#])


x = Dense(8, activation = 'sigmoid')(input_data)
predictions = Dense(output_dim)(x)

model = Model(inputs= input_data, outputs = predictions)

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])

early_stopping = EarlyStopping(monitor=monitor,min_delta = min_delta , patience=patience, verbose=0, mode='min')

model.fit(x_train,y_train, epochs = epochs, validation_split = 0.15 , shuffle = 1, callbacks=[early_stopping])

model.save('He8_radius.h5')

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

print x_test

y_test = model.predict(x_test)

raw_predic_data = np.concatenate((y_test,x_test),axis=1)

print "raw_predic_data="+str(raw_predic_data)

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
radius_converge = np.max(temp[:,0])



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
#plt.legend(loc = 'lower left') 
plt.title("r(infinite)="+str(radius_converge))
plt.ylim((0.5,2.5))
#plt.xlim((-50,100))
plt.savefig('He8_radius.jpg')
fig1.show()



#ax0.scatter(x_list_1, y_list_1, c='r', s=20, alpha=0.5)
#ax0.set_title('calculation')
#ax1.scatter(x_list_2, y_list_2, c='b', s=20, alpha=0.5)
#ax1.set_title('NN_prediction')

#fig.subplots_adjust(hspace =0.4)
#plt.show()

#x_list = range()
#y_list = 2/np.exp(-2*x_list)
#
#plt.figure('Scatter fig')
#ax = plt.gca()
#
#ax.set_xlabel('x')
#ax.set_ylabel('y')
#
#ax.scatter(x_list, y_list, c='r', s=20, alpha=0.5)
#
#plt.show()
file_path = "He8_radius_NN_prediction.txt"
with open(file_path,'w') as f_1:
    for loop1 in range(1,count):
        f_1.write('{:>-10.5f}'.format(y_test[loop1,0]))
        f_1.write('{:>10}'.format(x_test[loop1,0]))
        f_1.write('{:>10}'.format(x_test[loop1,1])+'\n')

import plot_radius as plot

input()
