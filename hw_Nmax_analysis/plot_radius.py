import numpy as np
import matplotlib.pyplot as plt
import re
from math import log
from math import e

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
                raw_data[loop2][1] = float(temp_1[1])
                raw_data[loop2][2] = float(temp_1[2])
                loop2 = loop2 + 1
            loop1 = loop1 + 1
        print loop2


file_path = "He8_radius_NN_prediction.txt"
data_num = len(open(file_path,'rU').readlines())
raw_data = np.zeros((data_num,3),dtype = np.float)
input_file(file_path,raw_data)

raw_data_new_4 = raw_data[np.where(raw_data[:,2] == 50 )]
raw_data_new_3 = raw_data[np.where(raw_data[:,2] == 40 )]
raw_data_new_2 = raw_data[np.where(raw_data[:,2] == 30 )]
raw_data_new_1 = raw_data[np.where(raw_data[:,2] == 20 )]

temp = (raw_data[np.where(raw_data[:,1]== 200)])
radius_converge = np.max(temp[:,0])



x_list_1 = raw_data_new_1 [:,1]
y_list_1 = np.log10(radius_converge - raw_data_new_1 [:,0])

x_list_2 = raw_data_new_2 [:,1]
y_list_2 = np.log10(radius_converge - raw_data_new_2 [:,0])

x_list_3 = raw_data_new_3 [:,1]
y_list_3 = np.log10(radius_converge - raw_data_new_3 [:,0])

x_list_4 = raw_data_new_4 [:,1]
y_list_4 = np.log10(radius_converge - raw_data_new_4 [:,0])




fig_1 = plt.figure('fig_1')
l1 = plt.scatter(x_list_1,y_list_1,color='k',linestyle='--',s = 10, marker = 'x', label='NN_prediction_hw=20')
l2 = plt.scatter(x_list_2,y_list_2,color='r',linestyle='--',s = 10, marker = 'x', label='NN_prediction_hw=30')
l3 = plt.scatter(x_list_3,y_list_3,color='g',linestyle='--',s = 10, marker = 'x', label='NN_prediction_hw=40')
l4 = plt.scatter(x_list_4,y_list_4,color='b',linestyle='--',s = 10, marker = 'x', label='NN_prediction_hw=50')



plt.title("r(infinite)="+str(radius_converge))
plt.ylabel("log_10")
plt.legend(loc = 'lower left')
#plt.ylim((1.2,2.8))
#plt.savefig('Li6_radius_NN_prediction.jpg')
fig_1.show()


x_list_1 = raw_data_new_1 [:,1]
y_list_1 = raw_data_new_1 [:,0]

x_list_2 = raw_data_new_2 [:,1]
y_list_2 = raw_data_new_2 [:,0]

x_list_3 = raw_data_new_3 [:,1]
y_list_3 = raw_data_new_3 [:,0]

x_list_4 = raw_data_new_4 [:,1]
y_list_4 = raw_data_new_4 [:,0]





fig_2 = plt.figure('fig_2')
l1 = plt.scatter(x_list_1,y_list_1,color='k',linestyle='--',s = 10, marker = 'x', label='NN_prediction_hw=20')
l2 = plt.scatter(x_list_2,y_list_2,color='r',linestyle='--',s = 10, marker = 'x', label='NN_prediction_hw=30')
l3 = plt.scatter(x_list_3,y_list_3,color='g',linestyle='--',s = 10, marker = 'x', label='NN_prediction_hw=40')
l4 = plt.scatter(x_list_4,y_list_4,color='b',linestyle='--',s = 10, marker = 'x', label='NN_prediction_hw=50')


plt.title("r(infinite)="+str(radius_converge))
plt.ylabel("r(infinite)")
plt.legend(loc = 'lower left')
#plt.ylim((1.2,2.8))
#plt.savefig('Li6_radius_NN_prediction.jpg')
fig_2.show()

input()
