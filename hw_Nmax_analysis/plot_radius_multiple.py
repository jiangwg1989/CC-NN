import numpy as np
import matplotlib.pyplot as plt
import re
from math import log
from math import e
import seaborn as sns 
import matplotlib as mpl
import os

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
                raw_data[loop2][0] = float(temp_1[1])
                raw_data[loop2][1] = float(temp_1[2])
                raw_data[loop2][2] = float(temp_1[3])
                loop2 = loop2 + 1
            loop1 = loop1 + 1
        print loop2

pwd  = os.getcwd()
print 'pwd='+pwd
temp = re.findall(r"[-+]?\d+\.?\d*",pwd)
max_nmax_fit = -1*int(temp[2])


file_path = "radius_NN_info.txt"
data_num = len(open(file_path,'rU').readlines())
raw_data = np.zeros((data_num,3),dtype = np.float)
input_file_2(file_path,raw_data)
raw_data_new = raw_data[0:100,:]

x_list = raw_data_new[:,0]
y_list = raw_data_new[:,1]

fig_1 = plt.figure('fig_1')
l = plt.scatter(x_list,y_list,color='b',s = 20, marker = 'o')

plt.title("He4_Multi-NN_Nmax4~"+str(max_nmax_fit))
plt.xlabel("radius_energy")
plt.ylabel("loss")
#plt.legend(loc = 'lower left')
plt.ylim((0,0.00002))
plt.xlim((1.3,1.7))
#plt.savefig('Li6_radius_NN_prediction.jpg')
plot_path = 'Multi-NN.eps'
plt.savefig(plot_path)
fig_1.show()



raw_data_new_2 = raw_data_new[np.where((raw_data_new[:,0]>1.35)&(raw_data_new[:,0]<1.8))]
x_list_1 = raw_data_new_2[:,0]

print x_list_1

fig_2 = plt.figure('fig_2')

sns.set_palette("hls") 
mpl.rc("figure", figsize=(6,4)) 
sns.distplot(x_list_1,bins=20,kde_kws={"color":"seagreen", "lw":3 }, hist_kws={ "color": "lightblue"}) 

#plt.hist(x_list_1,200,normed=2,histtype='bar',facecolor='yellowgreen',alpha=0.75)
#l = plt.scatter(x_list,y_list,color='k',linestyle='--',s = 10, marker = 'x', label='E(infinite)')
#
#
#plt.title("E(converge)="+str(gs_converge))
plt.ylabel("count")
plt.xlabel("radius (fm)")
#plt.legend(loc = 'lower left')
plt.xlim((1.3,1.7))
plt.ylim((0,16))
##plt.savefig('Li6_radius_NN_prediction.jpg')
plot_path = 'multi-NN_distribution.eps'
plt.savefig(plot_path)
fig_2.show()







input()
