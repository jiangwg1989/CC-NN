import numpy as np
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import re
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
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

pwd  = os.getcwd()
temp = re.findall(r"[-+]?\d+\.?\d*",pwd)
max_nmax_fit = -1*int(temp[2])

#######################################
#######################################
## setting 
#######################################
#######################################
x_min = 2.35
x_max = 2.7
y_min = 0
y_max = 0.0001



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
plt.ylim((y_min,y_max))
plt.xlim((x_min,x_max))
#plt.savefig('Li6_radius_NN_prediction.jpg')
plot_path = 'Multi-NN.pdf'
plt.savefig(plot_path)
#fig_1.show()



raw_data_new_2 = raw_data_new[np.where((raw_data_new[:,0]>x_min)&(raw_data_new[:,0]<x_max))]
x_list_1 = raw_data_new_2[:,0]


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
plt.xlim((x_min,x_max))
plt.ylim((0,50))
##plt.savefig('Li6_radius_NN_prediction.jpg')
plot_path = 'multi-NN_distribution.pdf'
plt.savefig(plot_path)
plt.close('all')


#################
#################
#  Curve fit
#################
#################
x = raw_data_new_2[:,0]
y = raw_data_new_2[:,1]

num_bins = 35  
n, bins_left_x, patches = plt.hist(x, num_bins,normed=1, facecolor='blue', alpha=0.5)
#print(bins_left_x)
#print(n)


def fun_gauss(x,a,x0,sigma):
    return a*exp(-(x-x0)**2./(2*sigma**2))

x = (bins_left_x[1:len(bins_left_x)]+bins_left_x[0:len(bins_left_x)-1])/2
y = n 
# set trail value
n = len(x)
mean = sum(x)/n
sigma = sum(y*(x-mean)**2)/2

popt,pcov = curve_fit(fun_gauss,x,y,p0=[1,mean,sigma])
print('popt='+str(popt))
print('mean='+str(popt[1]))
print('FWHM='+str(np.abs(popt[2])*2.35482004503))

x_new =  np.linspace(x_min,x_max,1000)
plt.plot(x_new,fun_gauss(x_new,popt[0],popt[1],popt[2]),label='fit')
plt.title("mean = "+str(popt[1])+"  FWHM = "+str(np.abs(popt[2])*2.35482004503))
plot_path= 'test.pdf'
plt.savefig(plot_path)

