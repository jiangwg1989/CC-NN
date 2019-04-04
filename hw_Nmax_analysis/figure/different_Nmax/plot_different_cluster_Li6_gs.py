#
#  with corelate loss, each interpolation point is correlate with the near by points
#



import keras
import numpy as np
import re
import os
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.losses
import scipy as sp


from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Input, Dense, Dropout, Activation, Multiply
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

from scipy.interpolate import interp1d


matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
 

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

##########################################################
##########################################################
# plot gs
##########################################################
##########################################################
capsize= 2


raw_data = np.zeros((6,3))
raw_data2= np.zeros((6,3))
file_path_1= "Li6_gs_different_Nmax.txt"
input_file_2(file_path_1,raw_data)
file_path_2= "Li6_gs_different_Nmax_IR.txt"
input_file_2(file_path_2,raw_data2)

fig_1 = plt.figure('fig_1',figsize=(4,4))
#plt.subplots_adjust(wspace =0.3, hspace =0.2)

matplotlib.rcParams['xtick.direction'] = 'in' 
matplotlib.rcParams['ytick.direction'] = 'in' 

ax1 = fig_1.add_subplot(2,1,1)
plt.tick_params(top='on',bottom='on',left='on',right='off')

x     = raw_data[:,0]  
mean  = raw_data[:,1]
error = raw_data[:,2]/2.355
plt.plot(x,mean, color='g', linestyle = '',linewidth=0.5,marker='s', markerfacecolor='none',mew=1,markersize=5,zorder=5,label='NN extrapolation')
plt.errorbar(x,mean,error,linestyle="None",ecolor='g',zorder=5,capsize=capsize)#,fmt='.k',ecolor='b' )

#f2 = interp1d(x,mean,kind='cubic',fill_value="extrapolate")
#xnew = np.linspace(9.5,20.5,num=41,endpoint = True)
#plt.plot(xnew,f2(xnew),linestyle='--',color='crimson',linewidth=1)


x     = raw_data2[:,0]  
mean  = raw_data2[:,1]
error = raw_data2[:,2]
plt.plot(x,mean, color='crimson', linestyle = '',linewidth=0.5,marker='o', markerfacecolor='none',mew=1,markersize=5,zorder=1,label='IR extrapolation')
plt.errorbar(x,mean,error,linestyle="None",ecolor='crimson',zorder=1,capsize=capsize)

#f2 = interp1d(x,mean,kind='quadratic',fill_value="extrapolate")
#xnew = np.linspace(9.5,20.5,num=41,endpoint = True)
#plt.plot(xnew,f2(xnew),linestyle='--',color='g',linewidth=1)

#xnew = np.linspace(9.5,20.5,num=41,endpoint = True)
#plt.plot(xnew,f2(xnew),linestyle='--',color='g',linewidth=1)

##########################################################
### setting parameters
##########################################################
#y_fontsize = 8
#x_lim_min  = 9
#x_lim_max  = 21
#y_lim_min  = -27.75
#y_lim_max  = -27.50
#x_tick_min = x_lim_max
#x_tick_max = y_lim_max
#x_tick_gap = 2
#y_tick_min = y_lim_min
#y_tick_max = -27.499
#y_tick_gap = 0.05
#y_label_f  = 12

y_fontsize = 8 
x_lim_min  = 11
x_lim_max  = 23
y_lim_min  = -31.0
y_lim_max  = -29.6
x_tick_min = 11
x_tick_max = 23
x_tick_gap = 2 
y_tick_min = y_lim_min
y_tick_max = -29.599
y_tick_gap = 0.2 
y_label_f  = 12




#plt.xlabel()
plt.ylabel(r'$E_{gs} \ \rm{(MeV)}$',fontsize=y_label_f)
#plt.xticks([])
plt.yticks(np.arange(y_tick_min,y_tick_max,y_tick_gap),fontsize = y_fontsize)
plt.xticks(fontsize=False,color='white')
plt.xlim((x_lim_min,x_lim_max))
plt.ylim((y_lim_min,y_lim_max))

plt.legend(loc=1,frameon=False,fontsize=9,labelspacing=0.3,handletextpad=0.1)


##########################################################
##########################################################
# plot radius
##########################################################
##########################################################


raw_data = np.zeros((6,3))
raw_data2 = np.zeros((6,3))
file_path_1= "Li6_radius_different_Nmax.txt"
input_file_2(file_path_1,raw_data)
file_path_2= "Li6_radius_different_Nmax_IR.txt"
input_file_2(file_path_2,raw_data2)



matplotlib.rcParams['xtick.direction'] = 'in' 
matplotlib.rcParams['ytick.direction'] = 'in' 

ax2 = fig_1.add_subplot(2,1,2)
plt.tick_params(top='on',bottom='on',left='on',right='off')

x     = raw_data[:,0]  
mean  = raw_data[:,1]
error = raw_data[:,2]/2.355
plt.plot(x,mean, color='g', linestyle = '',linewidth=0.5,marker='s', markerfacecolor='none',mew=1,markersize=5,zorder=5,label='NN extrapolation')
plt.errorbar(x,mean,error,linestyle="None",ecolor='g',zorder=5,capsize=capsize)

#f2 = interp1d(x,mean,kind='cubic',fill_value="extrapolate")
#xnew = np.linspace(9.5,20.5,num=41,endpoint = True)
#plt.plot(xnew,f2(xnew),linestyle='--',color='crimson',linewidth=1)

x     = raw_data2[:,0]  
mean  = raw_data2[:,1]
error = raw_data2[:,2]
plt.plot(x,mean, color='crimson', linestyle = '',linewidth=0.5,marker='o', markerfacecolor='none',mew=1,markersize=5,zorder=1,label='IR extrapolation')
plt.errorbar(x,mean,error,linestyle="None",ecolor='crimson' ,zorder=1,capsize=capsize)

#f2 = interp1d(x,mean,kind='cubic',fill_value="extrapolate")
#xnew = np.linspace(9.5,20.5,num=41,endpoint = True)
#plt.plot(xnew,f2(xnew),linestyle='--',color='g',linewidth=1)


##########################################################
### setting parameters
##########################################################
#y_fontsize = 8
#x_lim_min  = 9
#x_lim_max  = 21
#y_lim_min  = 1.424
#y_lim_max  = 1.444
#x_tick_min = x_lim_max
#x_tick_max = y_lim_max
#x_tick_gap = 2
#y_tick_min = y_lim_min
#y_tick_max = y_lim_max+0.00001
#y_tick_gap = 0.004
#y_label_f  = 12

y_lim_min  = 2.18
y_lim_max  = 2.52
x_tick_gap = 2
y_tick_min = 2.2
y_tick_max = 2.6
y_tick_gap = 0.1


plt.legend(loc=4,frameon=False,fontsize=9,labelspacing=0.3,handletextpad=0.1)


plt.xlabel(r'$N_{max}$',fontsize=y_label_f)
plt.ylabel(r'$r \ \rm{(fm)}$',fontsize=y_label_f)
#plt.xticks([])
plt.yticks(np.arange(y_tick_min,y_tick_max,y_tick_gap),fontsize = y_fontsize)
plt.xlim((x_lim_min,x_lim_max))
plt.ylim((y_lim_min,y_lim_max))
fig_1.tight_layout()
plot_path = 'different_Nmax_observables_Li6.pdf'
plt.savefig(plot_path,bbox_inches='tight')








