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
import matplotlib.gridspec as gridspec

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

raw_data = np.zeros((3,3))
file_path_1= "O16_gs_different_Nmax.txt"
input_file_2(file_path_1,raw_data)
fig_1 = plt.figure('fig_1',figsize=[3,3])
plt.subplots_adjust(wspace =0, hspace =0.5)

gs    = gridspec.GridSpec(3,3)
ax1   = fig_1.add_subplot(gs[0:2,:])
#plt.subplots_adjust(wspace =0.3, hspace =0.2)

matplotlib.rcParams['xtick.direction'] = 'in' 
matplotlib.rcParams['ytick.direction'] = 'in' 

#ax1 = fig_1.add_subplot(2,1,1)
plt.tick_params(top=True,bottom=True,left=True,right=False)

x     = raw_data[:,0]  
mean  = raw_data[:,1]
error = raw_data[:,2]/2.355
plt.errorbar(x,mean,error,fmt='.k',ecolor='b' )

##########################################################
### setting parameters
##########################################################
x_fontsize = 8
y_fontsize = 8
x_lim_min  = 7
x_lim_max  = 13
y_lim_min  = -132
y_lim_max  = -129
x_tick_min = x_lim_min
x_tick_max = y_lim_max
x_tick_gap = 2
y_tick_min = y_lim_min
y_tick_max = -120
y_tick_gap = 1
y_label_f  = 12

#plt.xlabel()
plt.ylabel(r'$E_{gs} \ \rm{(MeV)}$',fontsize=y_label_f)
plt.xticks(np.arange(8,13,2),fontsize = x_fontsize)
plt.yticks(np.arange(y_tick_min,y_tick_max,y_tick_gap),fontsize = y_fontsize)
plt.xlim((x_lim_min,x_lim_max))
plt.ylim((y_lim_min,y_lim_max))





##########################################################
##########################################################
# plot gs
##########################################################
##########################################################

raw_data = np.zeros((3,3))
file_path_1= "O16_gs_different_Nmax.txt"
input_file_2(file_path_1,raw_data)

ax2 = fig_1.add_subplot(gs[2,:])
#plt.subplots_adjust(wspace =0.3, hspace =0.2)


#ax1 = fig_1.add_subplot(2,1,1)
plt.tick_params(top=True,bottom=True,left=True,right=False)

x     = raw_data[:,0]  
mean  = raw_data[:,1]
error = raw_data[:,2]/2.355

plt.plot(x,mean, color='g', linestyle = '',linewidth=0.5,marker='s', markerfacecolor='none',mew=1,markersize=5,zorder=5,label='NN extrapolation')
plt.errorbar(x,mean,error,linestyle="None",ecolor='g',capsize=capsize)




##########################################################
### setting parameters
##########################################################
x_fontsize = 8
y_fontsize = 8
x_lim_min  = 7
x_lim_max  = 13
y_lim_min  = -132
y_lim_max  = -129
x_tick_min = x_lim_min
x_tick_max = y_lim_max
x_tick_gap = 2
y_tick_min = y_lim_min
y_tick_max = -120
y_tick_gap = 1
y_label_f  = 10

plt.xlabel('$N_{max}$',fontsize=y_label_f)
#plt.ylabel(r'$E_{gs} \ \rm{(MeV)}$',fontsize=y_label_f)
plt.xticks(np.arange(8,13,2),fontsize = x_fontsize)
plt.yticks(np.arange(y_tick_min,y_tick_max,y_tick_gap),fontsize = y_fontsize)
plt.xlim((x_lim_min,x_lim_max))
plt.ylim((y_lim_min,y_lim_max))




plot_path = 'different_Nmax_observables_O16.pdf'
plt.savefig(plot_path,bbox_inches='tight')








