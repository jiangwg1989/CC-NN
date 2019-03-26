import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import re
from math import log
from math import e
import seaborn as sns 
import matplotlib as mpl


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

##########################################################
##########################################################
### setting parameters
##########################################################
##########################################################
y_fontsize = 8
x_lim_min  = -28.04
x_lim_max  = -27.3
y_lim_min  = 0
y_lim_max  = 15
x_tick_min = -28.00
x_tick_max = -27.35
x_tick_gap = 0.2
y_tick_min = 0
y_tick_max = 15
y_tick_gap = 5



##########################################################
##########################################################
### start the plot
##########################################################
##########################################################

fig_1 = plt.figure('fig_1')
plt.subplots_adjust(wspace =0.3, hspace =0)


####################################################
## corr_loss Nmax_10
####################################################
file_path = "gs_NN_info_corr_He4_10.txt"
data_num = len(open(file_path,'rU').readlines())
raw_data = np.zeros((data_num,3),dtype = np.float)
input_file_2(file_path,raw_data)
raw_data_new = raw_data[0:100,:]

x_list = raw_data_new[:,0]
y_list = raw_data_new[:,1]
raw_data_new_2 = raw_data_new[np.where((raw_data_new[:,0]>-28)&(raw_data_new[:,0]<-27.4))]
x_list_1 = raw_data_new_2[:,0]


#fig_1 = plt.figure('fig_1')
#l = plt.scatter(x_list,y_list,color='b',s = 20, marker = 'o')
#
#plt.title("He4_Multi-NN_Nmax4~18")
#plt.xlabel("gs_energy (MeV)")
#plt.ylabel("loss")
##plt.legend(loc = 'lower left')
#plt.ylim((0,0.05))
#plt.xlim((-28,-27.4))
##plt.savefig('Li6_radius_NN_prediction.jpg')
#plot_path = 'Multi-NN.eps'
#plt.savefig(plot_path)
#fig_1.show()



matplotlib.rcParams['xtick.direction'] = 'in' 
matplotlib.rcParams['ytick.direction'] = 'in' 
ax1 = fig_1.add_subplot(6,2,1)

plt.tick_params(top=True,bottom=True,left=True,right=False)

sns.set_palette("hls") 
mpl.rc("figure")#, figsize=(6,4)) 
l1=sns.distplot(x_list_1,bins=15,kde_kws={"color":"seagreen", "lw":3 }, hist_kws={ "color": "lightskyblue"}) 

#plt.hist(x_list_1,200,normed=2,histtype='bar',facecolor='yellowgreen',alpha=0.75)
#l = plt.scatter(x_list,y_list,color='k',linestyle='--',s = 10, marker = 'x', label='E(infinite)')
#
#
#plt.title("E(converge)="+str(gs_converge))
plt.ylabel("count")
#plt.xlabel("gs_energy (MeV)")
#plt.legend(loc = 'lower left')
plt.xticks(np.arange(x_tick_min,x_tick_max,x_tick_gap),fontsize = 0)
plt.yticks(np.arange(y_tick_min,y_tick_max+0.01,y_tick_gap),fontsize = y_fontsize)
plt.xlim((x_lim_min,x_lim_max))
plt.ylim((y_lim_min,y_lim_max))



#########################################################
## corr_loss Nmax12
#########################################################
file_path = "gs_NN_info_corr_He4_12.txt"
data_num = len(open(file_path,'rU').readlines())
raw_data = np.zeros((data_num,3),dtype = np.float)
input_file_2(file_path,raw_data)
raw_data_new = raw_data[0:100,:]
raw_data_new_2 = raw_data_new[np.where((raw_data_new[:,0]>-28)&(raw_data_new[:,0]<-27.4))]
x_list_1 = raw_data_new_2[:,0]



matplotlib.rcParams['xtick.direction'] = 'in' 
matplotlib.rcParams['ytick.direction'] = 'in' 
plt.subplot(623)
plt.tick_params(top=True,bottom=True,left=True,right=False)

sns.set_palette("hls") 
mpl.rc("figure")#, figsize=(6,4)) 
l2=sns.distplot(x_list_1,bins=15,kde_kws={"color":"seagreen", "lw":3 }, hist_kws={ "color": "lightskyblue"}) 

plt.ylabel("count")
#plt.xlabel("gs_energy (MeV)")
#plt.legend(loc = 'lower left')
plt.xticks(np.arange(x_tick_min,x_tick_max,x_tick_gap),fontsize = 0)
plt.yticks(np.arange(y_tick_min,y_tick_max+0.01,y_tick_gap),fontsize = y_fontsize)


plt.xlim((x_lim_min,x_lim_max))
plt.ylim((y_lim_min,y_lim_max))
plt.yticks(np.arange(0,15,5))


#########################################################
## corr_loss Nmax14
#########################################################
file_path = "gs_NN_info_corr_He4_14.txt"
data_num = len(open(file_path,'rU').readlines())
raw_data = np.zeros((data_num,3),dtype = np.float)
input_file_2(file_path,raw_data)
raw_data_new = raw_data[0:100,:]
raw_data_new_2 = raw_data_new[np.where((raw_data_new[:,0]>-28)&(raw_data_new[:,0]<-27.4))]
x_list_1 = raw_data_new_2[:,0]



matplotlib.rcParams['xtick.direction'] = 'in' 
matplotlib.rcParams['ytick.direction'] = 'in' 
plt.subplot(625)
plt.tick_params(top=True,bottom=True,left=True,right=False)

sns.set_palette("hls") 
mpl.rc("figure")#, figsize=(6,4)) 
l2=sns.distplot(x_list_1,bins=15,kde_kws={"color":"seagreen", "lw":3 }, hist_kws={ "color": "lightskyblue"}) 

plt.ylabel("count")
#plt.xlabel("gs_energy (MeV)")
#plt.legend(loc = 'lower left')
plt.xticks(np.arange(x_tick_min,x_tick_max,x_tick_gap),fontsize = 0)
plt.yticks(np.arange(y_tick_min,y_tick_max+0.01,y_tick_gap),fontsize = y_fontsize)


plt.xlim((x_lim_min,x_lim_max))
plt.ylim((y_lim_min,y_lim_max))
plt.yticks(np.arange(0,15,5))



#########################################################
## corr_loss Nmax16
#########################################################
file_path = "gs_NN_info_corr_He4_16.txt"
data_num = len(open(file_path,'rU').readlines())
raw_data = np.zeros((data_num,3),dtype = np.float)
input_file_2(file_path,raw_data)
raw_data_new = raw_data[0:100,:]
raw_data_new_2 = raw_data_new[np.where((raw_data_new[:,0]>-28)&(raw_data_new[:,0]<-27.4))]
x_list_1 = raw_data_new_2[:,0]



matplotlib.rcParams['xtick.direction'] = 'in' 
matplotlib.rcParams['ytick.direction'] = 'in' 
plt.subplot(627)
plt.tick_params(top=True,bottom=True,left=True,right=False)

sns.set_palette("hls") 
mpl.rc("figure")#, figsize=(6,4)) 
l2=sns.distplot(x_list_1,bins=15,kde_kws={"color":"seagreen", "lw":3 }, hist_kws={ "color": "lightskyblue"}) 

plt.ylabel("count")
#plt.xlabel("gs_energy (MeV)")
#plt.legend(loc = 'lower left')
plt.xticks(np.arange(x_tick_min,x_tick_max,x_tick_gap),fontsize = 0)
plt.yticks(np.arange(y_tick_min,y_tick_max+0.01,y_tick_gap),fontsize = y_fontsize)


plt.xlim((x_lim_min,x_lim_max))
plt.ylim((y_lim_min,y_lim_max))
plt.yticks(np.arange(0,15,5))


#########################################################
## corr_loss Nmax18
#########################################################
file_path = "gs_NN_info_corr_He4_16.txt"
data_num = len(open(file_path,'rU').readlines())
raw_data = np.zeros((data_num,3),dtype = np.float)
input_file_2(file_path,raw_data)
raw_data_new = raw_data[0:100,:]
raw_data_new_2 = raw_data_new[np.where((raw_data_new[:,0]>-28)&(raw_data_new[:,0]<-27.4))]
x_list_1 = raw_data_new_2[:,0]



matplotlib.rcParams['xtick.direction'] = 'in' 
matplotlib.rcParams['ytick.direction'] = 'in' 
plt.subplot(629)
plt.tick_params(top=True,bottom=True,left=True,right=False)

sns.set_palette("hls") 
mpl.rc("figure")#, figsize=(6,4)) 
l2=sns.distplot(x_list_1,bins=15,kde_kws={"color":"seagreen", "lw":3 }, hist_kws={ "color": "lightskyblue"}) 

plt.ylabel("count")
#plt.xlabel("gs_energy (MeV)")
#plt.legend(loc = 'lower left')

plt.xticks(np.arange(x_tick_min,x_tick_max,x_tick_gap),fontsize = 0)
plt.yticks(np.arange(y_tick_min,y_tick_max+0.01,y_tick_gap),fontsize = y_fontsize)

plt.xlim((x_lim_min,x_lim_max))
plt.ylim((y_lim_min,y_lim_max))
plt.yticks(np.arange(0,15,5))


#########################################################
## corr_loss Nmax20
#########################################################
file_path = "gs_NN_info_corr_He4_20.txt"
data_num = len(open(file_path,'rU').readlines())
raw_data = np.zeros((data_num,3),dtype = np.float)
input_file_2(file_path,raw_data)
raw_data_new = raw_data[0:100,:]
raw_data_new_2 = raw_data_new[np.where((raw_data_new[:,0]>-28)&(raw_data_new[:,0]<-27.4))]
x_list_1 = raw_data_new_2[:,0]



matplotlib.rcParams['xtick.direction'] = 'in' 
matplotlib.rcParams['ytick.direction'] = 'in' 
ax6 = fig_1.add_subplot(6,2,11)
plt.tick_params(top=True,bottom=True,left=True,right=False)

sns.set_palette("hls") 
mpl.rc("figure")#, figsize=(6,4)) 
l2=sns.distplot(x_list_1,bins=15,kde_kws={"color":"seagreen", "lw":3 }, hist_kws={ "color": "lightskyblue"}) 

plt.ylabel("count")
plt.xlabel(r"$E_{gs} \ \rm{(MeV)}$")
#plt.legend(loc = 'lower left')

plt.xticks(np.arange(x_tick_min,x_tick_max,x_tick_gap),fontsize = y_fontsize)
plt.yticks(np.arange(y_tick_min,y_tick_max+0.01,y_tick_gap),fontsize = y_fontsize)

plt.xlim((x_lim_min,x_lim_max))
plt.ylim((y_lim_min,y_lim_max))
plt.yticks(np.arange(0,15,5))




####################################################
## balance_loss Nmax_10
####################################################
file_path = "gs_NN_info_balance_He4_10.txt"
data_num = len(open(file_path,'rU').readlines())
raw_data = np.zeros((data_num,3),dtype = np.float)
input_file_2(file_path,raw_data)
raw_data_new = raw_data[0:100,:]

x_list = raw_data_new[:,0]
y_list = raw_data_new[:,1]
raw_data_new_2 = raw_data_new[np.where((raw_data_new[:,0]>-28)&(raw_data_new[:,0]<-27.4))]
x_list_1 = raw_data_new_2[:,0]


#fig_1 = plt.figure('fig_1')
#l = plt.scatter(x_list,y_list,color='b',s = 20, marker = 'o')
#
#plt.title("He4_Multi-NN_Nmax4~18")
#plt.xlabel("gs_energy (MeV)")
#plt.ylabel("loss")
##plt.legend(loc = 'lower left')
#plt.ylim((0,0.05))
#plt.xlim((-28,-27.4))
##plt.savefig('Li6_radius_NN_prediction.jpg')
#plot_path = 'Multi-NN.eps'
#plt.savefig(plot_path)
#fig_1.show()



matplotlib.rcParams['xtick.direction'] = 'in' 
matplotlib.rcParams['ytick.direction'] = 'in' 
ax1 = fig_1.add_subplot(6,2,2)

plt.tick_params(top=True,bottom=True,left=True,right=False)

sns.set_palette("hls") 
mpl.rc("figure")#, figsize=(6,4)) 
l1=sns.distplot(x_list_1,bins=15,kde_kws={"color":"seagreen", "lw":3 }, hist_kws={ "color": "lightskyblue"}) 

#plt.hist(x_list_1,200,normed=2,histtype='bar',facecolor='yellowgreen',alpha=0.75)
#l = plt.scatter(x_list,y_list,color='k',linestyle='--',s = 10, marker = 'x', label='E(infinite)')
#
#
#plt.title("E(converge)="+str(gs_converge))
plt.ylabel("count")
#plt.xlabel("gs_energy (MeV)")
#plt.legend(loc = 'lower left')
plt.xticks(np.arange(x_tick_min,x_tick_max,x_tick_gap),fontsize = 0)
plt.yticks(np.arange(y_tick_min,y_tick_max+0.01,y_tick_gap),fontsize = y_fontsize)
plt.xlim((x_lim_min,x_lim_max))
plt.ylim((y_lim_min,y_lim_max))



#########################################################
## balance_loss Nmax12
#########################################################
file_path = "gs_NN_info_balance_He4_12.txt"
data_num = len(open(file_path,'rU').readlines())
raw_data = np.zeros((data_num,3),dtype = np.float)
input_file_2(file_path,raw_data)
raw_data_new = raw_data[0:100,:]
raw_data_new_2 = raw_data_new[np.where((raw_data_new[:,0]>-28)&(raw_data_new[:,0]<-27.4))]
x_list_1 = raw_data_new_2[:,0]



matplotlib.rcParams['xtick.direction'] = 'in' 
matplotlib.rcParams['ytick.direction'] = 'in' 
plt.subplot(624)
plt.tick_params(top=True,bottom=True,left=True,right=False)

sns.set_palette("hls") 
mpl.rc("figure")#, figsize=(6,4)) 
l2=sns.distplot(x_list_1,bins=15,kde_kws={"color":"seagreen", "lw":3 }, hist_kws={ "color": "lightskyblue"}) 

plt.ylabel("count")
#plt.xlabel("gs_energy (MeV)")
#plt.legend(loc = 'lower left')
plt.xticks(np.arange(x_tick_min,x_tick_max,x_tick_gap),fontsize = 0)
plt.yticks(np.arange(y_tick_min,y_tick_max+0.01,y_tick_gap),fontsize = y_fontsize)


plt.xlim((x_lim_min,x_lim_max))
plt.ylim((y_lim_min,y_lim_max))
plt.yticks(np.arange(0,15,5))


#########################################################
## balance_loss Nmax14
#########################################################
file_path = "gs_NN_info_balance_He4_14.txt"
data_num = len(open(file_path,'rU').readlines())
raw_data = np.zeros((data_num,3),dtype = np.float)
input_file_2(file_path,raw_data)
raw_data_new = raw_data[0:100,:]
raw_data_new_2 = raw_data_new[np.where((raw_data_new[:,0]>-28)&(raw_data_new[:,0]<-27.4))]
x_list_1 = raw_data_new_2[:,0]



matplotlib.rcParams['xtick.direction'] = 'in' 
matplotlib.rcParams['ytick.direction'] = 'in' 
plt.subplot(626)
plt.tick_params(top=True,bottom=True,left=True,right=False)

sns.set_palette("hls") 
mpl.rc("figure")#, figsize=(6,4)) 
l2=sns.distplot(x_list_1,bins=15,kde_kws={"color":"seagreen", "lw":3 }, hist_kws={ "color": "lightskyblue"}) 

plt.ylabel("count")
#plt.xlabel("gs_energy (MeV)")
#plt.legend(loc = 'lower left')
plt.xticks(np.arange(x_tick_min,x_tick_max,x_tick_gap),fontsize = 0)
plt.yticks(np.arange(y_tick_min,y_tick_max+0.01,y_tick_gap),fontsize = y_fontsize)


plt.xlim((x_lim_min,x_lim_max))
plt.ylim((y_lim_min,y_lim_max))
plt.yticks(np.arange(0,15,5))



#########################################################
## balance_loss Nmax16
#########################################################
file_path = "gs_NN_info_balance_He4_16.txt"
data_num = len(open(file_path,'rU').readlines())
raw_data = np.zeros((data_num,3),dtype = np.float)
input_file_2(file_path,raw_data)
raw_data_new = raw_data[0:100,:]
raw_data_new_2 = raw_data_new[np.where((raw_data_new[:,0]>-28)&(raw_data_new[:,0]<-27.4))]
x_list_1 = raw_data_new_2[:,0]



matplotlib.rcParams['xtick.direction'] = 'in' 
matplotlib.rcParams['ytick.direction'] = 'in' 
plt.subplot(628)
plt.tick_params(top=True,bottom=True,left=True,right=False)

sns.set_palette("hls") 
mpl.rc("figure")#, figsize=(6,4)) 
l2=sns.distplot(x_list_1,bins=15,kde_kws={"color":"seagreen", "lw":3 }, hist_kws={ "color": "lightskyblue"}) 

plt.ylabel("count")
#plt.xlabel("gs_energy (MeV)")
#plt.legend(loc = 'lower left')
plt.xticks(np.arange(x_tick_min,x_tick_max,x_tick_gap),fontsize = 0)
plt.yticks(np.arange(y_tick_min,y_tick_max+0.01,y_tick_gap),fontsize = y_fontsize)


plt.xlim((x_lim_min,x_lim_max))
plt.ylim((y_lim_min,y_lim_max))
plt.yticks(np.arange(0,15,5))


#########################################################
## balance_loss Nmax18
#########################################################
file_path = "gs_NN_info_balance_He4_16.txt"
data_num = len(open(file_path,'rU').readlines())
raw_data = np.zeros((data_num,3),dtype = np.float)
input_file_2(file_path,raw_data)
raw_data_new = raw_data[0:100,:]
raw_data_new_2 = raw_data_new[np.where((raw_data_new[:,0]>-28)&(raw_data_new[:,0]<-27.4))]
x_list_1 = raw_data_new_2[:,0]



matplotlib.rcParams['xtick.direction'] = 'in' 
matplotlib.rcParams['ytick.direction'] = 'in' 
ax55 = fig_1.add_subplot(6,2,10)
plt.tick_params(top=True,bottom=True,left=True,right=False)

sns.set_palette("hls") 
mpl.rc("figure")#, figsize=(6,4)) 
l2=sns.distplot(x_list_1,bins=15,kde_kws={"color":"seagreen", "lw":3 }, hist_kws={ "color": "lightskyblue"}) 

plt.ylabel("count")
#plt.xlabel("gs_energy (MeV)")
#plt.legend(loc = 'lower left')

plt.xticks(np.arange(x_tick_min,x_tick_max,x_tick_gap),fontsize = 0)
plt.yticks(np.arange(y_tick_min,y_tick_max+0.01,y_tick_gap),fontsize = y_fontsize)

plt.xlim((x_lim_min,x_lim_max))
plt.ylim((y_lim_min,y_lim_max))
plt.yticks(np.arange(0,15,5))


#########################################################
## balance_loss Nmax20
#########################################################
file_path = "gs_NN_info_balance_He4_20.txt"
data_num = len(open(file_path,'rU').readlines())
raw_data = np.zeros((data_num,3),dtype = np.float)
input_file_2(file_path,raw_data)
raw_data_new = raw_data[0:100,:]
raw_data_new_2 = raw_data_new[np.where((raw_data_new[:,0]>-28)&(raw_data_new[:,0]<-27.4))]
x_list_1 = raw_data_new_2[:,0]



matplotlib.rcParams['xtick.direction'] = 'in' 
matplotlib.rcParams['ytick.direction'] = 'in' 
ax66 = fig_1.add_subplot(6,2,12)
plt.tick_params(top=True,bottom=True,left=True,right=False)

sns.set_palette("hls") 
mpl.rc("figure")#, figsize=(6,4)) 
l2=sns.distplot(x_list_1,bins=15,kde_kws={"color":"seagreen", "lw":3 }, hist_kws={ "color": "lightskyblue"}) 

plt.ylabel("count")
plt.xlabel(r"$E_{gs} \ \rm{(MeV)}$")
#plt.legend(loc = 'lower left')

plt.xticks(np.arange(x_tick_min,x_tick_max,x_tick_gap),fontsize = y_fontsize)
plt.yticks(np.arange(y_tick_min,y_tick_max+0.01,y_tick_gap),fontsize = y_fontsize)

plt.xlim((x_lim_min,x_lim_max))
plt.ylim((y_lim_min,y_lim_max))
plt.yticks(np.arange(0,15,5))









plot_path = 'multi-NN_distribution.pdf'
plt.savefig(plot_path)
#fig_2.show()





#input()
