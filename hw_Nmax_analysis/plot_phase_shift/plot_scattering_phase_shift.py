import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import interpolate
import re
from math import log
from math import e


def input_residual_data(file_path,raw_data):
    with open(file_path,'r') as f_2:
        count = len(open(file_path,'rU').readlines())
        data = f_2.readlines()
        loop2 = 0
        loop1 = 0
        wtf = re.match('#', 'abc',flags=0)
        while loop1 < count:
            if ( re.match('#', data[loop1],flags=0) == wtf):
                temp_2 = data[loop1][6:]
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",temp_2)
                if ( float(temp_1[energy_line])*pow(10,int(temp_1[energy_line+1]))  <= E_max):
                    raw_data[loop2][0] = float(temp_1[theo_line])     * pow(10,int(temp_1[theo_line+1]))
                    raw_data[loop2][1] = float(temp_1[exp_line])      * pow(10,int(temp_1[exp_line+1] ))
                    raw_data[loop2][2] = float(temp_1[exp_error_line])* pow(10,int(temp_1[exp_error_line+1]))
                    raw_data[loop2][3] = float(temp_1[energy_line])   * pow(10,int(temp_1[energy_line+1]))
                #print raw_data[loop2][0] 
                    loop2 = loop2 + 1
            loop1 = loop1 + 1
#        print ('vec_input='+str(vec_input))
        #print ('raw_data[0][0]='+str(raw_data[0][0]))


def plot_phase_shift():
    file_path = "./residual_data_weight10.txt"
    with open(file_path,'r') as f:
        count = len(open(file_path,'rU').readlines())
        data = f.readlines()
        loop2 = 0
        loop1 = 0
        wtf = re.match('#', 'abc',flags=0)
        while loop1 < count :
            if ( re.match('#', data[loop1],flags=0) == wtf):
                temp_2 = data[loop1][6:]
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",temp_2)
                if ( float(temp_1[energy_line])*pow(10,int(temp_1[energy_line+1]))  <= E_max):
                    loop2 = loop2 + 1
            loop1 = loop1 + 1
    scattering_data_num = loop2
    print ('scattering_data_num='+str(scattering_data_num))
    raw_data_1 = np.zeros((scattering_data_num,4),dtype = np.float)
    input_residual_data(file_path,raw_data_1)
    fig_1 = plt.figure('fig_1')
    plt.subplots_adjust(wspace =0.5, hspace =0)


    ## pp 1S0
    matplotlib.rcParams['xtick.direction'] = 'in' 
    matplotlib.rcParams['ytick.direction'] = 'in' 
    plt.subplot(421)
    plt.tick_params(top=True,bottom=True,left=True,right=False)
    start_line = 0
    x_list     = raw_data_1[start_line:start_line+8,3]  # energy
    y_list_1   = raw_data_1[start_line:start_line+8,1]  # exp
    y_list_2   = raw_data_1[start_line:start_line+8,0]  # theo
    x_list_new = np.linspace(np.min(x_list),np.max(x_list),num=plot_interpol_count)
    func       = interpolate.interp1d(x_list,y_list_2,kind=kind)
    y_list_2_new = func(x_list_new) 
    #print('y_list='+str(y_list_2))
    #print('y_list_2_new='+str(y_list_2_new))
    l_exp      = plt.scatter(x_list,y_list_1,color = 'k',s = point_size,marker ='.')
    l_theo     = plt.plot   (x_list,y_list_2,color = 'b',linestyle = '-') 
    
    #plt.xlabel(fontsize = x_fontsize)
    plt.ylabel('$\delta(^1S_0)$(deg)')
    plt.yticks(np.arange(-30,91,30),fontsize = y_fontsize)
    plt.xticks(np.arange(0,225,50),[])

    ## pp 3P0
    matplotlib.rcParams['xtick.direction'] = 'in' 
    matplotlib.rcParams['ytick.direction'] = 'in' 
    plt.subplot(422)
    plt.tick_params(top=True,bottom=True,left=True,right=False)
    start_line = 24
    x_list     = raw_data_1[start_line:start_line+8,3]  # energy
    y_list_1   = raw_data_1[start_line:start_line+8,1]  # exp
    y_list_2   = raw_data_1[start_line:start_line+8,0]  # theo
    x_list_new = np.linspace(np.min(x_list),np.max(x_list),num=plot_interpol_count)
    func       = interpolate.interp1d(x_list,y_list_2,kind=kind)
    y_list_2_new = func(x_list_new) 

    l_exp      = plt.scatter(x_list,y_list_1,color = 'k',s = point_size,marker ='.')
    l_theo     = plt.plot   (x_list_new,y_list_2_new,color = 'b',linestyle = '-') 

    plt.ylabel('$\delta(^3P_0)$(deg)')
    plt.yticks(np.arange(-40,81,40),fontsize = y_fontsize)
    plt.xticks(np.arange(0,225,50),[])


    ## pp 3P1
    matplotlib.rcParams['xtick.direction'] = 'in' 
    matplotlib.rcParams['ytick.direction'] = 'in' 
    plt.subplot(423)
    plt.tick_params(top=True,bottom=True,left=True,right=False)
    start_line = 32
    x_list     = raw_data_1[start_line:start_line+8,3]  # energy
    y_list_1   = raw_data_1[start_line:start_line+8,1]  # exp
    y_list_2   = raw_data_1[start_line:start_line+8,0]  # theo
    x_list_new = np.linspace(np.min(x_list),np.max(x_list),num=plot_interpol_count)
    func       = interpolate.interp1d(x_list,y_list_2,kind=kind)
    y_list_2_new = func(x_list_new) 

    l_exp      = plt.scatter(x_list,y_list_1,color = 'k',s = point_size,marker ='.')
    l_theo     = plt.plot   (x_list_new,y_list_2_new,color = 'b',linestyle = '-') 

    plt.ylabel('$\delta(^3P_1)$(deg)')
    plt.yticks(np.arange(-60,21,20),fontsize = y_fontsize)
    plt.xticks(np.arange(0,225,50),[])

    ## pp 3P2
    matplotlib.rcParams['xtick.direction'] = 'in' 
    matplotlib.rcParams['ytick.direction'] = 'in' 
    plt.subplot(424)
    plt.tick_params(top=True,bottom=True,left=True,right=False)
    start_line = 48
    x_list     = raw_data_1[start_line:start_line+8,3]  # energy
    y_list_1   = raw_data_1[start_line:start_line+8,1]  # exp
    y_list_2   = raw_data_1[start_line:start_line+8,0]  # theo
    x_list_new = np.linspace(np.min(x_list),np.max(x_list),num=plot_interpol_count)
    func       = interpolate.interp1d(x_list,y_list_2,kind=kind)
    y_list_2_new = func(x_list_new) 

    l_exp      = plt.scatter(x_list,y_list_1,color = 'k',s = point_size,marker ='.')
    l_theo     = plt.plot   (x_list_new,y_list_2_new,color = 'b',linestyle = '-') 

    plt.ylabel('$\delta(^3P_2)$(deg)')
    plt.yticks(np.arange(-6,22,6),fontsize = y_fontsize)
    plt.xticks(np.arange(0,225,50),[])

    ## pp 1D2
    matplotlib.rcParams['xtick.direction'] = 'in' 
    matplotlib.rcParams['ytick.direction'] = 'in' 
    plt.subplot(425)
    plt.tick_params(top=True,bottom=True,left=True,right=False)
    start_line = 8
    x_list     = raw_data_1[start_line:start_line+8,3]  # energy
    y_list_1   = raw_data_1[start_line:start_line+8,1]  # exp
    y_list_2   = raw_data_1[start_line:start_line+8,0]  # theo
    x_list_new = np.linspace(np.min(x_list),np.max(x_list),num=plot_interpol_count)
    func       = interpolate.interp1d(x_list,y_list_2,kind=kind)
    y_list_2_new = func(x_list_new) 

    l_exp      = plt.scatter(x_list,y_list_1,color = 'k',s = point_size,marker ='.')
    l_theo     = plt.plot   (x_list_new,y_list_2_new,color = 'b',linestyle = '-') 

    plt.ylabel('$\delta(^1D_2)$(deg)')
    plt.yticks(np.arange(-4,13,4),fontsize = y_fontsize)
    plt.xticks(np.arange(0,225,50),[])

    ## pp 3F2
    matplotlib.rcParams['xtick.direction'] = 'in' 
    matplotlib.rcParams['ytick.direction'] = 'in' 
    plt.subplot(426)
    plt.tick_params(top=True,bottom=True,left=True,right=False)
    start_line = 64
    x_list     = raw_data_1[start_line:start_line+8,3]  # energy
    y_list_1   = raw_data_1[start_line:start_line+8,1]  # exp
    y_list_2   = raw_data_1[start_line:start_line+8,0]  # theo
    x_list_new = np.linspace(np.min(x_list),np.max(x_list),num=plot_interpol_count)
    func       = interpolate.interp1d(x_list,y_list_2,kind=kind)
    y_list_2_new = func(x_list_new) 

    l_exp      = plt.scatter(x_list,y_list_1,color = 'k',s = point_size,marker ='.')
    l_theo     = plt.plot   (x_list_new,y_list_2_new,color = 'b',linestyle = '-') 

    plt.ylabel('$\delta(^3F_2)$(deg)')
    plt.yticks(np.arange(-0.2,2.6,0.6),fontsize = y_fontsize)
    plt.xticks(np.arange(0,225,50),[])

    ## pp 3F3
    matplotlib.rcParams['xtick.direction'] = 'in' 
    matplotlib.rcParams['ytick.direction'] = 'in' 
    plt.subplot(427)
    plt.tick_params(top=True,bottom=True,left=True,right=False)
    start_line = 40
    x_list     = raw_data_1[start_line:start_line+8,3]  # energy
    y_list_1   = raw_data_1[start_line:start_line+8,1]  # exp
    y_list_2   = raw_data_1[start_line:start_line+8,0]  # theo
    x_list_new = np.linspace(np.min(x_list),np.max(x_list),num=plot_interpol_count)
    func       = interpolate.interp1d(x_list,y_list_2,kind=kind)
    y_list_2_new = func(x_list_new) 

    l_exp      = plt.scatter(x_list,y_list_1,color = 'k',s = point_size,marker ='.')
    l_theo     = plt.plot   (x_list_new,y_list_2_new,color = 'b',linestyle = '-') 

    plt.ylabel('$\delta(^3F_3)$(deg)')
    plt.yticks(np.arange(-4,0.5,1),fontsize = y_fontsize)
    plt.xticks(np.arange(0,225,50),[])





    plot_path = 'pp_phase_shift_2.jpg'
    plt.savefig(plot_path)
    plt.show()




#    plt.close("all")


##########################################################
##########################################################
### setting parameters
##########################################################
##########################################################
theo_line          = 5
exp_line           = 7
exp_error_line     = 9
energy_line        = 1
E_max              = 200
plot_interpol_count= 1000
point_size         = 50
x_fontsize         = 8
y_fontsize         = 8
kind               = 'linear' 
plot_phase_shift()
