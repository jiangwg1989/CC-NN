import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
    file_path = "residual_data.txt"
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
    ## pp 1s0
#    plt.subplot(421)
    start_line = 0
    x_list     = raw_data_1[start_line:start_line+8,3]  # energy
    y_list_1   = raw_data_1[start_line:start_line+8,1]  # exp
    y_list_2   = raw_data_1[start_line:start_line+8,0]  # theo
    l_exp      = plt.scatter(x_list,y_list_1,color = 'k',s = 15,marker ='.')
    l_theo     = plt.plot   (x_list,y_list_2,color = 'b',linestyle = '--') 
    plt.ylabel('$/delta(^1S_0)$(deg)')

    plot_path = './plot_output/pp_phase_shift.eps'
    plt.savefig(plot_path)
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

plot_phase_shift()
