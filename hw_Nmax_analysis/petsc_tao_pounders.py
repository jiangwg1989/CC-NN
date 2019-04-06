import sys, petsc4py
petsc4py.init(sys.argv)


import os
import numpy as np
from petsc4py import PETSc
import math
import re
from scipy import interpolate

np.set_printoptions(suppress='Ture')

def start_from_break_point(file_path,line_num):
    with open(file_path,'r') as f_1:
        data = f_1.readlines()
        wtf = re.match('#', 'abc',flags=0)
        temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[line_num-1])
        DNNLO450_input[0:6] = temp_1[0:6] 
        temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[line_num])
        DNNLO450_input[6:12] = temp_1[0:6] 
        temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[line_num+1])
        DNNLO450_input[12:17] = temp_1[0:5] 



######################################################
######################################################
### run nuclear matter program 
######################################################
######################################################
def output_ccm_in_file(file_path,vec_input,particle_num,matter_type,density,nmax):
    with open(file_path,'w') as f_1:
        f_1.write('!Chiral order for Deltas(LO = 0,NLO=2,NNLO=3,N3LO=4) and cutoff'+'\n')
        f_1.write('3, 450\n')
        f_1.write('! cE and cD 3nf parameters:'+'\n' )
        f_1.write('%.12f, %.12f\n' % (vec_input[16],vec_input[15]))
        f_1.write('! LEC ci \n')
        f_1.write('%.12f, %.12f, %.12f, %.12f \n' % (vec_input[0],vec_input[1],vec_input[2],vec_input[3]))
        f_1.write('!c1s0 & c3s1 \n')
        f_1.write('%.12f, %.12f, %.12f, %.12f, %.12f, %.12f \n' % (vec_input[6],vec_input[4],vec_input[5],vec_input[7],vec_input[7],vec_input[7]))
        f_1.write('! cnlo(7) \n')
        f_1.write('%.12f, %.12f, %.12f, %.12f, %.12f, %.12f, %.12f \n' % (vec_input[8],vec_input[9],vec_input[12],vec_input[10],vec_input[13],vec_input[14],vec_input[11]))
        f_1.write('! number of particles'+'\n')
        f_1.write('%d\n' % (particle_num) )
        f_1.write('! specify: pnm/snm, input type: density/kfermi'+'\n')
        f_1.write(matter_type+', density'+'\n')
        f_1.write('! specify boundary conditions (PBC/TABC/TABCsp)'+'\n')       
        f_1.write('PBC'+'\n')
        f_1.write('! dens/kf, ntwist,  nmax'+'\n')
        f_1.write('%.12f, 1, %d\n' % (density, nmax))
        f_1.write('! specify cluster approximation: CCD, CCDT'+'\n')
        f_1.write('CCD(T)'+'\n')
        f_1.write('! tnf switch (T/F) and specify 3nf approximation: 0=tnf0b, 1=tnf1b, 2=tnf2b'+'\n')
        f_1.write('T, 3'+'\n')
        f_1.write('! 3nf cutoff(MeV),non-local reg. exp'+'\n')
        f_1.write('450, 3'+'\n')

def read_nucl_matt_out(file_path):  # converge: flag = 1    converge: flag =0
    with open(file_path,'r') as f_1:
        converge_flag = int (1)
        count = len(open(file_path,'rU').readlines())
        #if ( count > 1500 ):
        #    converge_flag =int (0)
        data =  f_1.readlines()
        wtf = re.match('#', 'abc',flags=0)
        for loop1 in range(0,count):
            if ( re.search('E/N', data[loop1],flags=0) != wtf):
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop1+1])
                energy_per_nucleon = float(temp_1[2])
                return energy_per_nucleon#,converge_flag
        print ('No "E/A" found in the file:'+file_path)
        return float('nan')

def nuclear_matter(vec_input):
    neutron_num  = 14
    particle_num = 28
    density      = 0
    density_min  = 0.14
    density_max  = 0.22
    nmax         = 2
    interpolation_count = 1000
    snm_dens    = np.zeros(5)    
    snm_energy_per_nucleon = np.zeros(5)    
    snm_dens_new = np.zeros(interpolation_count)
    snm_energy_per_nucleon_new = np.zeros(interpolation_count)

    #calculate snm saturation point
    for loop1 in range(5):
        density = density_min+loop1*0.02
        nucl_matt_in_dir   = './nuclear_matter/ccm_in_snm_%.2f' % (density) 
        nucl_matt_out_dir  = './nuclear_matter/snm_rho_%.2f.out' % (density)
        output_ccm_in_file(nucl_matt_in_dir,vec_input,particle_num,'snm',density,nmax)
        #print ('mpirun -np 8 '+nucl_matt_exe+' '+nucl_matt_in_dir+' > '+nucl_matt_out_dir) 
        os.system('mpirun -np 8 '+nucl_matt_exe+' '+nucl_matt_in_dir+' > '+nucl_matt_out_dir)
        snm_dens[loop1] = density
        snm_energy_per_nucleon[loop1] = read_nucl_matt_out(nucl_matt_out_dir)

    f = interpolate.interp1d(snm_dens,snm_energy_per_nucleon,kind = "quadratic")
    snm_dens_new = np.linspace(density_min,density_max,num = interpolation_count) 
    snm_energy_per_nucleon_new = f (snm_dens_new)
    saturation_energy = np.min(snm_energy_per_nucleon_new)
    temp = snm_dens_new[np.where(snm_energy_per_nucleon_new == saturation_energy)]
    saturation_dens   = temp[0]
    df = np.diff(snm_energy_per_nucleon_new) / np.diff(snm_dens_new)
    ddf= np.zeros(len(df)+1)
    ddf[0:len(df)-1] = np.diff(df) / np.diff(snm_dens_new[0:len(snm_dens_new)-1])
    ddf[len(df)]     = ddf[len(df)-2]
    ddf[len(df)-1]     = ddf[len(df)-2]
    temp = ddf[np.where(snm_energy_per_nucleon_new == saturation_energy)]
    ddf_saturation_dens = temp[0]
    K0 = 9 * pow(saturation_dens,2)* ddf_saturation_dens
 
    #calculate pnm
    nucl_matt_in_dir   = './nuclear_matter/ccm_in_pnm'
    nucl_matt_out_dir  = './nuclear_matter/pnm_rho.out'
    density = saturation_dens
    output_ccm_in_file(nucl_matt_in_dir,vec_input,neutron_num,'pnm',density,nmax)
    os.system('mpirun -np 8 '+nucl_matt_exe+' '+nucl_matt_in_dir+' > '+nucl_matt_out_dir)
    pnm_energy_per_nucleon = read_nucl_matt_out(nucl_matt_out_dir)



    #print ('vec_input='+str(vec_input))
    #print ('pnm_energy_per_nucleon='+str(pnm_energy_per_nucleon))
    return pnm_energy_per_nucleon, saturation_dens, saturation_energy, K0, ddf_saturation_dens



######################################################
######################################################
### run nucleon-scattering program 
######################################################
######################################################
def nucleon_scattering(vec_input):
    raw_data = np.zeros((scattering_data_num,4))
    #vec_input_2 = vec_input.astype('float32')
    file_path = 'test.ini'
    with open (file_path,'w') as f_1:
        f_1.write('include default_inputs/default.ini\n\
include default_inputs/print_progress.ini\n\
include default_inputs/io.ini\n\
include ncsm_450.ini\n\
\n\
program_mode=single\n\
\n\
#define nuclear interaction\n\
include interactions/DNNLO_450-RSe2_isoT.ini\n\
#activate electromagnetic interaction(s)\n\
include include/electromagnetic/electromagnetic_grenada.ini\n\
\n\
#set isospin limits (-2:pp 0:np +2:nn (no phases for nn))\n\
tz_min=-2\n\
tz_max=+2\n\
\n\
process_data_record.print_record=yes\n\
\n\
#select database file of phases. \n\
#database_file=phases/granada_NN_1S0.dat\n\
database_file=phases/granada_NN_N3LOall.dat\n\
\n\
include include/effective_range.ini\n\
\n\
#maximum energy Tlab (MeV) to evaluate\n\
Emin=0.0\n\
Emax=350.0\n\
\n\
#number of grid points in R-matrix evaluation.\n\
#more points -> longer run-time \n\
N=150\n\
\n\
#PWD basis limits, if less than e.g. PWs in database, the\n\
#code will return an error\n\
L_min=0\n\
L_max=6\n\
\n\
# execute with ns_pounders.exe to get a residual list in file\n\
# with name prefix.txt. This one is easier to read than the results-file\n\
residual_list.print=yes\n\
residual_list.print.prefix = residual_data\n\
\n\
\n\
##################\n\
# interaction LECs\n\
\n\
\n\
par.c1  = %.5f\n\
par.c2  = %.5f\n\
par.c3  = %.5f\n\
par.c4  = %.5f\n\
par.c_D  = %.5f\n\
par.c_E  = %.5f\n\
\n\
#LO + isospin-breaking contacts T=1 (in nn,pp,np channels)\n\
par.Ct_1S0np   =  %.14f\n\
par.Ct_1S0nn   =  %.14f\n\
par.Ct_1S0pp   =  %.14f\n\
\n\
#LO + isospin-breaking contacts T=0 (only in np channels)\n\
par.Ct_3S1     =  %.14f\n\
\n\
#NLO contacts T=1 (in nn,pp,np channels)\n\
par.C_1S0      =  %.14f\n\
par.C_3P0      =  %.14f\n\
par.C_3P1      =  %.14f\n\
par.C_3P2      =  %.14f\n\
\n\
#NLO contacts T=0 (only in np channels)\n\
par.C_1P1      =  %.14f\n\
par.C_3S1      =  %.14f\n\
par.C_3S1-3D1  =  %.14f\n\
\n\
par[]=par.c1\n\
par[]=par.c2\n\
par[]=par.c3\n\
par[]=par.c4\n\
par[]=par.c_D\n\
par[]=par.c_E\n\
par[]=par.Ct_1S0np\n\
par[]=par.Ct_1S0nn\n\
par[]=par.Ct_1S0pp\n\
par[]=par.Ct_3S1\n\
par[]=par.C_1S0\n\
par[]=par.C_3P0\n\
par[]=par.C_3P1\n\
par[]=par.C_3P2\n\
par[]=par.C_1P1\n\
par[]=par.C_3S1\n\
par[]=par.C_3S1-3D1\n\
\n' % (float(vec_input[0]),float(vec_input[1]),float(vec_input[2]),float(vec_input[3]),
       float(vec_input[15]),float(vec_input[16]),
       float(vec_input[4]),float(vec_input[5]),float(vec_input[6]),float(vec_input[7]),
       float(vec_input[8]),float(vec_input[9]),float(vec_input[10]),float(vec_input[11]),
       float(vec_input[12]),float(vec_input[13]),float(vec_input[14])))
    os.system(pounder_exe_dir+' '+ini_file_dir+' > '+output_file_dir)
    
    with open(residual_data,'r') as f_2:
        count = len(open(residual_data,'rU').readlines())
        data = f_2.readlines() 
        loop2 = 0 
        loop1 = 0 
        wtf = re.match('#', 'abc',flags=0)
        while loop1 < count:
            if ( re.match('#', data[loop1],flags=0) == wtf):
                temp_2 = data[loop1][6:]
                temp_1 = re.findall(r"[-+]?\d+\.?\d*",temp_2)
                #temp_1 = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
                #print('temp_1'+str(temp_1)) 
                if ( float(temp_1[energy_line])*pow(10,int(temp_1[energy_line+1]))  <= E_max):
                    if (float(temp_1[theo_line])!=0):
                        if ((float(temp_1[0]) < 1000) or  (float(temp_1[0]) == 300002)):
                            raw_data[loop2][0] = float(temp_1[theo_line])     * pow(10,int(temp_1[theo_line+1]))
                            raw_data[loop2][1] = float(temp_1[exp_line])      * pow(10,int(temp_1[exp_line+1] ))
                            raw_data[loop2][2] = float(temp_1[exp_error_line])* pow(10,int(temp_1[exp_error_line+1]))
                            raw_data[loop2][3] = float(temp_1[energy_line])   * pow(10,int(temp_1[energy_line+1]))
                #print raw_data[loop2][0] 
                            loop2 = loop2 + 1 
            loop1 = loop1 + 1
#        print ('vec_input='+str(vec_input))
#        print ('raw_data[0][0]='+str(raw_data))
        return raw_data
        #print ('count='+str(count))
        #print (str(loop2))  
        #print ('theo_line='+str(theo_line))  

######################################################
######################################################
##### read light nuclei 
######################################################
######################################################
def light_nuclei():
    H2_theo_BE = 0
    H2_exp_BE  = -2.224575
    H2_theo_R  = 0
    H2_exp_R   = 1.97535
    H3_theo_BE = 0
    H3_exp_BE  = -8.48
    H3_theo_R  = 0
    H3_exp_R   = 1.587
    He3_theo_BE= 0
    He3_exp_BE = -7.718
    He3_theo_R = 0
    He3_exp_R  = 1.77
    He4_theo_BE= 0
    He4_exp_BE = -28.30
    He4_theo_R = 0
    He4_exp_R  = 1.455
    results_file = 'results.dat'
    with open(results_file,'r') as f_1:
        count = len(open(results_file,'rU').readlines())
        data = f_1.readlines() 
        loop1 = 0
        wtf = re.match('#', 'abc',flags=0)
        while loop1 < count :
            if ( re.search('H2 BINDING ENERGY', data[loop1],flags=0) != wtf):
                temp = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
                H2_theo_BE = float(temp[1])
                temp = re.findall(r"[-+]?\d+\.?\d*",data[loop1+1])
                H2_exp_BE = float(temp[0])
            if ( re.search('H2 STRUCT RADIUS', data[loop1],flags=0) != wtf):
                temp = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
                H2_theo_R = float(temp[1])
                temp = re.findall(r"[-+]?\d+\.?\d*",data[loop1+1])
                H2_exp_R = float(temp[0])
            if ( re.search('H2 QUADRUPOLE MOMENT', data[loop1],flags=0) != wtf):
                temp = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
                H2_quad_moment = float(temp[1])
            if ( re.search('H2 D-STATE PROBABILITY', data[loop1],flags=0) != wtf):
                temp = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
                H2_D_state_probability = float(temp[1])
            if ( re.search('H3 BINDING ENERGY', data[loop1],flags=0) != wtf):
                temp = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
                H3_theo_BE = float(temp[1])
                temp = re.findall(r"[-+]?\d+\.?\d*",data[loop1+1])
                H3_exp_BE = float(temp[0])
            if ( re.search('H3 POINT-P RADIUS', data[loop1],flags=0) != wtf):
                temp = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
                H3_theo_R = float(temp[1])
                temp = re.findall(r"[-+]?\d+\.?\d*",data[loop1+1])
                H3_exp_R = float(temp[0])
            if ( re.search('He3 BINDING ENERGY', data[loop1],flags=0) != wtf):
                temp = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
                He3_theo_BE = float(temp[1])
                temp = re.findall(r"[-+]?\d+\.?\d*",data[loop1+1])
                He3_exp_BE = float(temp[0])
            if ( re.search('He3 POINT-P RADIUS', data[loop1],flags=0) != wtf):
                temp = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
                He3_theo_R = float(temp[1])
                temp = re.findall(r"[-+]?\d+\.?\d*",data[loop1+1])
                He3_exp_R = float(temp[0])
            if ( re.search('He4 BINDING ENERGY', data[loop1],flags=0) != wtf):
                temp = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
                He4_theo_BE = float(temp[1])
                temp = re.findall(r"[-+]?\d+\.?\d*",data[loop1+1])
                He4_exp_BE = float(temp[0])
            if ( re.search('He4 POINT-P RADIUS', data[loop1],flags=0) != wtf):
                temp = re.findall(r"[-+]?\d+\.?\d*",data[loop1])
                He4_theo_R = float(temp[1])
                temp = re.findall(r"[-+]?\d+\.?\d*",data[loop1+1])
                He4_exp_R = float(temp[0])
            loop1 = loop1 + 1
    return H2_theo_BE,H2_exp_BE,H2_theo_R,H2_exp_R,\
           H3_theo_BE,H3_exp_BE,H3_theo_R,H3_exp_R,\
           He3_theo_BE,He3_exp_BE,He3_theo_R,He3_exp_R,\
           He4_theo_BE,He4_exp_BE,He4_theo_R,He4_exp_R,\
           H2_quad_moment,H2_D_state_probability



######################################################
######################################################
##### Use pounders optimizer 
######################################################
######################################################
class pounders_optimization(object):

    """
    Finds the nonlinear least-squares solution to the model
    y = exp(-b1*x)/(b2+b3*x)  +  e
    """

    def __init__(self):
        #BETA = [3.1,2.7,3.4]
        #scattering_data_num = 380
        NOBSERVATIONS = scattering_data_num + 12
        NPARAMETERS = 13
        raw_data = np.zeros((scattering_data_num,4))
        pounders_loop = 0        

        #np.random.seed(456)
        #x = np.random.rand(NOBSERVATIONS)
        #e = np.random.rand(NOBSERVATIONS)
        #e = 0
        #y = np.exp(-BETA[0]*x)/(BETA[1] + BETA[2]*x) + e
        #y = (x-3)^2         
        #x =
        #y = pow((BETA[0] - 3.1),2)+pow((BETA[1]-2.7),4)+pow((BETA[2]-3.4),2)
        self.scattering_data_num = scattering_data_num
        self.NOBSERVATIONS = NOBSERVATIONS
        self.NPARAMETERS = NPARAMETERS
        self.loop = pounders_loop
        self.pnm = 0
        self.snm_dens = 0
        self.snm = 0        

        #self.x = x
#        y = np.zeros(NOBSERVATIONS)
        weight = np.zeros(NOBSERVATIONS) 
#        self.y      = y
        self.weight = weight
        self.raw_data = raw_data      
#        self.pnm_residual =

    def createVecs(self):
        X = PETSc.Vec().create(PETSc.COMM_SELF)
        X.setSizes(self.NPARAMETERS)
        F = PETSc.Vec().create(PETSc.COMM_SELF)
        F.setSizes(self.NOBSERVATIONS)
        return X, F

    def formInitialGuess(self, X):
        X = X
        XX = X.getArray()
        #print ('x_in='+str(XX))
        #print ("x_in=%f, %f, %f" % (XX[0],XX[1],XX[2]))
        #print (sys.getsizeof(X))
        #X[0] = 3.1
        #X[1] = 2.6
        #X[2] = 3.4

    def formObjective(self, tao, X, F):
        raw_data = self.raw_data
        #set up vec_input
        new_parameter_ = np.append(DNNLO450_input[0:4],X.array / relative_step_size * (DNNLO450_uper_bound[4:17] - DNNLO450_lower_bound[4:17]) + DNNLO450_lower_bound[4:17])
        #vec_input = np.append(DNNLO450_input[0:4],XX)
        vec_input = new_parameter_
        #do the calculation including nucleon scattering, nuclear matter, ncsm for light nuclei
        raw_data                                       = nucleon_scattering(vec_input)
        pnm_energy, saturation_dens, saturation_energy, K0, ddf = nuclear_matter(vec_input)
        H2_theo_BE,H2_exp_BE,H2_theo_R,H2_exp_R,\
        H3_theo_BE,H3_exp_BE,H3_theo_R,H3_exp_R,\
        He3_theo_BE,He3_exp_BE,He3_theo_R,He3_exp_R,\
        He4_theo_BE,He4_exp_BE,He4_theo_R,He4_exp_R,\
        H2_quad_moment,H2_D_state_probability    = light_nuclei()


        #setting weights
        #print ('!!!='+str(pnm_energy))
        # set up the loss function (loss = F^2)
        # fit phase shift
        F.setValues(range(self.scattering_data_num),(raw_data[:,0]-raw_data[:,1])/raw_data[:,2])
        # fit saturation point
        F.setValues(self.scattering_data_num,  (pnm_energy-17.10)        *10. * (10)  * (self.scattering_data_num/2.) )
        F.setValues(self.scattering_data_num+1,(saturation_dens-0.183)   *10. * (500) * (self.scattering_data_num/2.) )
        F.setValues(self.scattering_data_num+2,(saturation_energy+16.50) *10. * (10)  * (self.scattering_data_num/2.) )
        # fit light nuclei
        F.setValues(self.scattering_data_num+3,(H2_theo_BE-H2_exp_BE)   *10. * (10)  * (self.scattering_data_num/2.) )
        F.setValues(self.scattering_data_num+4,(H2_theo_R-H2_exp_R)     *10. * (10)  * (self.scattering_data_num/2.) )
        F.setValues(self.scattering_data_num+5,(H3_theo_BE-H3_exp_BE)   *10. * (10)  * (self.scattering_data_num/2.) )
        F.setValues(self.scattering_data_num+6,(H3_theo_R-H3_exp_R)     *10. * (10)  * (self.scattering_data_num/2.) )
        F.setValues(self.scattering_data_num+7,(He3_theo_BE-He3_exp_BE) *10. * (10)  * (self.scattering_data_num/2.) )
        F.setValues(self.scattering_data_num+8,(He3_theo_R-He3_exp_R)   *10. * (10)  * (self.scattering_data_num/2.) )
        F.setValues(self.scattering_data_num+9,(He4_theo_BE-He4_exp_BE) *10. * (10)  * (self.scattering_data_num/2.) )
        F.setValues(self.scattering_data_num+10,(He4_theo_R-He4_exp_R)  *10. * (10)  * (self.scattering_data_num/2.) )
        F.setValues(self.scattering_data_num+11,(H2_quad_moment-0.27)   *10. * (10)  * (self.scattering_data_num/2.) )

        self.raw_data = raw_data
        temp = F.array.copy()
        temp = np.power(temp,2)
        total_loss = np.sum(temp[0:scattering_data_num-1])

        #new_parameter_ = np.append(DNNLO450_input[0:4],vec_input[4:15] * (DNNLO450_uper_bound[4:15] - DNNLO450_lower_bound[4:15]) + DNNLO450_lower_bound[4:15])
        if ( (self.loop % 5) == 0 ):
            with open('./pounders.out','a') as f_3:
                f_3.write(str(new_parameter_)+'\n')
                f_3.write('nucleon_scattering_total_loss = '+str(total_loss)+'    ')
                f_3.write('pnm_energy = '+ str(pnm_energy)+'\n')
                f_3.write('saturation_dens = '+ str(saturation_dens)+'    ')
                f_3.write('saturation_energy = '+ str(saturation_energy)+'\n')
                f_3.write('H2_BE = '+ str(H2_theo_BE)+'    ')
                f_3.write('H2_R = '+ str(H2_theo_R)+'    ')
                f_3.write('H2_Q = '+ str (H2_quad_moment)+'    ')
                f_3.write('H2_D = '+ str (H2_D_state_probability)+'\n')
                f_3.write('H3_BE = '+ str(H3_theo_BE)+'    ')
                f_3.write('H3_R = '+ str(H3_theo_R)+'    ')
                f_3.write('He3_BE = '+ str(He3_theo_BE)+'    ')
                f_3.write('He3_R = '+ str(He3_theo_R)+'\n')
                f_3.write('He4_BE = '+ str(He4_theo_BE)+'    ')
                f_3.write('He4_R = '+ str(He4_theo_R)+'\n')
                f_3.write('K0 = '+ str(K0)+'    ')
                f_3.write('ddf = '+ str(ddf)+'\n')
                f_3.write('###############################################################################\n')
        self.loop = self.loop + 1
        self.pnm  = pnm_energy
        self.snm_dens = saturation_dens
        self.snm  = saturation_energy
        #FF = F.getArray()
        #print ('F ='+str(F.array))
        #print ('vec_input ='+str(X.array))
        #F.array = y - np.exp(-b1*x)/(b2 + b3*x)
        #F.array = pow((b1 - 3.1),2)+pow((b2-2.7),2)+pow((b3-3.4),2)
        #print (sys.getsizeof(F[0]))
 
    def plotSolution(self, X , F):
#        try:
#            from matplotlib import pylab
#        except ImportError:
#            return
        #b1, b2, b3 = X.array
        #print (sys.getsizeof(b1))
        XX = X.getArray()
        #print ('x_out='+str(XX))
        FF = F.getArray()
        pnm = self.pnm
        snm_dens = self.snm_dens
        snm = self.snm
        #pnm_residual = FF[scattering_data_num]
        #snm_dens_residual = FF
        #print ('total_loss='+str(pnm_residual))
        FF = np.power(FF,2)
        total_loss = np.sum(FF[0:scattering_data_num - 1 ])
        #print ('subloop = '+ str(self.loop))
        #print ('total_loss='+str(FF))
        #print ('total_loss='+str(total_loss))
        #print ('total_loss='+str(len(FF)))
        return X.array,total_loss,pnm,snm_dens,snm
        #x, y = self.x, self.y
        #u = np.linspace(x.min(), x.max(), 100)
        #v = np.exp(-b1*u)/(b2+b3*u)
        #pylab.plot(x, y, 'ro')
        #pylab.plot(u, v, 'b-')
        #pylab.show()

#OptDB = PETSc.Options()

def iterate_pounders():
    loop = 0
    x_initial = np.zeros(13)
    x_in      = np.zeros(13)
    x_out     = np.zeros(13)
#    x_initial = (DNNLO450_input-DNNLO450_lower_bound)/(DNNLO450_uper_bound-DNNLO450_lower_bound)
    x_initial = (DNNLO450_input[4:17]-DNNLO450_lower_bound[4:17])/(DNNLO450_uper_bound[4:17]-DNNLO450_lower_bound[4:17]) * relative_step_size
#    x_initial = DNNLO450_input[4:15]
    #while ( theta > 0.001):DNNLO450_uper_bound-DNNLO450_lower_bound 
    for i in range(pounder_iter_step):
        #x_initial[0] = -3
        #x_initial[1] = 4
        #x_initial[2] = 6
        if (loop == 0):
            x_in = x_initial
        else:
            x_in = x_out

        user = pounders_optimization()
        x, f = user.createVecs()
        x.setFromOptions()
        f.setFromOptions()
        
        tao = PETSc.TAO().create(PETSc.COMM_SELF)
        tao.setType(PETSc.TAO.Type.POUNDERS)
        tao.setSeparableObjective(user.formObjective, f)
        tao.setFromOptions()
        
        x.setValues(range(13),x_in)
        user.formInitialGuess(x)
#        tao.setTolerances(grtol=0.000000001)
        tao.solve(x)
        
        #plot = OptDB.getBool('plot', False)
        #if plot: user.plotSolution(x)
        x_out,total_loss,pnm,snm_dens,snm = user.plotSolution(x,f)
        #print('x_out ')
#        new_parameter = x_out * (DNNLO450_uper_bound - DNNLO450_lower_bound) + DNNLO450_lower_bound
        new_parameter = np.append(DNNLO450_input[0:4],x_out / relative_step_size * (DNNLO450_uper_bound[4:17] - DNNLO450_lower_bound[4:17]) + DNNLO450_lower_bound[4:17])
        loop = loop +1 
        x.destroy()
        f.destroy()
        tao.destroy()

        with open('./pounders.out','a') as f_3:
            f_3.write('###############################################################################\n')
            f_3.write('###############################################################################\n')
            f_3.write('###############################################################################\n')
            f_3.write('loop'+ str(loop-1)+'\n')
            f_3.write('new_parameter = '+str(new_parameter)+'\n')
            f_3.write('nucleon_scattering_total_loss = '+str(total_loss)+'    ')
            f_3.write('nucleon_scattering_average_residual = '+str(pow(total_loss/scattering_data_num,0.5))+'\n')
            f_3.write('pnm = '+ str(pnm))
            f_3.write('  snm_dens = '+ str(snm_dens))
            f_3.write('  snm = '+ str(snm)+'\n')
            f_3.write('###############################################################################\n')
            f_3.write('###############################################################################\n')
            f_3.write('###############################################################################\n')


######################################################
######################################################
##### Setup every thing
######################################################
######################################################
DNNLO450_input = np.array([-0.74, #ci
            -0.49,
            -0.65,
             0.96,
            -0.33813946528363, #Ct_1S0np
            -0.33802308849055, #Ct_1S0nn
            -0.33713665635790, #Ct_1S0pp 
            -0.22931007944035, #Ct_3S1(pp,nn,np)
             2.47658908242147, #C_1S0
             0.64555041107198, #C_3P0
            -1.02235931835913, #C_3P1
            -0.87020321739728, #C_3P2
            -0.02854100307153, #C_1P1
             0.69595320984261, #C_3S1
             0.35832984489387, #C_3S1-3D1
             0.79,             #cD
             0.017])           #cE
DNNLO450_uper_bound = np.array([-0.72,
                                -0.32, 
                                -0.43,
                                 1.07,
                                 0.4,
                                 0.4,
                                 0.4,
                                 0.4,
                                 3,
                                 3,
                                 3,
                                 3,
                                 3,
                                 3,
                                 3,
                                 3,
                                 1])


DNNLO450_lower_bound  = np.array([-0.76,
                                -0.66,
                                -0.87,
                                 0.85,
                                -0.4,
                                -0.4,
                                -0.4,
                                -0.4,
                                -3, 
                                -3, 
                                -3, 
                                -3, 
                                -3, 
                                -3, 
                                -3,
                                -3,
                                -1]) 

pounder_exe_dir    = '../ns_pounders.exe'
ini_file_dir       = 'test.ini'
output_file_dir    = 'aa.out'
residual_data      = 'residual_data.txt'
#nucl_matt_in_dir   = './nuclear_matter/ccm_in'
#nucl_matt_out_dir  = './nuclear_matter/pnm_rho_0.16.out'
nucl_matt_exe      = './nuclear_matter/prog_ccm.exe'
theo_line          = 5
exp_line           = 7
exp_error_line     = 9
energy_line        = 1
E_max              = 200
scattering_data_num= 0
pounder_iter_step  = 10
cD                 = 0
cE                 = 0
relative_step_size = 5
#print (raw_data)
#print (pnm_energy)

with open(residual_data,'r') as f:
    count = len(open(residual_data,'rU').readlines())
    data = f.readlines() 
    loop2 = 0 
    loop1 = 0 
    wtf = re.match('#', 'abc',flags=0)
    while loop1 < count :
        if ( re.match('#', data[loop1],flags=0) == wtf):
            temp_2 = data[loop1][6:]
            temp_1 = re.findall(r"[-+]?\d+\.?\d*",temp_2)
           # print ( 'energy = '+str(temp_1[energy_line]))# * pow(10,int(temp_1[energy_line+1])) )
           # print ( 'order = '+str(pow(10,int(temp_1[energy_line+1]))))
           # print ( 'all = '+str(float(temp_1[energy_line])*pow(10,int(temp_1[energy_line+1]))))
            if ( float(temp_1[energy_line])*pow(10,int(temp_1[energy_line+1]))  <= E_max):
                if (float(temp_1[theo_line])!=0):
                    if ((float(temp_1[0]) < 1000) or  (float(temp_1[0]) == 300002)):
                        loop2 = loop2 + 1 
        loop1 = loop1 + 1
scattering_data_num = loop2 
print ('scattering_data_num = '+str(scattering_data_num))
#pnm_energy_per_nucleon, saturation_dens, saturation_energy = nuclear_matter(DNNLO450_input)
#print('pnm=%f, snm_dens=%f, snm_energy=%f' % (pnm_energy_per_nucleon, saturation_dens, saturation_energy))
start_from_break_point('./optimize_result/pounders_new_constrain_5.out',355)
#start_from_break_point('setup_interaction.out',1)
nucleon_scattering(DNNLO450_input)
#nuclear_matter(DNNLO450_input)
#start_from_break_point('pounders.out',427)
#print (DNNLO450_input)
#iterate_pounders()



