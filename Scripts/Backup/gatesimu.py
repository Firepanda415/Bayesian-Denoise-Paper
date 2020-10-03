# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 17:10:41 2020

@author: Muqing Zheng
"""

# Qiskit
from qiskit import QuantumCircuit,transpile,execute
from qiskit.tools.monitor import job_monitor
# Numerical/Stats pack
import csv
import pandas as pd
import numpy as np
import scipy.stats as ss
import numpy.linalg as nl
import matplotlib.pyplot as plt 
# For optimization
from cvxopt import matrix, solvers

from measfilter import *


######################## For Parameter Characterzation ########################
def gate_circ(nGates, gate_type, interested_qubit, itr):
    circ = QuantumCircuit(5,5)
    for _ in range(nGates):
        if gate_type == 'X':
            circ.x(interested_qubit)
        elif gate_type == 'Y':
            circ.y(interested_qubit)
        elif gate_type == 'Z':
            circ.z(interested_qubit)
        else:
            raise Exception('Choose gate_type from X, Y, Z')
        circ.barrier(interested_qubit)
    circ.measure([interested_qubit],[interested_qubit])
    print('Circ depth is ', circ.depth())
    circs = []
    for i in range(itr):
        circs.append(circ.copy('itr'+str(i)))
    return circs


def Gateexp(nGates, gate_type, interested_qubit, itr, backend, file_address = ''):
    circs = gate_circ(nGates, gate_type, interested_qubit, itr)
    # Run on real device
    job_exp = execute(circs, backend = backend, shots = 8192, memory = True, optimization_level = 0)
    job_monitor(job_exp)
    
    # Record bit string
    exp_results = job_exp.result()
    readout = np.array([])
    for i in range(itr):
        readout = np.append(readout,exp_results.get_memory(experiment = ('itr' + str(i))))
        
    with open(file_address + 'Readout_{}{}Q{}.csv'.format(nGates,gate_type,interested_qubit), mode='w') as sgr:
        read_writer = csv.writer(sgr, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        read_writer.writerow(readout)
        
# used to call QoI
def QoI_gate(prior_lambdas, ideal_p0, gate_num):
    # Function equivalent to Q(lambda) in https://doi.org/10.1137/16M1087229
    # Inputs:
    # prior_lambdas - an np array of arrays. each subarray is an individual prior lambda
    #
    # Outputs:
    # qs - an np array of binaries, each binary represents an output of the circuit
    
    #############################################################################################
    #### Let us first only consider measurement error on measuring |0> in an n qubit machine ####
    #############################################################################################
    shape = prior_lambdas.shape
    nQubit = 1
    
    # Initialize the output array
    qs = np.array([], dtype = np.float64)
    
    # Smiluate measurement error, assume independence
    p0 = ideal_p0
    p1 = 1-p0
    # Compute Fourier coefficient
    phat0 = 1/2 * (p0 * (-1)**(0*0) + p1 * (-1)**(0*1))
    phat1 = 1/2 * (p0 * (-1)**(1*0) + p1 * (-1)**(1*1))
    
    for i in range(shape[0]):
        eps = prior_lambdas[i][2]
        noisy_p0 = ((1-eps)**(0))**gate_num * phat0 * (-1)**(0*0) + ((1-eps)**(1))**gate_num * phat1 * (-1)**(1*0)
        noisy_p1 = ((1-eps)**(0))**gate_num * phat0 * (-1)**(0*1) + ((1-eps)**(1))**gate_num * phat1 * (-1)**(1*1)
        M_ideal = np.array([noisy_p0,noisy_p1])

        A = errMitMat(prior_lambdas[i])
        M_meaerr = np.dot(A, M_ideal)

        # Only record interested qubits
        qs = np.append(qs,M_meaerr[0])
    return qs
        
# Used to call output, delete ideal_p0 parameter
def output_gate(d, interested_qubit, M, params, gate_sd, meas_sd, gate_type, gate_num, seed = 127, file_address = ''):
    # Algorithm 1 of https://doi.org/10.1137/16M1087229
    
    average_lambdas = np.array([1 - params[interested_qubit]['pm1p0'],1 - params[interested_qubit]['pm0p1']])
    if gate_type == 'X':
        average_lambdas = np.append(average_lambdas, 2*params[interested_qubit]['u3_error'])
        if gate_num % 2:
            ideal_p0 = 0
        else:
            ideal_p0 = 1
    elif gate_type == 'Y':
        average_lambdas = np.append(average_lambdas, 2*params[interested_qubit]['u3_error'])
        if gate_num % 2:
            ideal_p0 = 0
        else:
            ideal_p0 = 1
    elif gate_type == 'Z':
        average_lambdas = np.append(average_lambdas, 2*params[interested_qubit]['u1_error'])
        ideal_p0 = 0
    else:
        raise Exception('Choose gate_type from X, Y, Z')

    # write data for standard bayesian inference
    with open(file_address + 'Qubit{}.csv'.format(interested_qubit), mode='w',newline='') as sgr:
        read_writer = csv.writer(sgr, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        read_writer.writerow(['x','y'])
        for i in range(len(d)):
            read_writer.writerow([ideal_p0,d[i]])  
    
    np.random.seed(seed)
    num_lambdas = 3
    # Get distribution of data (Gaussian KDE)
    d_ker = ss.gaussian_kde(d) # i.e., pi_D^{obs}(q), q = Q(lambda)
    
    if average_lambdas[0] == 1 or average_lambdas[0] < 0.7:
        average_lambdas[0] = 0.9
    if average_lambdas[1] == 1 or average_lambdas[1] < 0.7:
        average_lambdas[1] = 0.9
        

    # Sample prior lambdas, assume prior distribution is Normal distribution with mean as the given probality from IBM
    # Absolute value is used here to avoid negative values, so it is little twisted, may consider Gamma Distribution
    prior_lambdas = np.zeros(M*num_lambdas).reshape((M,num_lambdas))
    
    for i in range(M):
        one_sample = np.zeros(num_lambdas)
        for j in range(num_lambdas):
            if j < 2:
                one_sample[j] = tnorm01(average_lambdas[j],meas_sd)
                # while one_sample[j]<= 0 or one_sample[j] > 1:
                #     one_sample[j] = np.random.normal(average_lambdas[j],meas_sd,1)
            else:
                one_sample[j] = tnorm01(average_lambdas[j],gate_sd)
                # while one_sample[j]<= 0 or one_sample[j] > 1:
                #     one_sample[j] = np.random.normal(average_lambdas[j],gate_sd,1)
        prior_lambdas[i] = one_sample

    # Produce prior QoI
    qs = QoI_gate(prior_lambdas, ideal_p0, gate_num)
    #print(qs)
    qs_ker = ss.gaussian_kde(qs) # i.e., pi_D^{Q(prior)}(q), q = Q(lambda)
    
    # Plot and Print
    print('Given Lambdas',average_lambdas)

    # Algorithm 2 of https://doi.org/10.1137/16M1087229

    # Find the max ratio r(Q(lambda)) over all lambdas
    max_r,max_ind = findM(qs_ker,d_ker,qs)
    # Print and Check
    print('Final Accepted Posterior Lambdas')
    print('M: %.6g Index: %.d pi_obs = %.6g pi_Q(prior) = %.6g'%(max_r,max_ind,d_ker(qs[max_ind]),qs_ker(qs[max_ind])))

    post_lambdas = np.array([])
    # Go to Rejection Iteration
    for p in range(M):
        r = d_ker(qs[p])/qs_ker(qs[p])
        eta = r/max_r
        if eta>np.random.uniform(0,1,1):
            post_lambdas = np.append(post_lambdas,prior_lambdas[p])

    post_lambdas = post_lambdas.reshape(int(post_lambdas.size/num_lambdas),num_lambdas) # Reshape since append destory subarrays
    post_qs = QoI_gate(post_lambdas, ideal_p0, gate_num)
    post_ker = ss.gaussian_kde(post_qs)
    
    xs = np.linspace(0, 1, 1000)
    xsd = np.linspace(0.2, 0.8, 1000)

    I = 0
    for i in range(xs.size-1):
        q = xs[i]
        if qs_ker(q) > 0:
            r = d_ker(q)/qs_ker(q)
            I += r * qs_ker.pdf(q) * (xs[i+1] - xs[i]) # Just Riemann Sum
            
    print('Accepted Number N: %.d, %.3f'%(post_lambdas.shape[0],post_lambdas.shape[0]/M))
    print('I(pi^post_Lambda) = %.5g'%(I)) # Need to close to 1
    print('Posterior Lambda Mean',closest_average(post_lambdas))
    print('Posterior Lambda Mode',closest_mode(post_lambdas))
    
    print('0 to 1: KL-Div(pi_D^Q(post),pi_D^obs) = %6g'%(ss.entropy(post_ker(xs),d_ker(xs))))
    print('0 to 1: KL-Div(pi_D^obs,pi_D^Q(post)) = %6g'%(ss.entropy(d_ker(xs),post_ker(xs))))

    with open(file_address + 'Post_Qubit{}.csv'.format(interested_qubit), mode='w', newline='') as sgm:
        writer = csv.writer(sgm, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(post_lambdas.shape[0]):
            writer.writerow(post_lambdas[i])
    
    # Plots
    plt.plot(xsd,d_ker(xsd),color='Red',linestyle='dashed', linewidth=3,label = 'Observed QoI')
    plt.plot(xsd,post_ker(xsd),color='Blue',label = 'QoI by Posterior')
    plt.xlabel('Pr(Meas. 0)')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(file_address + 'QoI-Qubit%g.jpg'%interested_qubit)
    plt.show()
    
    
    eps_ker = ss.gaussian_kde(post_lambdas[:,2])
    eps_line = np.linspace(np.min(post_lambdas, axis = 0)[2], np.max(post_lambdas, axis = 0)[2], 1000)
    plt.plot(eps_line, eps_ker(eps_line),color='Blue')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(-5,1))
    plt.xlabel('$\epsilon_g$')
    plt.ylabel('Density')
    #plt.legend()
    plt.savefig(file_address + 'Eps-Qubit%g.jpg'%interested_qubit)
    plt.show()
    return prior_lambdas,post_lambdas




def read_data(interested_qubit, gate_type, gate_num,  file_address = ''):
    """Read out bit string data from csv file generated by collect_filter_data().

    Args:
      file_address:
        file address, string ends with '/' if not empty

    Returns:
      An array of bit strings.
    """
    with open(file_address + 'Readout_{}{}Q{}.csv'.format(gate_num, gate_type, interested_qubit), mode='r') as measfile:
        reader = csv.reader(measfile)    
        cali01 = np.asarray([row for row in reader][0])
        
    return cali01


def read_post(itr, shots, interested_qubits, gate_type, gate_num, file_address = ''):
    post = {}
    for q in interested_qubits:
        data = read_data(q, gate_type, gate_num, file_address = file_address)
        d = getData0(data, int(itr*shots/1024), q)
        post_lambdas = pd.read_csv(self.file_address + 'Post_Qubit{}.csv'.format(i),header = None).to_numpy()
        post['Qubit'+str(i)] = post_lambdas
    
        # information part
        xs = np.linspace(0, 1, 1000)
        xsd = np.linspace(0.2, 0.8, 1000)
        d_ker = ss.gaussian_kde(d)
        print('Posterior Lambda Mean',closest_average(post_lambdas))
        print('Posterior Lambda Mode',closest_mode(post_lambdas))
        print('0 to 1: KL-Div(pi_D^Q(post),pi_D^obs) = %6g'%(ss.entropy(post_ker(xs),d_ker(xs))))
        print('0 to 1: KL-Div(pi_D^obs,pi_D^Q(post)) = %6g'%(ss.entropy(d_ker(xs),post_ker(xs))))
            
        post_qs = QoI_gate(post_lambdas, ideal_p0, gate_num)
        post_ker = ss.gaussian_kde(post_qs)
        plt.plot(xsd,d_ker(xsd),color='Red',linestyle='dashed', linewidth=3,label = 'Observed QoI')
        plt.plot(xsd,post_ker(xsd),color='Blue',label = 'QoI by Posterior')
        plt.xlabel('Pr(Meas. 0)')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(file_address + 'QoI-Qubit%g.jpg'%interested_qubit)
        plt.show()
        
        eps_ker = ss.gaussian_kde(post_lambdas[:,2])
        eps_line = np.linspace(np.min(post_lambdas, axis = 0)[2], np.max(post_lambdas, axis = 0)[2], 1000)
        plt.plot(eps_line, eps_ker(eps_line),color='Blue')
        plt.ticklabel_format(axis="x", style="sci", scilimits=(-5,1))
        plt.xlabel('$\epsilon_g$')
        plt.ylabel('Density')
        #plt.legend()
        plt.savefig(file_address + 'Eps-Qubit%g.jpg'%interested_qubit)
        plt.show()
    
    return post


def plotComparsion(data,post_lambdas, q, file_address = ''):
    postSB = pd.read_csv(file_address + 'StandPostQubit{}.csv'.format(q)).to_numpy()
    SB = QoI_gate(postSB,1,200)
    OB = QoI_gate(post_lambdas,1,200)
    minSB = min(SB)
    minOB = min(OB)
    maxSB = max(SB)
    maxOB = max(OB)
    line = np.linspace(min(minSB,minOB),max(maxSB,maxOB),1000)
    SBker = ss.gaussian_kde(SB)
    OBker = ss.gaussian_kde(OB)
    dker = ss.gaussian_kde(data)
    plt.plot(line,SBker(line), color='Green', linewidth=2, label = 'Standard')
    plt.plot(line,OBker(line), color='Blue', linewidth=4, label = 'BJW')
    plt.plot(line,dker(line), color='Red',linestyle='dashed', linewidth=6, label = 'Data')
    plt.xlabel('Pr(Meas. 0)')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(file_address + 'SBOB-Qubit%g.jpg'%q)
    plt.show()
    
    
#################### For Error Filtering #########################
def sxpower(s,x):
    total = 0
    length = len(s)
    for i in range(length):
        si = int(s[i])
        xi = int(x[i])
        total += si*xi
    return total

def count1(s):
    count = 0
    for i in range(len(s)):
        if s[i] == '1':
            count += 1
    return count

def gate_matrix(length, eps, m):
    size = 2**length
    mat = np.empty([size,size],dtype = np.float64)
    for row in range(size):
        for col in range(size):
            x = ("{0:0"+str(length)+"b}").format(row)
            s = ("{0:0"+str(length)+"b}").format(col)
            power = sxpower(s,x)
            mat[row,col] = ((-1)**power) * ((1 - eps)**(count1(s)*m))
    return mat

def find_least_norm(ptilde): # only for one-qubit case
    # Formulation
    Q = 2*matrix(np.identity(2))
    p = -2*matrix(ptilde)

    G = matrix(np.array([[0,1],[0,-1]]),(2,2), 'd')
    h = 0.5*matrix(np.ones(2))

    A = matrix(np.array([1,0]),(1,2), 'd')
    b = matrix(0.5)

    solvers.options['show_progress'] = False
    sol=solvers.qp(Q, p, G, h, A, b)
    return sol['status'],sol['x']

def gate_denoise(p0s, lambdas):
    denoised = []
    meas_err_mat = errMitMat([lambdas[0], lambdas[1]])
    M = gate_matrix(1, lambdas[2], m)
    for p0 in p0s:
        ptilde = np.array([p0, 1 - p0])
        gate_ptilde = np.linalg.solve(meas_err_mat,ptilde)
        phat = np.linalg.solve(M,gate_ptilde)
        status, opt_phat = find_least_norm(phat)
        opt_recovered_p0 = opt_phat[0] + opt_phat[1] * (-1) ** (1 * 0) # phat(0) + phat(1)
        opt_recovered_p1 = opt_phat[0] + opt_phat[1] * (-1) ** (1 * 1) # phat(0) - phat(1)
        denoised.append(opt_recovered_p0)
        
    return denoised   
    
    
    
    
    
    
########################## Class for Error Filtering ########################
class GMFilter:
    """Gate and Measurement error filter.

    Attributes:
        qubit_order: an array of ints, using order[LastQubit, ..., FirstQubit].
        file_address: the address for saving Params.csv and Filter_data.csv. 
                      End with '/' if not empty.
    """
    def __init__(self, interested_qubits, gate_type, gate_num, file_address = ''):  
        self.interested_qubits = interested_qubits
        self.file_address = file_address
        self.gate_type = gate_type
        self.gate_num = gate_num
        self.post = {}
        self.modes = None
        self.means = None
        
    def mean(self):
        res = {}
        for q in self.interested_qubits:
            res['Qubit'+str(q)] = closest_average(self.post['Qubit'+str(q)])
        return res
        
    def mode(self):
        res = {}
        for q in self.interested_qubits:
            res['Qubit'+str(q)] = closest_mode(self.post['Qubit'+str(q)])
        return res
    
    def inference(self, nPrior = 40000, meas_sd = 0.1, gate_sd = 0.01,
                  seed = 127, shots_per_point = 1024):
        
        self.data = read_filter_data(self.file_address)
        self.params = read_params(self.file_address)
        
        itr = self.params[0]['itr']
        shots = self.params[0]['shots']
        info = {}
        for i in self.interested_qubits:
            print('Qubit %d'%(i))
            data = read_data(i, self.gate_type, self.gate_num, file_address = file_address)
            d = getData0(data, int(itr*shots/shots_per_point), q)
            _,post_lambdas = output_gate(d, i, nPrior, self.params, 
                                         gate_sd, meas_sd, 
                                         self.gate_type, self.gate_num, 
                                         file_address = file_address)
            self.post['Qubit'+str(i)] = post_lambdas
        self.modes = self.mode()
        self.means = self.mean()
            
    def post_from_file(self):
        for i in self.interested_qubits:
            post_lambdas = pd.read_csv(self.file_address + 'Post_Qubit{}.csv'.format(i),header = None).to_numpy()
            self.post['Qubit'+str(i)] = post_lambdas
        self.modes = self.mode()
        self.means = self.mean()
    
    def filter_mean(self,p0s, qubit_index):
        return gate_denoise(p0s, self.means[str(qubit_index)])
    
    def filter_mode(self,p0s, qubit_index):
        return gate_denoise(p0s, self.modes[str(qubit_index)])