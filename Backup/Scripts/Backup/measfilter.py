# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 18:19:38 2020

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


def param_record(backend, itr = 32, shots = 8192, if_write = True,file_address = ''):
    """Write backend property into an array of dict and save as csv if permissible.

    Args:
      backend:
        backend from provider.get_backend().
      itr:
        number of iterations of job submission.
      shots:
        number of shots per each job submission.
      if_write:
        True if save the properties as a csv file.
      file_address:
        file address, string ends with '/' if not empty

    Returns:
      An array of dicts. Each dict records all characterization of one qubit.
    """
    prop_dict = backend.properties().to_dict()
    nQubits = len(prop_dict['qubits'])
    backend_name = prop_dict['backend_name']
    
    target_qubits = range(nQubits)
    allParam = np.array([])
    for target_qubit in target_qubits:
        params = {
            'qubit':target_qubit,
            'update_date':prop_dict['last_update_date'],
            'T1':prop_dict['qubits'][target_qubit][0]['value'],
            'T2':prop_dict['qubits'][target_qubit][1]['value'],
            'freq':prop_dict['qubits'][target_qubit][2]['value'],
            'readout_err':prop_dict['qubits'][target_qubit][3]['value'],
            'pm0p1':prop_dict['qubits'][target_qubit][4]['value'],
            'pm1p0':prop_dict['qubits'][target_qubit][5]['value'],
            'id_error':prop_dict['gates'][4*target_qubit]['parameters'][0]['value'],
            'id_len':prop_dict['gates'][4*target_qubit]['parameters'][1]['value'],
            'u1_error':prop_dict['gates'][4*target_qubit+1]['parameters'][0]['value'],
            'u1_len':prop_dict['gates'][4*target_qubit+1]['parameters'][1]['value'],
            'u2_error':prop_dict['gates'][4*target_qubit+2]['parameters'][0]['value'],
            'u2_len':prop_dict['gates'][4*target_qubit+2]['parameters'][1]['value'],    
            'u3_error':prop_dict['gates'][4*target_qubit+3]['parameters'][0]['value'],
            'u3_len':prop_dict['gates'][4*target_qubit+3]['parameters'][1]['value'],
            'itr':itr,
            'shots':shots,
        }
        allParam = np.append(allParam,params)
    
    if if_write:
        with open(file_address + 'Params.csv', mode='w', newline='') as sgm:
            param_writer = csv.writer(sgm, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for pa in allParam:
                for key, val in pa.items():
                    param_writer.writerow([key,val])
                param_writer.writerow(['End'])
                
    return allParam


def meas_circ(nQubits,backend,itr = 32):
    """Generate circuit for inferring measurement error.

    Args:
      nQubits:
        int. Number of qubits.
      backend:
        backend from provider.get_backend().
      itr:
        int. Number of data points

    Returns:
      A list of circuits with name 'itr0', 'itr1',...
    """
    circ = QuantumCircuit(nQubits,nQubits)
    for i in range(nQubits):
        circ.h(i)
    circ.measure(range(nQubits),range(nQubits))
    circ_trans = transpile(circ,backend,initial_layout=range(nQubits))
    
    
    circs = []
    for i in range(itr):
        circs.append(circ_trans.copy('itr'+str(i)))
    return circs



def collect_filter_data(backend, itr = 32, shots = 8192, if_monitor_job = True, if_write = True, file_address = ''):
    """Collect data for constructing error filter.

    Args:
      backend:
        backend from provider.get_backend().
      itr:
        number of iterations of job submission.
      shots:
        number of shots per each job submission.
      if_monitor_job:
        if show information for job progress.
      if_write:
        True if save the properties as a csv file.
      file_address:
        file address, string ends with '/' if not empty

    Returns:
      An array of bit strings which is the output from circuits.
    """
    param_record(backend, itr, shots, if_write, file_address)
    
    nQubits = len(backend.properties().to_dict()['qubits'])
    readout_m0 = np.array([])
    circ = meas_circ(nQubits,backend,itr = itr)
    
    # Excute jobs
    job_m0 = execute(circ,backend = backend,shots = shots,memory=True)
    if if_monitor_job:
        job_monitor(job_m0)
        
    # Record bit string
    m0_res = job_m0.result()
    for i in range(itr):
        readout_m0 = np.append(readout_m0,m0_res.get_memory(experiment = ('itr' + str(i))))
        
    if if_write:
        with open(file_address + 'Filter_data.csv', mode='w') as sgr:
            read_writer = csv.writer(sgr, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            read_writer.writerow(readout_m0)  
        
    return readout_m0



def read_params(file_address = ''):
    """Read out backend properties from csv file generated by param_record().

    Args:
      file_address:
        file address, string ends with '/' if not empty

    Returns:
      An array of dicts. Each dict records all characterization of one qubit.
    """
    textKeys = ['name','update_date','qubit']
    intKeys = ['itr','shots']
    # Read Parameters
    with open(file_address + 'Params.csv', mode='r') as sgm:
        reader = csv.reader(sgm)
        params = np.array([])
        singleQubit = {}
        first = True
        for row in reader:   
            if row[0] == 'End':
                params = np.append(params,singleQubit)
                singleQubit = {}
            else:
                singleQubit[row[0]] = row[1]
                
    # Convert corresponding terms into floats or ints
    for qubit in params:
        for key in qubit.keys():
            if key not in textKeys:
                qubit[key] = float(qubit[key])   
            if key in intKeys:
                qubit[key] = int(qubit[key]) 
                
    return params

def read_filter_data(file_address = ''):
    """Read out bit string data from csv file generated by collect_filter_data().

    Args:
      file_address:
        file address, string ends with '/' if not empty

    Returns:
      An array of bit strings.
    """
    with open(file_address + 'Filter_data.csv', mode='r') as measfile:
        reader = csv.reader(measfile)    
        cali01 = np.asarray([row for row in reader][0])
        
    return cali01



def tnorm01(center, sd, size = 1):
    upper = 1
    lower = 0
    a, b = (lower - center) / sd, (upper - center) / sd
    return ss.truncnorm.rvs(a, b, size=size) * sd + center


def find_mode(data):
    """Find the mode through Gaussian KDE.

    Args:
      data:
        an array of floats

    Returns:
      the mode.
    """
    kde = ss.gaussian_kde(data)
    line = np.linspace(min(data),max(data),10000)
    return line[np.argmax(kde(line))]


def closest_mode(post_lambdas):    
    # p0m0 = post_lambdas[:,0]
    # p1m1 = post_lambdas[:,1]
    # p0m0_mode = find_mode(p0m0)
    # p1m1_mode = find_mode(p1m1)
    
    mode_lam = []
    for j in range(post_lambdas.shape[1]):
        mode_lam.append(find_mode(post_lambdas[:,j]))
    
    sol = np.array([])
    smallest_norm = nl.norm(post_lambdas[0])
    mode_lam = np.array(mode_lam)
    for lam in post_lambdas:
        norm_diff = nl.norm(lam - mode_lam)
        if norm_diff < smallest_norm:
            smallest_norm = norm_diff
            sol = lam
    return sol

def closest_average(post_lambdas):
    sol = np.array([])
    smallest_norm = nl.norm(post_lambdas[0])

    ave_lam = np.mean(post_lambdas,axis = 0)

    for lam in post_lambdas:
        norm_diff = nl.norm(lam - ave_lam)

        if norm_diff < smallest_norm:
            smallest_norm = norm_diff
            sol = lam
    return sol

def dictToVec(nQubits,counts):
    vec = np.zeros(2**nQubits)
    form = "{0:0"+str(nQubits)+"b}"
    for i in range(2**nQubits):
        key = form.format(i)
        if key in counts.keys():
            vec[i] = int(counts[key])
        else:
            vec[i] = 0
    return vec

def dictToVec_inv(nQubits,counts):
    vec = np.zeros(2**nQubits)
    form = "{0:0"+str(nQubits)+"b}"
    for i in range(2**nQubits):
        key = form.format(i)[::-1]
        if key in counts.keys():
            vec[i] = int(counts[key])
        else:
            vec[i] = 0
    return vec

def vecToDict(nQubits,shots,vec):
    counts = {}
    form = "{0:0"+str(nQubits)+"b}"
    for i in range(2**nQubits):
        key = form.format(i)
        counts[key] = int(vec[i]*shots)
    return counts

def vecToDict_inv(nQubits,shots,vec):
    counts = {}
    form = "{0:0"+str(nQubits)+"b}"
    for i in range(2**nQubits):
        key = form.format(i)[::-1]
        counts[key] = int(vec[i]*shots)
    return counts

# Functions
def getData0(data,num_group,interested_qubit):
    prob0 = np.zeros(num_group)
    groups = np.split(data, num_group)
    for i in range(num_group):
        count = 0
        for d in groups[i]:
            d_rev = d[::-1]
            if d_rev[interested_qubit] == '0':
                count += 1

        prob0[i] = count/groups[i].size
    return(prob0)

def errMitMat(lambdas_sample):
    # Compute the matrix A from
    # Ax = b, where A is the error mitigation matrix,
    # x is the number appearence of a basis in theory
    # b is the number appearence of a basis in practice with noise
    #
    # Input; lambdas_sample - np.array, array of (1 - error rate) whose length is number of qubits
    # Output; A - np.ndarray, as described above
    pm0p0 = lambdas_sample[0]
    pm1p1 = lambdas_sample[1]
    # Initialize the matrix
    A = np.array([[pm0p0, 1-pm1p1],[1-pm0p0,pm1p1]])
    return(A)

def QoI(prior_lambdas):
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
    qs = np.array([])

    # Smiluate measurement error, assume independence
    for i in range(shape[0]):
        M_ideal = np.ones(2**nQubit)/2**nQubit

        A = errMitMat(prior_lambdas[i])
        M_noisy = np.dot(A, M_ideal)

        # Only record interested qubits
        qs = np.append(qs,M_noisy[0])
    return qs

def findM(qs_ker,d_ker,qs):
    # Function for finding the M, the largest r(Q(lambda)) over all lambda in Algorithm 2 of https://doi.org/10.1137/16M1087229
    # For now we just use the largest ratio r from all given P samples
    # Inputs:
    # qs_ker - gaussian_kde object in scipy.stats, the Q_D^{Q(prior)}(q) in Algorithm 1 of https://doi.org/10.1137/16M1087229
    # d_ker - gaussian_kde object in scipy.stats, pi_D^{obs}(q) in A997 of https://doi.org/10.1137/16M1087229
    # qs - np array of ints, samples generated from some given lambdas
    #
    # Outpus
    # M - the largest r(Q(lambda)) over all lambda
    # index - corresponding index 

    M = -1 # probablities cannot be negative, so -1 is small enough
    index = -1
    for i in range(qs.size):
        if qs_ker(qs[i]) > 0: 
            if M <= d_ker(qs[i])/qs_ker(qs[i]):
                M = d_ker(qs[i])/qs_ker(qs[i])
                index = i
    return M,index

def find_least_norm(nQubits, ptilde):
    # Formulation
    Q = 2*matrix(np.identity(2**nQubits))
    p = -2*matrix(ptilde)

    G = -matrix(np.identity(2**nQubits))
    h = matrix(np.zeros(2**nQubits))

    A = matrix(np.ones(2**nQubits), (1,2**nQubits))
    b = matrix(1.0)

    solvers.options['show_progress'] = False
    sol=solvers.qp(Q, p, G, h, A, b)
    return sol['status'],sol['x']

def output(d, interested_qubit, M, params, prior_sd, seed = 127, show_denoised = False, file_address = ''):
    # Algorithm 1 of https://doi.org/10.1137/16M1087229
    np.random.seed(seed)
    num_lambdas = 2
    # Get distribution of data (Gaussian KDE)
    d_ker = ss.gaussian_kde(d) # i.e., pi_D^{obs}(q), q = Q(lambda)
    average_lambdas = np.array([1 - params[interested_qubit]['pm1p0'],1 - params[interested_qubit]['pm0p1']])
    
    # Compute distribution of Pr(meas. 0) from Qiskit results
    given_errmat = errMitMat(average_lambdas)
    qiskit_p0 = np.empty(len(d))
    for i in range(len(d)):
        single_res = nl.solve(given_errmat,[d[i],1 - d[i]])
        qiskit_p0[i] = single_res[0]
    qiskit_ker = ss.gaussian_kde(qiskit_p0)
    
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
            one_sample[j] = tnorm01(average_lambdas[j],prior_sd)
            # while one_sample[j]<= 0 or one_sample[j] > 1:
            #     one_sample[j] = np.random.normal(average_lambdas[j],prior_sd,1)
        prior_lambdas[i] = one_sample

    # Produce prior QoI
    qs = QoI(prior_lambdas)
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
    post_qs = QoI(post_lambdas)
    post_ker = ss.gaussian_kde(post_qs)
    
    xs = np.linspace(0,1,1000)
    xsd = np.linspace(0.3,0.7,1000)

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
    print('0 to 1: KL-Div(qiskit,pi_D^obs) = %6g'%(ss.entropy(qiskit_ker(xs),d_ker(xs))))
    print('0 to 1: KL-Div(pi_D^obs,qiskit) = %6g'%(ss.entropy(d_ker(xs),qiskit_ker(xs))))
    
    print('Post and Data: Sum of Differences ',np.sum(np.abs(post_ker(xs) - d_ker(xs))/1000))
    print('Qisk and Data: Sum of Differences ',np.sum(np.abs(qiskit_ker(xs) - d_ker(xs))/1000))

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
    
    if show_denoised:
        res_proc = np.array([])
        for lam in post_lambdas:
            M_sub = errMitMat(lam)
            res_proc = np.append(res_proc,np.array([nl.solve(M_sub,[d[ind],1 - d[ind]])[0] for ind in range(len(d))]))
        proc_ker = ss.gaussian_kde(res_proc)
        
        # Denoised by Qiskit Parameters
        M_qsub = errMitMat(np.array([1 - params[interested_qubit]['pm1p0'],1 - params[interested_qubit]['pm0p1']]))
        res_qisk = np.array([nl.solve(M_qsub,[d[ind],1 - d[ind]])[0] for ind in range(len(d))])
        qisk_ker = ss.gaussian_kde(res_qisk)
        
        #plt.plot(xsd,d_ker(xsd),color='Red',linestyle='dashed',label = '(Noisy) Data')
        plt.plot(xsd,proc_ker(xsd),color='Blue',label = 'By Posteriors')
        plt.plot(xsd,qisk_ker(xsd),color='green',label = 'By Provided Params')
        plt.axvline(x=0.5,color='black',label = 'Ideal Value')
        plt.xlabel('Pr(Meas. 0)')
        plt.ylabel('Density')
        #plt.title('Denoised Pr(Meas. 0), Qubit %g'%interested_qubit)
        plt.legend()
        plt.savefig(file_address + 'DQoI-Qubit%g.jpg'%interested_qubit)
        plt.show()
    return prior_lambdas,post_lambdas


class MeasFilter:
    """Measurement error filter.

    Attributes:
        qubit_order: an array of ints, using order[LastQubit, ..., FirstQubit].
        file_address: the address for saving Params.csv and Filter_data.csv. 
                      End with '/' if not empty.
    """
    def __init__(self, qubit_order, file_address = ''):        
        self.file_address = file_address
        self.qubit_order = qubit_order
        self.prior = {}
        self.post = {}
        self.mat_mean = None
        self.mat_mode = None
        
    def create_filter_mat(self):
        # Create filter matrix with Posterior mean
        first = True
        for q in self.qubit_order:
            if first:
                postLam_mean = closest_average(self.post['Qubit'+str(q)])
                Mx = errMitMat(postLam_mean)
                first = False
            else:
                postLam_mean = closest_average(self.post['Qubit'+str(q)])
                Msub = errMitMat(postLam_mean)
                Mx = np.kron(Mx,Msub)
        self.mat_mean = Mx

        # Create filter matrix with Posterior mode
        first = True
        for q in self.qubit_order:
            if first:
                postLam_mean = closest_mode(self.post['Qubit'+str(q)])
                Mx = errMitMat(postLam_mean)
                first = False
            else:
                postLam_mean = closest_mode(self.post['Qubit'+str(q)])
                Msub = errMitMat(postLam_mean)
                Mx = np.kron(Mx,Msub)
        self.mat_mode = Mx
    
    def inference(self, nPrior = 40000, Priod_sd = 0.1, 
                  seed = 227, shots_per_point = 1024, show_denoised = False):
        
        self.data = read_filter_data(self.file_address)
        self.params = read_params(self.file_address)
        
        itr = self.params[0]['itr']
        shots = self.params[0]['shots']
        info = {}
        for i in self.qubit_order:
            print('Qubit %d'%(i))
            d = getData0(self.data,int(itr*shots/ shots_per_point),i)
            prior_lambdas,post_lambdas = output(d,i,nPrior,self.params,Priod_sd,
                                                seed = seed, show_denoised = show_denoised,
                                                file_address = self.file_address)   
            self.prior['Qubit'+str(i)] = prior_lambdas
            self.post['Qubit'+str(i)] = post_lambdas
        self.create_filter_mat()
            
    def post_from_file(self):
        for i in self.qubit_order:
            post_lambdas = pd.read_csv(self.file_address + 'Post_Qubit{}.csv'.format(i),header = None).to_numpy()
            self.post['Qubit'+str(i)] = post_lambdas
        self.create_filter_mat()
            
    def mean(self):
        res = {}
        for q in self.qubit_order:
            res['Qubit'+str(q)] = closest_average(self.post['Qubit'+str(q)])
        return res
        
    def mode(self):
        res = {}
        for q in self.qubit_order:
            res['Qubit'+str(q)] = closest_mode(self.post['Qubit'+str(q)])
        return res
    
    def filter_mean(self,counts):
        shots = 0
        for key in counts:
            shots += counts[key]
            
        real_vec = dictToVec(len(self.qubit_order),counts)/shots
        proc_status,proc_vec = find_least_norm(len(self.qubit_order),nl.solve(self.mat_mean,real_vec))
        if proc_status != 'optimal':
            raise Exception('Sorry, filtering has failed')
        proc_counts = vecToDict(len(self.qubit_order),shots,proc_vec)
        return proc_counts
    
    def filter_mode(self,counts):            
        shots = 0
        for key in counts:
            shots += counts[key]
            
        real_vec = dictToVec(len(self.qubit_order),counts)/shots
        proc_status,proc_vec = find_least_norm(len(self.qubit_order),nl.solve(self.mat_mode,real_vec))
        if proc_status != 'optimal':
            raise Exception('Sorry, filtering has failed')
        proc_counts = vecToDict(len(self.qubit_order),shots,proc_vec)
        return proc_counts

    