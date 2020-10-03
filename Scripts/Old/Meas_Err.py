# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 15:15:02 2020

@author: Muqing
"""

# Numerical/Stats pack
import csv
import pandas as pd
import numpy as np
import scipy.stats as ss
import numpy.linalg as nl


# We import plotting tools 
import matplotlib.pyplot as plt 
from   matplotlib import cm
from   matplotlib.ticker import LinearLocator, FormatStrFormatter

# For optimization
from cvxopt import matrix, solvers


pd.set_option('precision', 6)


def closed_mode(post_lambdas):
    sol = np.array([])
    smallest_norm = nl.norm(post_lambdas[0])
    
    mode_lam = ss.mode(post_lambdas,axis = 0)[0][0]
    
    for lam in post_lambdas:
        norm_diff = nl.norm(lam - mode_lam)
        
        if norm_diff < smallest_norm:
            smallest_norm = norm_diff
            sol = lam
            
    return sol




def closed_average(post_lambdas):
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

def vecToDict(nQubits,shots,vec):
    counts = {}
    form = "{0:0"+str(nQubits)+"b}"
    for i in range(2**nQubits):
        key = form.format(i)
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


def output(d, num_qubits,num_lambdas,interested_qubit,M,params,prior_sd,itr,seed = 127,show_denoised = False):
    # Algorithm 1 of https://doi.org/10.1137/16M1087229
    np.random.seed(seed)
    
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
            while one_sample[j]<= 0 or one_sample[j] > 1:
                one_sample[j] = np.random.normal(average_lambdas[j],prior_sd,1)
        prior_lambdas[i] = one_sample


    # Produce prior QoI
    #print(prior_lambdas)
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
    xsd = np.linspace(0.4,0.7,1000)

    I = 0
    for i in range(xs.size-1):
        q = xs[i]
        if qs_ker(q) > 0:
            r = d_ker(q)/qs_ker(q)
            I += r * qs_ker.pdf(q) * (xs[i+1] - xs[i]) # Just Riemann Sum
            
    print('Accepted Number N: %.d, %.3f'%(post_lambdas.shape[0],post_lambdas.shape[0]/M))
    print('I(pi^post_Lambda) = %.5g'%(I)) # Need to close to 1
    print('Posterior Lambda Mean',np.mean(post_lambdas,axis = 0))
    
    
    print('0 to 1: KL-Div(pi_D^Q(post),pi_D^obs) = %6g'%(ss.entropy(post_ker(xs),d_ker(xs))))
    print('0 to 1: KL-Div(pi_D^obs,pi_D^Q(post)) = %6g'%(ss.entropy(d_ker(xs),post_ker(xs))))
    print('0 to 1: KL-Div(qiskit,pi_D^obs) = %6g'%(ss.entropy(qiskit_ker(xs),d_ker(xs))))
    print('0 to 1: KL-Div(pi_D^obs,qiskit) = %6g'%(ss.entropy(d_ker(xs),qiskit_ker(xs))))
    # print('min(data,post_QoI) to 1: KL-Div(pi_D^Q(post),pi_D^obs) = %6g'%(ss.entropy(post_ker(xsd),d_ker(xsd))))
    # print('min(data,post_QoI) to 1: KL-Div(pi_D^obs,pi_D^Q(post)) = %6g'%(ss.entropy(d_ker(xsd),post_ker(xsd))))
    
    
    print('Post and Data: Sum of Differences ',np.sum(np.abs(post_ker(xs) - d_ker(xs))/1000))
    print('Qisk and Data: Sum of Differences ',np.sum(np.abs(qiskit_ker(xs) - d_ker(xs))/1000))
    
    # Plots
    plt.plot(xsd,d_ker(xsd),color='Red',linestyle='dashed', linewidth=3,label = 'Observed QoI')
    plt.plot(xsd,post_ker(xsd),color='Blue',label = 'QoI by Posterior')
    #plt.plot(xsd,qs_ker(xsd),color='lightgreen',label = 'Prior')
    #plt.title('Noisy Pr(Meas. 0), Qubit %g'%interested_qubit)
    plt.xlabel('Pr(Meas. 0)')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('QoI-Qubit%g.jpg'%interested_qubit)
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
        plt.axvline(x=0.5,color='Red',label = 'Ideal Value')
        plt.xlabel('Pr(Meas. 0)')
        plt.ylabel('Density')
        #plt.title('Denoised Pr(Meas. 0), Qubit %g'%interested_qubit)
        plt.legend()
        plt.savefig('DQoI-Qubit%g.jpg'%interested_qubit)
        plt.show()

#     plt.scatter(post_lambdas[:,0],post_lambdas[:,1],color='Blue', s=0.1,label = 'Posterior Lambda')
#     plt.axvline(x=average_lambdas[0],color='Red',label = 'Given $P(Meas. 0| Prep. 0)$')
#     plt.hlines(average_lambdas[1],xmin = min(post_lambdas[:,0]), xmax = 1,color='black',label = 'Given $P(Meas. 1| Prep. 1)$')
#     plt.xlabel('Pr(Meas. 0| Prep. 0)')
#     plt.ylabel('Pr(Meas. 1| Prep. 1)')
#     plt.title('Measurement Error Parameters, Qubit %g'%interested_qubit)
#     plt.legend()
#     plt.show()
    
    return qs,post_qs,average_lambdas,prior_lambdas,post_lambdas