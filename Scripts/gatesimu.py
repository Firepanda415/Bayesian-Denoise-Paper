# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 17:10:41 2020

@author: Muqing Zheng
"""

# Qiskit
from qiskit import QuantumCircuit, transpile, execute
from qiskit.tools.monitor import job_monitor
# Numerical/Stats pack
import csv
import pandas as pd
import numpy as np
import scipy.stats as ss
import numpy.linalg as nl

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] 
# For optimization
from cvxopt import matrix, solvers

from measfilter import *

_widths = {
    # a4paper columnwidth = 426.79135 pt = 5.93 in
    # letterpaper columnwidth = 443.57848 pt = 6.16 in
    'onecolumn': {
        'a4paper' : 5.93,
        'letterpaper' : 6.16
    },
    # a4paper columnwidth = 231.84843 pt = 3.22 in
    # letterpaper columnwidth = 240.24199 pt = 3.34 in
    'twocolumn': {
        'a4paper' : 3.22,
        'letterpaper' : 3.34
    }
}

_wide_widths = {
    # a4paper wide columnwidth = 426.79135 pt = 5.93 in
    # letterpaper wide columnwidth = 443.57848 pt = 6.16 in
    'onecolumn': {
        'a4paper' : 5.93,
        'letterpaper' : 6.16
    },
    # a4paper wide linewidth = 483.69687 pt = 6.72 in
    # letterpaper wide linewidth = 500.48400 pt = 6.95 in
    'twocolumn': {
        'a4paper' : 6.72,
        'letterpaper' : 6.95
    }
}

_fontsizes = {
    10 : {
        'tiny' : 5,
        'scriptsize' : 7,
        'footnotesize' : 8, 
        'small' : 9, 
        'normalsize' : 10,
        'large' : 12, 
        'Large' : 14, 
        'LARGE' : 17,
        'huge' : 20,
        'Huge' : 25
    },
    11 : {
        'tiny' : 6,
        'scriptsize' : 8,
        'footnotesize' : 9, 
        'small' : 10, 
        'normalsize' : 11,
        'large' : 12, 
        'Large' : 14, 
        'LARGE' : 17,
        'huge' :  20,
        'Huge' :  25
    },
    12 : {
        'tiny' : 6,
        'scriptsize' : 8,
        'footnotesize' : 10, 
        'small' : 11, 
        'normalsize' : 12,
        'large' : 14, 
        'Large' : 17, 
        'LARGE' : 20,
        'huge' :  25,
        'Huge' :  25
    }
}

_width         = 1
_wide_width    = 1
_quantumviolet = '#53257F'
_quantumgray   = '#555555'


columns = 'twocolumn'
paper = 'a4paper'
fontsize = 10


plt.rcdefaults()
    
# Seaborn white is a good base style
plt.style.use(['seaborn-white', '../Plots/quantum-plots.mplstyle'])


_width = _widths[columns][paper]

 
_wide_width = _wide_widths[columns][paper]

# Use the default fontsize scaling of LaTeX

fontsizes = _fontsizes[fontsize]

plt.rcParams['axes.labelsize'] = fontsizes['small']
plt.rcParams['axes.titlesize'] = fontsizes['large']
plt.rcParams['xtick.labelsize'] = fontsizes['footnotesize']
plt.rcParams['ytick.labelsize'] = fontsizes['footnotesize']
plt.rcParams['font.size'] = fontsizes['small']

aspect_ratio = 1/1.62
width_ratio = 1.0
wide = False

width = (_wide_width if wide else _width) * width_ratio
height = width * aspect_ratio



######################## For Parameter Characterzation ########################
def gate_circ(nGates, gate_type, interested_qubit, itr):
    """
      Generate circuits for gate error experiment

    Parameters
    ----------
    nGates : int
        number of gates.
    gate_type : String
        Chosen between "X", "Y", and "Z".
    interested_qubit : int
        on which qubit that those gates apply on.
    itr : int
        number of iteration to submit on the same qubit 
        with defacult 8192 shots. (So in total runs itr*8192 times)

    Returns
    -------
    None.

    """
    circ = QuantumCircuit(5, 5)
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
    circ.measure([interested_qubit], [interested_qubit])
    print('Circ depth is ', circ.depth())
    circs = []
    for i in range(itr):
        circs.append(circ.copy('itr' + str(i)))
    return circs


def Gateexp(nGates,
            gate_type,
            interested_qubit,
            itr,
            backend,
            file_address=''):
    """
      Function for collect data for gate error experiment

    Parameters
    ----------
    nGates : int
        number of gates.
    gate_type : String
        Chosen between "X", "Y", and "Z".
    interested_qubit : int
        on which qubit that those gates apply on.
    itr : int
        number of iteration to submit on the same qubit 
        with defacult 8192 shots. (So in total runs itr*8192 times)
    backend: IBMQBackend
        backend from provider.get_backend().
    file_address: string, optional
        The relative file address to save data file. 
        Ends with '/' if not empty
        The default is ''.

    Returns
    -------
    None.

    """

    circs = gate_circ(nGates, gate_type, interested_qubit, itr)
    # Run on real device
    job_exp = execute(circs,
                      backend=backend,
                      shots=8192,
                      memory=True,
                      optimization_level=0)
    job_monitor(job_exp)

    # Record bit string
    exp_results = job_exp.result()
    readout = np.array([])
    for i in range(itr):
        readout = np.append(
            readout, exp_results.get_memory(experiment=('itr' + str(i))))

    with open(
            file_address +
            'Readout_{}{}Q{}.csv'.format(nGates, gate_type, interested_qubit),
            mode='w') as sgr:
        read_writer = csv.writer(sgr,
                                 delimiter=',',
                                 quotechar='"',
                                 quoting=csv.QUOTE_MINIMAL)
        read_writer.writerow(readout)


# used to call QoI
def QoI_gate(prior_lambdas, ideal_p0, gate_num):
    """
    Function equivalent to Q(lambda) in https://doi.org/10.1137/16M1087229

    Parameters
    ----------
    prior_lambdas : numpy array
        each subarray is an individual prior lambda.
    ideal_p0: float
        probability of measuring 0 without any noise.
    gate_num: int
        number of gates in the circuit. 
        Should be the same as nGates in Gateexp().

    Returns
    -------
    qs : numpy array
        QoI's. Here they are the probability of measuring 0 with each given
        prior lambdas in prior_lambdas.

    """
    shape = prior_lambdas.shape
    nQubit = 1

    # Initialize the output array
    qs = np.array([], dtype=np.float64)

    # Smiluate measurement error, assume independence
    p0 = ideal_p0
    p1 = 1 - p0
    # Compute Fourier coefficient
    phat0 = 1 / 2 * (p0 * (-1)**(0 * 0) + p1 * (-1)**(0 * 1))
    phat1 = 1 / 2 * (p0 * (-1)**(1 * 0) + p1 * (-1)**(1 * 1))

    for i in range(shape[0]):
        eps = prior_lambdas[i][2]
        noisy_p0 = ((1 - eps)**(0))**gate_num * phat0 * (-1)**(0 * 0) + (
            (1 - eps)**(1))**gate_num * phat1 * (-1)**(1 * 0)
        noisy_p1 = ((1 - eps)**(0))**gate_num * phat0 * (-1)**(0 * 1) + (
            (1 - eps)**(1))**gate_num * phat1 * (-1)**(1 * 1)
        M_ideal = np.array([noisy_p0, noisy_p1])

        A = errMitMat(prior_lambdas[i])
        M_meaerr = np.dot(A, M_ideal)

        # Only record interested qubits
        qs = np.append(qs, M_meaerr[0])
    return qs


# Used to call output, delete ideal_p0 parameter
def output_gate(d,
                interested_qubit,
                M,
                params,
                gate_sd,
                meas_sd,
                gate_type,
                gate_num,
                seed=127,
                file_address='',
                write_data_for_SB=False):
    """
      The main function that do all Bayesian inferrence part

    Parameters
    ----------
    d : array
        array of data (Observed QoI). Here, it is array of prob. of meas. 0.
    interested_qubit : int
        The index of qubit that we are looking at. 
        For naming the figure file only.
    M : int
        Number of priors required.
    params : dict
        A dictionary records backend properties. Must have
        {'pm1p0': float # Pr(Meas. 1| Prep. 0)
         'pm0p1': float # Pr(Meas. 0| Prep. 1)
         }
    gate_sd : float
        standard deviation for truncated normal distribution when generating 
        prior gate error parameters.
    meas_sd : float
        same as gate_sd but for meausrement error parameters.
    gate_type : String
        Chosen between "X", "Y", and "Z". 
        Should be the same as gate_type in Gateexp().
    gate_num : int
        number of gates in the experiment circuit.
        Should be the same as nGates in Gateexp().
    seed : int, optional
        Seed for random numbers. The default is 127.
    file_address: string, optional
        The relative file address to save data file. 
        Ends with '/' if not empty
        The default is ''.
    write_data_for_SB : boolean, optional
        If write data to execute standard Bayesian.
        This parameter is only for the purpose of writing paper.
        Just IGNORE IT.
        The default is False.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    prior_lambdas : numpy array
        prior lambdas in the form of a-by-b matrix where 
        a is the number of priors and m is the number of model parameters
    post_lambdas : numpy array
        prior lambdas in the form of a-by-b matrix where 
        a is the number of posterior and m is the number of model parameters

    """

    # Algorithm 1 of https://doi.org/10.1137/16M1087229

    average_lambdas = np.array([
        1 - params[interested_qubit]['pm1p0'],
        1 - params[interested_qubit]['pm0p1']
    ])
    if gate_type == 'X':
        average_lambdas = np.append(average_lambdas,
                                    2 * params[interested_qubit]['u3_error'])
        if gate_num % 2:
            ideal_p0 = 0
        else:
            ideal_p0 = 1
    elif gate_type == 'Y':
        average_lambdas = np.append(average_lambdas,
                                    2 * params[interested_qubit]['u3_error'])
        if gate_num % 2:
            ideal_p0 = 0
        else:
            ideal_p0 = 1
    elif gate_type == 'Z':
        average_lambdas = np.append(average_lambdas,
                                    2 * params[interested_qubit]['u1_error'])
        ideal_p0 = 0
    else:
        raise Exception('Choose gate_type from X, Y, Z')

    # write data for standard bayesian inference
    if write_data_for_SB:
        with open(file_address + 'Qubit{}.csv'.format(interested_qubit),
                  mode='w',
                  newline='') as sgr:
            read_writer = csv.writer(sgr,
                                     delimiter=',',
                                     quotechar='"',
                                     quoting=csv.QUOTE_MINIMAL)
            read_writer.writerow(['x', 'y'])
            for i in range(len(d)):
                read_writer.writerow([ideal_p0, d[i]])

    np.random.seed(seed)
    num_lambdas = 3
    # Get distribution of data (Gaussian KDE)
    d_ker = ss.gaussian_kde(d)  # i.e., pi_D^{obs}(q), q = Q(lambda)

    if average_lambdas[0] == 1 or average_lambdas[0] < 0.7:
        average_lambdas[0] = 0.9
    if average_lambdas[1] == 1 or average_lambdas[1] < 0.7:
        average_lambdas[1] = 0.9

    # Sample prior lambdas, assume prior distribution is Normal distribution with mean as the given probality from IBM
    # Absolute value is used here to avoid negative values, so it is little twisted, may consider Gamma Distribution
    prior_lambdas = np.zeros(M * num_lambdas).reshape((M, num_lambdas))

    for i in range(M):
        one_sample = np.zeros(num_lambdas)
        for j in range(num_lambdas):
            if j < 2:
                one_sample[j] = tnorm01(average_lambdas[j], meas_sd)
                # while one_sample[j]<= 0 or one_sample[j] > 1:
                #     one_sample[j] = np.random.normal(average_lambdas[j],meas_sd,1)
            else:
                one_sample[j] = tnorm01(average_lambdas[j], gate_sd)
                # while one_sample[j]<= 0 or one_sample[j] > 1:
                #     one_sample[j] = np.random.normal(average_lambdas[j],gate_sd,1)
        prior_lambdas[i] = one_sample

    # Produce prior QoI
    qs = QoI_gate(prior_lambdas, ideal_p0, gate_num)
    #print(qs)
    qs_ker = ss.gaussian_kde(qs)  # i.e., pi_D^{Q(prior)}(q), q = Q(lambda)

    # Plot and Print
    print('Given Lambdas', average_lambdas)

    # Algorithm 2 of https://doi.org/10.1137/16M1087229

    # Find the max ratio r(Q(lambda)) over all lambdas
#    max_r, max_ind = findM(qs_ker, d_ker, qs)
#    # Print and Check
#    print('Final Accepted Posterior Lambdas')
#    print('M: %.6g Index: %.d pi_obs = %.6g pi_Q(prior) = %.6g' %
#          (max_r, max_ind, d_ker(qs[max_ind]), qs_ker(qs[max_ind])))
    
    max_r, max_q = findM(qs_ker, d_ker)
    # Print and Check
    print('Final Accepted Posterior Lambdas')
    print('M: %.6g Maximizer: %.6g pi_obs = %.6g pi_Q(prior) = %.6g' %
          (max_r, max_q, d_ker(max_q), qs_ker(max_q)))

    post_lambdas = np.array([])
    # Go to Rejection Iteration
    for p in range(M):
        r = d_ker(qs[p]) / qs_ker(qs[p])
        eta = r / max_r
        if eta > np.random.uniform(0, 1, 1):
            post_lambdas = np.append(post_lambdas, prior_lambdas[p])

    post_lambdas = post_lambdas.reshape(
        int(post_lambdas.size / num_lambdas),
        num_lambdas)  # Reshape since append destory subarrays
    post_qs = QoI_gate(post_lambdas, ideal_p0, gate_num)
    post_ker = ss.gaussian_kde(post_qs)

    xs = np.linspace(0, 1, 1000)
    xsd = np.linspace(0.1, 0.9, 1000)

    I = 0
    for i in range(xs.size - 1):
        q = xs[i]
        if qs_ker(q) > 0:
            r = d_ker(q) / qs_ker(q)
            I += r * qs_ker.pdf(q) * (xs[i + 1] - xs[i])  # Just Riemann Sum

    print('Accepted Number N: %.d, fraction %.3f' %
          (post_lambdas.shape[0], post_lambdas.shape[0] / M))
    print('I(pi^post_Lambda) = %.5g' % (I))  # Need to close to 1
    print('Posterior Lambda Mean', closest_average(post_lambdas))
    print('Posterior Lambda Mode', closest_mode(post_lambdas))

    print('0 to 1: KL-Div(pi_D^Q(post),pi_D^obs) = %6g' %
          (ss.entropy(post_ker(xs), d_ker(xs))))
    print('0 to 1: KL-Div(pi_D^obs,pi_D^Q(post)) = %6g' %
          (ss.entropy(d_ker(xs), post_ker(xs))))
    
    # File name change from Post_Qubit{} to Gate_Post_Qubit{}
    with open(file_address + 'Gate_Post_Qubit{}.csv'.format(interested_qubit),
              mode='w',
              newline='') as sgm:
        writer = csv.writer(sgm,
                            delimiter=',',
                            quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        for i in range(post_lambdas.shape[0]):
            writer.writerow(post_lambdas[i])

    # Plots
    plt.figure(figsize=(width,height), dpi=120, facecolor='white')
    # figure(num=None, figsize=fig_size, dpi=fig_dpi, facecolor='w', edgecolor='k')
    plt.plot(xsd,
             d_ker(xsd),
             color='Red',
             linestyle='dashed',
             linewidth=3,
             label='$\\pi^{\\mathrm{obs}}_{\\mathcal{D}}$')
    plt.plot(xsd, post_ker(xsd), color='Blue', label='$\\pi_{\\mathcal{D}}^{Q(\\mathrm{post})}$')
    plt.xlabel('Pr(Meas. 0)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_address + 'QoI-Qubit%g.pdf' % interested_qubit)
    plt.show()
    
    plt.figure(figsize=(width,height), dpi=120, facecolor='white')
    # figure(num=None, figsize=fig_size, dpi=fig_dpi, facecolor='w', edgecolor='k')
    eps_ker = ss.gaussian_kde(post_lambdas[:, 2])
    eps_line = np.linspace(
        np.min(post_lambdas, axis=0)[2],
        np.max(post_lambdas, axis=0)[2], 1000)
    plt.plot(eps_line, eps_ker(eps_line), color='Blue')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(-5, 1))
    plt.xlabel('$\epsilon_g$')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig(file_address + 'Eps-Qubit%g.pdf' % interested_qubit)
    plt.show()
    return prior_lambdas, post_lambdas


def read_data(interested_qubit, gate_type, gate_num, file_address=''):
    """Read out bit string data from csv file generated by collect_filter_data().

    Args:
      file_address:
        file address, string ends with '/' if not empty

    Returns:
      An array of bit strings.
    """
    with open(file_address + 'Readout_{}{}Q{}.csv'.format(
            gate_num, gate_type, interested_qubit),
              mode='r') as measfile:
        reader = csv.reader(measfile)
        cali01 = np.asarray([row for row in reader][0])

    return cali01

    # File name change from Post_Qubit{} to Gate_Post_Qubit{}
def read_post(itr,
              shots,
              interested_qubits,
              gate_type,
              gate_num,
              file_address=''):
    """
        Read posteror from files
        See output_gate() for explaintion for arguments and returns.
    """
    post = {}
    for q in interested_qubits:
        data = read_data(q, gate_type, gate_num, file_address=file_address)
        d = getData0(data, int(itr * shots / 1024), q)
        post_lambdas = pd.read_csv(self.file_address +
                                   'Gtae_Post_Qubit{}.csv'.format(i),
                                   header=None).to_numpy()
        post['Qubit' + str(i)] = post_lambdas

        # information part
        xs = np.linspace(0, 1, 1000)
        xsd = np.linspace(0.2, 0.8, 1000)
        d_ker = ss.gaussian_kde(d)
        print('Posterior Lambda Mean', closest_average(post_lambdas))
        print('Posterior Lambda Mode', closest_mode(post_lambdas))
        print('0 to 1: KL-Div(pi_D^Q(post),pi_D^obs) = %6g' %
              (ss.entropy(post_ker(xs), d_ker(xs))))
        print('0 to 1: KL-Div(pi_D^obs,pi_D^Q(post)) = %6g' %
              (ss.entropy(d_ker(xs), post_ker(xs))))
        
        # figure(num=None, figsize=fig_size, dpi=fig_dpi, facecolor='w', edgecolor='k')
        plt.figure(figsize=(width,height), dpi=120, facecolor='white')
        post_qs = QoI_gate(post_lambdas, ideal_p0, gate_num)
        post_ker = ss.gaussian_kde(post_qs)
        plt.plot(xsd,
                 d_ker(xsd),
                 color='Red',
                 linestyle='dashed',
                 linewidth=3,
                 label='Observed QoI')
        plt.plot(xsd, post_ker(xsd), color='Blue', label='QoI by Posterior')
        plt.xlabel('Pr(Meas. 0)')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(file_address + 'QoI-Qubit%g.pdf' % interested_qubit)
        plt.show()
        
        # figure(num=None, figsize=fig_size, dpi=fig_dpi, facecolor='w', edgecolor='k')
        plt.figure(figsize=(width,height), dpi=120, facecolor='white')
        eps_ker = ss.gaussian_kde(post_lambdas[:, 2])
        eps_line = np.linspace(
            np.min(post_lambdas, axis=0)[2],
            np.max(post_lambdas, axis=0)[2], 1000)
        plt.plot(eps_line, eps_ker(eps_line), color='Blue')
        plt.ticklabel_format(axis="x", style="sci", scilimits=(-5, 1))
        plt.xlabel('$\epsilon_g$')
        plt.ylabel('Density')
        plt.tight_layout()
        plt.savefig(file_address + 'Eps-Qubit%g.pdf' % interested_qubit)
        plt.show()

    return post


def plotComparsion(data, post_lambdas, q, file_address=''):
    """
        Plot comparision between BJW bayesian and standard bayesian.
        For writing paper only.

    """
    postSB = pd.read_csv(file_address +
                         'StandPostQubit{}.csv'.format(q)).to_numpy()
    SB = QoI_gate(postSB, 1, 200)
    OB = QoI_gate(post_lambdas, 1, 200)
    minSB = min(SB) 
    minOB = min(OB)
    maxSB = max(SB)
    maxOB = max(OB)
    line = np.linspace(min(minSB, minOB), max(maxSB, maxOB), 1000)
    SBker = ss.gaussian_kde(SB)
    OBker = ss.gaussian_kde(OB)
    dker = ss.gaussian_kde(data)
    
    # figure(num=None, figsize=fig_size, dpi=fig_dpi, facecolor='w', edgecolor='k')
    plt.figure(figsize=(width,height), dpi=120, facecolor='white')
    plt.plot(line, SBker(line), color='Green', linewidth=2, label='Stand.')
    plt.plot(line, OBker(line), color='Blue', linewidth=2, label='Cons.')
    plt.plot(line,
             dker(line),
             color='Red',
             linestyle='dashed',
             linewidth=4,
             label='Data')
    plt.xlabel('Pr(Meas. 0)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_address + 'SBOB-Qubit%g.pdf' % q)
    plt.show()


#################### For Error Filtering #########################
def sxpower(s, x):
    total = 0
    length = len(s)
    for i in range(length):
        si = int(s[i])
        xi = int(x[i])
        total += si * xi
    return total


def count1(s):
    count = 0
    for i in range(len(s)):
        if s[i] == '1':
            count += 1
    return count


def gate_matrix(length, eps, m):
    """
        Generate matrix for denosing gate error

    """
    size = 2**length
    mat = np.empty([size, size], dtype=np.float64)
    for row in range(size):
        for col in range(size):
            x = ("{0:0" + str(length) + "b}").format(row)
            s = ("{0:0" + str(length) + "b}").format(col)
            power = sxpower(s, x)
            mat[row, col] = ((-1)**power) * ((1 - eps)**(count1(s) * m))
    return mat


def find_least_norm_gate(ptilde):
    """
        Only for one-qubit. Similar to find_least_norm() in measfilter.py.

    """
    # Formulation
    Q = 2 * matrix(np.identity(2))
    p = -2 * matrix(ptilde)

    G = matrix(np.array([[0, 1], [0, -1]]), (2, 2), 'd')
    h = 0.5 * matrix(np.ones(2))

    A = matrix(np.array([1, 0]), (1, 2), 'd')
    b = matrix(0.5)

    solvers.options['show_progress'] = False
    sol = solvers.qp(Q, p, G, h, A, b)
    return sol['status'], sol['x']


def gate_denoise(m, p0s, lambdas):
    """
        Complete function for filter gate and measurement errors.

    """
    denoised = []
    meas_err_mat = errMitMat([lambdas[0], lambdas[1]])
    M = gate_matrix(1, lambdas[2], m)
    for p0 in p0s:
        ptilde = np.array([p0, 1 - p0])
        gate_ptilde = np.linalg.solve(meas_err_mat, ptilde)
        phat = np.linalg.solve(M, gate_ptilde)
        status, opt_phat = find_least_norm_gate(phat)
        opt_recovered_p0 = opt_phat[0] + opt_phat[1] * (-1)**(
            1 * 0)  # phat(0) + phat(1)
        opt_recovered_p1 = opt_phat[0] + opt_phat[1] * (-1)**(
            1 * 1)  # phat(0) - phat(1)
        denoised.append(opt_recovered_p0)

    return denoised


########################## Class for Error Filtering ########################
class GMFilter:
    """ Gate and Measurement error filter.

    Attributes:
        interested_qubits: array,
          qubit indices that experiments are applied on.
        gate_type : String
            Chosen between "X", "Y", and "Z". 
            Should be the same as gate_type in Gateexp().
        gate_num : int
            number of gates in the experiment circuit.
            Should be the same as nGates in Gateexp().
        device_param_address: string
          the address for saving Params.csv. 
          End with '/' if not empty.
        data_file_address: string
          the address for saving 
          Readout_{gate_num}{gate_type}Q{qubit_index}.csv. 
          End with '/' if not empty.
        post: dict
          posterior of each qubit. {'Qubit0':[...], 'Qubit1': [...], ...}
        params: dict
          backend properties. Not None after execute inference()
        modes: numpy array
          posterior MAP. {'Qubit0':[...], 'Qubit1': [...], ...}
          Not None after execute inference()
        means: numpy array
          posterior mean. {'Qubit0':[...], 'Qubit1': [...], ...}
          Not None after execute inference()
          
    """
    def __init__(self,
                 interested_qubits,
                 gate_num,
                 gate_type,
                 device_param_address='',
                 data_file_address=''):
        self.interested_qubits = interested_qubits
        self.device_param_address = device_param_address
        self.data_file_address = data_file_address
        self.gate_type = gate_type
        self.gate_num = gate_num
        self.params = None
        self.post = {}
        self.modes = None
        self.means = None

    def mean(self):
        res = {}
        for q in self.interested_qubits:
            res['Qubit' + str(q)] = closest_average(self.post['Qubit' +
                                                              str(q)])
        return res

    def mode(self):
        res = {}
        for q in self.interested_qubits:
            res['Qubit' + str(q)] = closest_mode(self.post['Qubit' + str(q)])
        return res

    def inference(self,
                  nPrior=40000,
                  meas_sd=0.1,
                  gate_sd=0.01,
                  seed=127,
                  shots_per_point=1024,
                  write_data_for_SB=False):
        """
          Same as output_gate().

        Parameters
        ----------
        nPrior : int, optional
            Number of priors required. The default is 40000.
        meas_sd : float, optional
            standard deviation for truncated normal distribution 
            when generating prior measurment error parameters. 
            The default is 0.1.
        gate_sd : TYPE, optional
            standard deviation for truncated normal distribution 
            when generating prior gate error parameters. 
            The default is 0.01.
        seed : int, optional
            Seed for random numbers. The default is 127.
        shots_per_point : int, optional
            how many shots you want to estimate one QoI (prob. of meas. 0).
            Used to control number of data points and accuracy.
            The default is 1024.
        write_data_for_SB : boolean, optional
            If write data to execute standard Bayesian.
            This parameter is only for the purpose of writing paper.
            Just IGNORE IT.
            The default is False.

        Returns
        -------
        None.

        """
        self.params = read_params(self.device_param_address)

        itr = self.params[0]['itr']
        shots = self.params[0]['shots']
        info = {}
        for i in self.interested_qubits:
            print('Qubit %d' % (i))
            data = read_data(i,
                             self.gate_type,
                             self.gate_num,
                             file_address=self.data_file_address)
            d = getData0(data, int(itr * shots / shots_per_point), i)
            _, post_lambdas = output_gate(d,
                                          i,
                                          nPrior,
                                          self.params,
                                          gate_sd,
                                          meas_sd,
                                          self.gate_type,
                                          self.gate_num,
                                          file_address=self.data_file_address)
            self.post['Qubit' + str(i)] = post_lambdas
        self.modes = self.mode()
        self.means = self.mean()
        
        # File name change from Post_Qubit{} to Gate_Post_Qubit{}
    def post_from_file(self):
        """
          Read posterior from file directly if inference() is already run once.

        Returns
        -------
        None.

        """
        for i in self.interested_qubits:
            post_lambdas = pd.read_csv(self.data_file_address +
                                       'Gate_Post_Qubit{}.csv'.format(i),
                                       header=None).to_numpy()
            self.post['Qubit' + str(i)] = post_lambdas
        self.modes = self.mode()
        self.means = self.mean()

    def filter_mean(self, p0s, qubit_index):
        """
          Use posteror mean to filter measurement and gate error out.

        Parameters
        ----------
        p0s : array
            An array of probabilities of measuring 0.
        qubit_index : int
            which qubit that p0s is corresponds to.

        Returns
        -------
        denoised_p0s: array
            p0s w/o gate and measurment error.

        """
        return gate_denoise(self.gate_num, p0s,
                            self.means['Qubit' + str(qubit_index)])

    def filter_mode(self, p0s, qubit_index):
        """
          Use posteror MAP to filter measurement and gate error out.

        Parameters
        ----------
        p0s : array
            An array of probabilities of measuring 0.
        qubit_index : int
            which qubit that p0s is corresponds to.

        Returns
        -------
        denoised_p0s: array
            p0s w/o gate and measurment error.

        """
        return gate_denoise(self.gate_num, p0s,
                            self.modes['Qubit' + str(qubit_index)])
