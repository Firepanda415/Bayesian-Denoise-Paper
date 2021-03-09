# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 18:19:38 2020

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

# For optimization
from cvxopt import matrix, solvers
from scipy.optimize import minimize_scalar
from scipy.special import j1

# fig_size = (8,6)
# fig_dpi = 100
import matplotlib        as     mpl
from   matplotlib        import rc
from   cycler            import cycler

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


def param_record(backend, itr=32, shots=8192, if_write=True, file_address=''):
    """Write backend property into an array of dict 
       and save as csv if permissible.

    Args:
      backend: IBMQBackend
        backend from provider.get_backend().
      itr: int
        number of iterations of job submission.
      shots: int
        number of shots per each job submission.
      if_write: boolean
        True if save the properties as a csv file.
      file_address: string
        The relative file address to save backend properties. 
        Ends with '/' if not empty
        The default is ''.

    Returns: numpy array
      An array of dicts. Each dict records all characterization of one qubit.
    """
    prop_dict = backend.properties().to_dict()
    nQubits = len(prop_dict['qubits'])
    backend_name = prop_dict['backend_name']

    target_qubits = range(nQubits)
    allParam = np.array([])
    for target_qubit in target_qubits:
        params = {
            'qubit':
            target_qubit,
            'update_date':
            prop_dict['last_update_date'],
            'T1':
            prop_dict['qubits'][target_qubit][0]['value'],
            'T2':
            prop_dict['qubits'][target_qubit][1]['value'],
            'freq':
            prop_dict['qubits'][target_qubit][2]['value'],
            'readout_err':
            prop_dict['qubits'][target_qubit][3]['value'],
            'pm0p1':
            prop_dict['qubits'][target_qubit][4]['value'],
            'pm1p0':
            prop_dict['qubits'][target_qubit][5]['value'],
            'id_error':
            prop_dict['gates'][4 * target_qubit]['parameters'][0]['value'],
            'id_len':
            prop_dict['gates'][4 * target_qubit]['parameters'][1]['value'],
            'u1_error':
            prop_dict['gates'][4 * target_qubit + 1]['parameters'][0]['value'],
            'u1_len':
            prop_dict['gates'][4 * target_qubit + 1]['parameters'][1]['value'],
            'u2_error':
            prop_dict['gates'][4 * target_qubit + 2]['parameters'][0]['value'],
            'u2_len':
            prop_dict['gates'][4 * target_qubit + 2]['parameters'][1]['value'],
            'u3_error':
            prop_dict['gates'][4 * target_qubit + 3]['parameters'][0]['value'],
            'u3_len':
            prop_dict['gates'][4 * target_qubit + 3]['parameters'][1]['value'],
            'itr':
            itr,
            'shots':
            shots,
        }
        allParam = np.append(allParam, params)

    if if_write:
        with open(file_address + 'Params.csv', mode='w', newline='') as sgm:
            param_writer = csv.writer(sgm,
                                      delimiter=',',
                                      quotechar='"',
                                      quoting=csv.QUOTE_MINIMAL)
            for pa in allParam:
                for key, val in pa.items():
                    param_writer.writerow([key, val])
                param_writer.writerow(['End'])

    return allParam


def meas_circ(nQubits, backend, itr=32):
    """Generate circuit for inferring measurement error.

    Args:
      nQubits: int
        Number of qubits.
      backend: IBMQBackend
        backend from provider.get_backend().
      itr: int
        Number of data points

    Returns: array
      A list of circuits with name 'itr0', 'itr1',...
    """
    circ = QuantumCircuit(nQubits, nQubits)
    for i in range(nQubits):
        circ.h(i)
    circ.measure(range(nQubits), range(nQubits))
    circ_trans = transpile(circ, backend, initial_layout=range(nQubits))

    circs = []
    for i in range(itr):
        circs.append(circ_trans.copy('itr' + str(i)))
    return circs


def collect_filter_data(backend,
                        itr=32,
                        shots=8192,
                        if_monitor_job=True,
                        if_write=True,
                        file_address=''):
    """Collect data for constructing error filter.

    Args:
      backend: IBMQBackend
        backend from provider.get_backend().
      itr: int
        number of iterations of job submission.
      shots: int
        number of shots per each job submission.
      if_monitor_job: boolean
        if show information for job progress.
      if_write: boolean
        True if save the properties as a csv file.
      file_address: string
        The relative file address to save data file. 
        Ends with '/' if not empty
        The default is ''.

    Returns: numpy array
      An array of bit strings which is the output from circuits.
    """
    param_record(backend, itr, shots, if_write, file_address)

    nQubits = len(backend.properties().to_dict()['qubits'])
    readout_m0 = np.array([])
    circ = meas_circ(nQubits, backend, itr=itr)

    # Excute jobs
    job_m0 = execute(circ, backend=backend, shots=shots, memory=True)
    if if_monitor_job:
        job_monitor(job_m0)

    # Record bit string
    m0_res = job_m0.result()
    for i in range(itr):
        readout_m0 = np.append(readout_m0,
                               m0_res.get_memory(experiment=('itr' + str(i))))

    if if_write:
        with open(file_address + 'Filter_data.csv', mode='w') as sgr:
            read_writer = csv.writer(sgr,
                                     delimiter=',',
                                     quotechar='"',
                                     quoting=csv.QUOTE_MINIMAL)
            read_writer.writerow(readout_m0)

    return readout_m0


def read_params(file_address=''):
    """Read out backend properties from csv file generated by param_record().

    Args:
      file_address: string
        The relative file address to read backend properties. 
        Ends with '/' if not empty
        The default is ''.

    Returns: numpy array
      An array of dicts. Each dict records all characterization of one qubit.
    """
    textKeys = ['name', 'update_date', 'qubit']
    intKeys = ['itr', 'shots']
    # Read Parameters
    with open(file_address + 'Params.csv', mode='r') as sgm:
        reader = csv.reader(sgm)
        params = np.array([])
        singleQubit = {}
        first = True
        for row in reader:
            if row[0] == 'End':
                params = np.append(params, singleQubit)
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


def read_filter_data(file_address=''):
    """Read out bit string data from csv file generated by collect_filter_data().

    Args:
      file_address: string
        The relative file address to read data for filter generation. 
        Ends with '/' if not empty
        The default is ''.

    Returns: numpy array
      An array of bit strings.
    """
    with open(file_address + 'Filter_data.csv', mode='r') as measfile:
        reader = csv.reader(measfile)
        cali01 = np.asarray([row for row in reader][0])

    return cali01


def tnorm01(center, sd, size=1):
    """ Generate random numbers for truncated normal with range [0,1]

    Args:
      center: float
        mean of normal distribution
      sd: float
        standard deviation of normal distribution
      size: int
        number of random numbers

    Returns: array
       an array of random numbers
    """
    upper = 1
    lower = 0
    a, b = (lower - center) / sd, (upper - center) / sd
    return ss.truncnorm.rvs(a, b, size=size) * sd + center


def find_mode(data):
    """Find the mode through Gaussian KDE.

    Args:
      data: array
        an array of floats

    Returns: float
      the mode.
    """
    kde = ss.gaussian_kde(data)
    line = np.linspace(min(data), max(data), 10000)
    return line[np.argmax(kde(line))]


def closest_mode(post_lambdas):
    """Find the tuple of model parameters that closed to 
       the Maximum A Posteriori (MAP) of 
       posterior distribution of each parameter

    Args:
      post_lambdas: numpy array
        an n-by-m array where n is the number of posteriors and m is number 
        of parameters in the model

    Returns: numpy array
      an array that contains the required model parameters.
    """

    mode_lam = []
    for j in range(post_lambdas.shape[1]):
        mode_lam.append(find_mode(post_lambdas[:, j]))

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
    """Find the tuple of model parameters that closed to 
       the mean of posterior distribution of each parameter

    Args:
      post_lambdas: numpy array
        an n-by-m array where n is the number of posteriors and m is number 
        of parameters in the model

    Returns: numpy array
      an array that contains the required model parameters.
    """
    sol = np.array([])
    smallest_norm = nl.norm(post_lambdas[0])

    ave_lam = np.mean(post_lambdas, axis=0)

    for lam in post_lambdas:
        norm_diff = nl.norm(lam - ave_lam)

        if norm_diff < smallest_norm:
            smallest_norm = norm_diff
            sol = lam
    return sol


def dictToVec(nQubits, counts):
    """ Transfer counts to probabilities

    Args:
      nQUbits: int
        number of qubits
      counts: dict
        an dictionary in the form {basis string: frequency}. E.g.
        {"01": 100
         "11": 100}
        dict key follow little-endian convention

    Returns: numpy array
      an probability vector (array). E.g.
      [0, 0.5, 0, 0.5] is the result from example above.
    """
    vec = np.zeros(2**nQubits)
    form = "{0:0" + str(nQubits) + "b}"
    for i in range(2**nQubits):
        key = form.format(i) # consider key = format(i,'0{}b'.format(nQubits))
                             # and delete variable form
        if key in counts.keys():
            vec[i] = int(counts[key])
        else:
            vec[i] = 0
    return vec


def dictToVec_inv(nQubits, counts):
    """ 
      Same as dictToVec() but key uses big-endian convention
    """
    vec = np.zeros(2**nQubits)
    form = "{0:0" + str(nQubits) + "b}"
    for i in range(2**nQubits):
        key = form.format(i)[::-1]
        if key in counts.keys():
            vec[i] = int(counts[key])
        else:
            vec[i] = 0
    return vec


def vecToDict(nQubits, shots, vec):
    """ Transfer probability vector to dict in the form 
        {basis string: frequency}. E.g. [0, 0.5, 0, 0.5] in 200 shots becomes
            {"01": 100
             "11": 100}
        dict key follow little-endian convention

    Parameters
    ----------
    nQubits : int
        number of qubits.
    shots : int
        number of shots.
    vec : array
        probability vector that sums to 1.

    Returns
    -------
    counts : dict
        Counts for each basis.

    """
    counts = {}
    form = "{0:0" + str(nQubits) + "b}"
    for i in range(2**nQubits):
        key = form.format(i)
        counts[key] = int(vec[i] * shots)
    return counts


def vecToDict_inv(nQubits, shots, vec):
    """ 
      Same as dictToVec() but key uses big-endian convention
    """
    counts = {}
    form = "{0:0" + str(nQubits) + "b}"
    for i in range(2**nQubits):
        key = form.format(i)[::-1]
        counts[key] = int(vec[i] * shots)
    return counts


# Functions
def getData0(data, num_group, interested_qubit):
    """ Get the probabilities of measuring 0 from binay readouts
        **Binary number follows little-endian convention**

    Parameters
    ----------
    data : array
        an array of binary readouts.
    num_group : int
        number of probabilities. E.g. If you have 1000 binary string readouts,
        you can set num_group = 10 so that you will have 10 probabilities,
        each probability is calculated from 100 binary string readouts
    interested_qubit : int
        which qubit you interested in.

    Returns
    -------
    prob0 : numpy array
        array of proabilitilies of measuring 0.

    """
    prob0 = np.zeros(num_group)
    groups = np.split(data, num_group)
    for i in range(num_group):
        count = 0
        for d in groups[i]:
            d_rev = d[::-1]
            if d_rev[interested_qubit] == '0':
                count += 1

        prob0[i] = count / groups[i].size
    return prob0


def errMitMat(lambdas_sample):
    """
    Compute the matrix A from
    Ax = b, where A is the error mitigation matrix (transition matrix),
    x is the number appearence of a basis in theory
    b is the number appearence of a basis in practice with noise

    Parameters
    ----------
    lambdas_sample : numpy array
        first two entry must be 
        Pr(Measuring 0|Preparing 0) and Pr(Measuring 1|Preparing 1)

    Returns
    -------
    A : numpy array
        Transition matrix that applies classical measurement error.

    """
    #
    # Input; lambdas_sample - np.array, array of (1 - error rate) whose length is number of qubits
    # Output; A - np.ndarray, as described above
    pm0p0 = lambdas_sample[0]
    pm1p1 = lambdas_sample[1]
    # Initialize the matrix
    A = np.array([[pm0p0, 1 - pm1p1], [1 - pm0p0, pm1p1]])
    return (A)


def QoI(prior_lambdas):
    """
    Function equivalent to Q(lambda) in https://doi.org/10.1137/16M1087229

    Parameters
    ----------
    prior_lambdas : numpy array
        each subarray is an individual prior lambda.

    Returns
    -------
    qs : numpy array
        QoI's. Here they are the probability of measuring 0 with each given
        prior lambdas in prior_lambdas.

    """
    shape = prior_lambdas.shape
    nQubit = 1

    # Initialize the output array
    qs = np.array([])

    # Smiluate measurement error, assume independence
    for i in range(shape[0]):
        M_ideal = np.ones(2**nQubit) / 2**nQubit

        A = errMitMat(prior_lambdas[i])
        M_noisy = np.dot(A, M_ideal)

        # Only record interested qubits
        qs = np.append(qs, M_noisy[0])
    return qs


def dq(x, qs_ker, d_ker):
    if qs_ker(x)[0] > 0:
        if np.abs(d_ker(x)[0]) <1e-6: # A lot of 0s in both sides may cause opt algorithm terminates
            return np.abs(0.5-x)
        else :
            return - d_ker(x)[0] / qs_ker(x)[0]
    else:
        return np.infty


def findM(qs_ker, d_ker):
    """
    Function for finding the M, the largest r(Q(lambda)) over all lambda
    in Algorithm 2 of https://doi.org/10.1137/16M1087229
    
    we use minimize_scalar from scipy

    Parameters
    ----------
    qs_ker : scipy.stats.gaussian_kde
        the Q_D^{Q(prior)}(q) in Algorithm 1 of https://doi.org/10.1137/16M1087229.
    d_ker : scipy.stats.gaussian_kde
        pi_D^{obs}(q) in A997 of https://doi.org/10.1137/16M1087229.
    qs : numpy array
        samples generated from some given lambdas

    Returns
    -------
    M : float
        the largest r(Q(lambda)) over all lambda
    optimizer : float
        corresponding index of M

    """
    # M = -1  # probablities cannot be negative, so -1 is small enough
    # index = -1
    # for i in range(qs.size):
    #     if qs_ker(qs[i]) > 0:
    #         if M <= d_ker(qs[i]) / qs_ker(qs[i]):
    #             M = d_ker(qs[i]) / qs_ker(qs[i])
    #             index = i
    
#                
    # xs = np.linspace(0, 1, 1000)
    # ys = np.array([dq(x, qs_ker, d_ker) for x in xs])
    # plt.plot(xs, ys)
    # plt.ylabel('d/q')
    # plt.show()
    
    xs = np.linspace(0, 1, 1000)
    ys = np.array([dq(x, qs_ker, d_ker) for x in xs], dtype=np.float64)
    plt.plot(xs, ys)
    plt.ylabel('d/q')
    plt.xlabel('x')
    plt.show()
    
    res = minimize_scalar(dq, args =(qs_ker, d_ker) ,bounds=(0,1), method='bounded', 
                          options={'maxiter': 5000})
    # res = minimize(dq, [0.5], args =(qs_ker, d_ker) ,bounds=((0,1)), method='L-BFGS-B')
    
    
    try:
        return -res.fun[0], res.x[0]
    except Exception:
        return -res.fun, res.x


def find_least_norm(nQubits, ptilde):
    """
    Solve min ||ptilde - p||_2
          s.t.
            each entry of p sums to 1
            each entry of p is non-negative

    Parameters
    ----------
    nQubits : int
        Number of qubits.
    ptilde : array
        probability vector.

    Returns
    -------
    sol['status']: String
        'optimal' if solve successfully.
    sol['x']: array
        the optimizer.

    """
    # Formulation
    Q = 2 * matrix(np.identity(2**nQubits))
    p = -2 * matrix(ptilde)

    G = -matrix(np.identity(2**nQubits))
    h = matrix(np.zeros(2**nQubits))

    A = matrix(np.ones(2**nQubits), (1, 2**nQubits))
    b = matrix(1.0)

    solvers.options['show_progress'] = False
    sol = solvers.qp(Q, p, G, h, A, b)
    return sol['status'], sol['x']


def output(d,
           interested_qubit,
           M,
           params,
           prior_sd,
           seed=127,
           show_denoised=False,
           file_address=''):
    """
      The main function that do all Bayesian inferrence part

    Parameters
    ----------
    d : array
        array of data (Observed QoI). Here, it is array of prob. of meas. 0.
    interested_qubit : int
        The index of qubit that we are looking at. 
        For the use of naming the figure file only.
    M : int
        Number of priors required.
    params : dict
        A dictionary records backend properties. Must have
        {'pm1p0': float # Pr(Meas. 1| Prep. 0)
         'pm0p1': float # Pr(Meas. 0| Prep. 1)
         }
    prior_sd : float
        standard deviation for truncated normal distribution when generating 
        prior parameters (for measurement error).
    seed : int, optional
        Seed for random numbers. The default is 127.
    show_denoised : boolean, optional
        If plot the comparision between post. parameter and 
        given parameters in params. The default is False since 
        it is very time consuming and unnecessary in most of case
    file_address : String, optional
        The relative file address to save posteriors and figures. 
        Ends with '/' if not empty
        The default is ''.

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
    np.random.seed(seed)
    num_lambdas = 2
    # Get distribution of data (Gaussian KDE)
    d_ker = ss.gaussian_kde(d)  # i.e., pi_D^{obs}(q), q = Q(lambda)
    average_lambdas = np.array([
        1 - params[interested_qubit]['pm1p0'],
        1 - params[interested_qubit]['pm0p1']
    ])

    # Compute distribution of Pr(meas. 0) from Qiskit results
    given_errmat = errMitMat(average_lambdas)
#    qiskit_p0 = np.empty(len(d))
#    for i in range(len(d)):
#        single_res = nl.solve(given_errmat, [d[i], 1 - d[i]])
#        qiskit_p0[i] = single_res[0]
#    qiskit_ker = ss.gaussian_kde(qiskit_p0)

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
            one_sample[j] = tnorm01(average_lambdas[j], prior_sd)
            # while one_sample[j]<= 0 or one_sample[j] > 1:
            #     one_sample[j] = np.random.normal(average_lambdas[j],prior_sd,1)
        prior_lambdas[i] = one_sample

    # Produce prior QoI
    qs = QoI(prior_lambdas)
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
        # Monitor Progress
        print('Progress: {:.3%}'.format(p/M), end='\r')
        
        r = d_ker(qs[p]) / qs_ker(qs[p])
        eta = r / max_r
        if eta > np.random.uniform(0, 1, 1):
            post_lambdas = np.append(post_lambdas, prior_lambdas[p])
    print()
    
    
    post_lambdas = post_lambdas.reshape(
        int(post_lambdas.size / num_lambdas),
        num_lambdas)  # Reshape since append destory subarrays
    post_qs = QoI(post_lambdas)
    post_ker = ss.gaussian_kde(post_qs)

    xs = np.linspace(0, 1, 1000)
    xsd = np.linspace(0.3, 0.7, 1000)

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
    # print('0 to 1: KL-Div(qiskit,pi_D^obs) = %6g' %
    #       (ss.entropy(qiskit_ker(xs), d_ker(xs))))
    # print('0 to 1: KL-Div(pi_D^obs,qiskit) = %6g' %
    #       (ss.entropy(d_ker(xs), qiskit_ker(xs))))

    # print('Post and Data: Sum of Differences ',
    #       np.sum(np.abs(post_ker(xs) - d_ker(xs)) / 1000))
    # print('Qisk and Data: Sum of Differences ',
    #       np.sum(np.abs(qiskit_ker(xs) - d_ker(xs)) / 1000))

    with open(file_address + 'Post_Qubit{}.csv'.format(interested_qubit),
              mode='w',
              newline='') as sgm:
        writer = csv.writer(sgm,
                            delimiter=',',
                            quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)
        for i in range(post_lambdas.shape[0]):
            writer.writerow(post_lambdas[i])

    # Plots
    # fig = plot_setup()
    # figure(num=None, figsize=fig_size, dpi=fig_dpi, facecolor='w', edgecolor='k')
    plt.figure(figsize=(width,height), dpi=120, facecolor='white')
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

    if show_denoised:
        res_proc = np.array([])
        for lam in post_lambdas:
            M_sub = errMitMat(lam)
            res_proc = np.append(
                res_proc,
                np.array([
                    nl.solve(M_sub, [d[ind], 1 - d[ind]])[0]
                    for ind in range(len(d))
                ]))
        proc_ker = ss.gaussian_kde(res_proc)

        # Denoised by Qiskit Parameters
        M_qsub = errMitMat(
            np.array([
                1 - params[interested_qubit]['pm1p0'],
                1 - params[interested_qubit]['pm0p1']
            ]))
        res_qisk = np.array([
            nl.solve(M_qsub, [d[ind], 1 - d[ind]])[0] for ind in range(len(d))
        ])
        qisk_ker = ss.gaussian_kde(res_qisk)
        
        # fig = plot_setup()
        # figure(num=None, figsize=fig_size, dpi=fig_dpi, facecolor='w', edgecolor='k')
        #plt.plot(xsd,d_ker(xsd),color='Red',linestyle='dashed',label = '(Noisy) Data')
        plt.figure(figsize=(width,height), dpi=120, facecolor='white')
        plt.plot(xsd, proc_ker(xsd), color='Blue', label='By Post')
        plt.plot(xsd, qisk_ker(xsd), color='green', label='By Prior')
        plt.axvline(x=0.5, color='black', label='Ideal')
        plt.xlabel('Pr(Meas. 0)')
        plt.ylabel('Density')
        #plt.title('Denoised Pr(Meas. 0), Qubit %g'%interested_qubit)
        plt.legend()
        plt.tight_layout()
        plt.savefig(file_address + 'DQoI-Qubit%g.pdf' % interested_qubit)
        plt.show()
    return prior_lambdas, post_lambdas


class MeasFilter:
    """Measurement error filter.

    Attributes:
        qubit_order: array,
          using order[LastQubit, ..., FirstQubit].
        file_address: string
          the address for saving Params.csv and Filter_data.csv. 
          End with '/' if not empty.
        prior: dict
          priors of each qubit. {'Qubit0':[...], 'Qubit1': [...], ...}
        post: dict
          posterior of each qubit. {'Qubit0':[...], 'Qubit1': [...], ...}
        params: dict
          backend properties. Not None after execute inference()
        data: array
          return of read_filter_data(). Not None after execute inference()
        mat_mean: numpy array
          transition matrix created from posterior mean.
          Not None after execute inference()
        mat_mode: numpy array
          transition matrix created from posterior mode.
          Not None after execute inference()
          
    """
    def __init__(self, qubit_order, file_address=''):
        self.file_address = file_address
        self.qubit_order = qubit_order
        self.prior = {}
        self.post = {}
        self.params = None
        self.data = None
        self.mat_mean = None
        self.mat_mode = None

    def create_filter_mat(self):
        # Create filter matrix with Posterior mean
        first = True
        for q in self.qubit_order:
            if first:
                postLam_mean = closest_average(self.post['Qubit' + str(q)])
                Mx = errMitMat(postLam_mean)
                first = False
            else:
                postLam_mean = closest_average(self.post['Qubit' + str(q)])
                Msub = errMitMat(postLam_mean)
                Mx = np.kron(Mx, Msub)
        self.mat_mean = Mx

        # Create filter matrix with Posterior mode
        first = True
        for q in self.qubit_order:
            if first:
                postLam_mean = closest_mode(self.post['Qubit' + str(q)])
                Mx = errMitMat(postLam_mean)
                first = False
            else:
                postLam_mean = closest_mode(self.post['Qubit' + str(q)])
                Msub = errMitMat(postLam_mean)
                Mx = np.kron(Mx, Msub)
        self.mat_mode = Mx

    def inference(self,
                  nPrior=40000,
                  Priod_sd=0.1,
                  seed=227,
                  shots_per_point=1024,
                  show_denoised=False):
        """
          Do Bayesian interence

        Parameters
        ----------
        nPrior : int, optional
            Number of priors required. The default is 40000.
            Same as M in output().
        Priod_sd : float, optional
            standard deviation for truncated normal distribution 
            when generating prior parameters. The default is 0.1.
            Same as prior_sd in output().
        seed : int, optional
            Seed for random numbers. The default is 227.
            Same as seed in output().
        shots_per_point : int, optional
            how many shots you want to estimate one QoI (prob. of meas. 0).
            Used to control number of data points and accuracy.
            The default is 1024.
        show_denoised : boolean, optional
            If plot the comparision between post. parameter and 
            given parameters in params. The default is False since 
            it is very time consuming and unnecessary in most of case.
            Same as show_denoised in output().

        Returns
        -------
        None.

        """

        self.data = read_filter_data(self.file_address)
        self.params = read_params(self.file_address)

        itr = self.params[0]['itr']
        shots = self.params[0]['shots']
        info = {}
        for i in self.qubit_order:
            print('Qubit %d' % (i))
            d = getData0(self.data, int(itr * shots / shots_per_point), i)
            prior_lambdas, post_lambdas = output(
                d,
                i,
                nPrior,
                self.params,
                Priod_sd,
                seed=seed,
                show_denoised=show_denoised,
                file_address=self.file_address)
            self.prior['Qubit' + str(i)] = prior_lambdas
            self.post['Qubit' + str(i)] = post_lambdas
        self.create_filter_mat()

    def post_from_file(self):
        """
          Read posterior from file directly if inference() is already run once.

        Returns
        -------
        None.

        """
        for i in self.qubit_order:
            post_lambdas = pd.read_csv(self.file_address +
                                       'Post_Qubit{}.csv'.format(i),
                                       header=None).to_numpy()
            self.post['Qubit' + str(i)] = post_lambdas
        self.create_filter_mat()

    def mean(self):
        """
           return posterior mean.

        Returns
        -------
        res : dict
            posterior mean of qubits. E.g. 
            {'Qubti0': [...], 'Qubti1': [...], ...}

        """
        res = {}
        for q in self.qubit_order:
            res['Qubit' + str(q)] = closest_average(self.post['Qubit' +
                                                              str(q)])
        return res

    def mode(self):
        """
           return posterior MAP.

        Returns
        -------
        res : dict
            posterior mean of qubits. E.g. 
            {'Qubti0': [...], 'Qubti1': [...], ...}

        """
        res = {}
        for q in self.qubit_order:
            res['Qubit' + str(q)] = closest_mode(self.post['Qubit' + str(q)])
        return res

    def filter_mean(self, counts):
        """
         Use posteror mean to filter measurement error out.

        Parameters
        ----------
        counts : Dict
            Counts of each basis.

        Raises
        ------
        Exception
            If we cannot filter the error.

        Returns
        -------
        proc_counts : dict
            Denoised counts.

        """
        shots = 0
        for key in counts:
            shots += counts[key]

        real_vec = dictToVec(len(self.qubit_order), counts) / shots
        proc_status, proc_vec = find_least_norm(
            len(self.qubit_order), nl.solve(self.mat_mean, real_vec))
        if proc_status != 'optimal':
            raise Exception('Sorry, filtering has failed')
        proc_counts = vecToDict(len(self.qubit_order), shots, proc_vec)
        return proc_counts

    def filter_mode(self, counts):
        """
         Use posteror MAP to filter measurement error out.

        Parameters
        ----------
        counts : Dict
            Counts of each basis.

        Raises
        ------
        Exception
            If we cannot filter the error.

        Returns
        -------
        proc_counts : dict
            Denoised counts.

        """
        shots = 0
        for key in counts:
            shots += counts[key]

        real_vec = dictToVec(len(self.qubit_order), counts) / shots
        proc_status, proc_vec = find_least_norm(
            len(self.qubit_order), nl.solve(self.mat_mode, real_vec))
        if proc_status != 'optimal':
            raise Exception('Sorry, filtering has failed')
        proc_counts = vecToDict(len(self.qubit_order), shots, proc_vec)
        return proc_counts
