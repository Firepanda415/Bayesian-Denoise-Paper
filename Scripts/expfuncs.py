# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 12:52:17 2020

@author: Muqing Zheng
"""

import csv
import numpy as np

from qiskit import Aer, IBMQ
from qiskit import QuantumCircuit, transpile, execute, QuantumRegister
from qiskit.tools.monitor import job_monitor
from qiskit.ignis.mitigation.measurement import (complete_meas_cal,
                                                 CompleteMeasFitter,
                                                 MeasurementFilter)

import sys
from measfilter import *

sys.path.insert(1, '../QREM')
import povmtools
from DetectorTomography import DetectorTomographyFitter, QDTCalibrationSetup
from quantum_tomography_qiskit import detector_tomography_circuits
from QDTErrorMitigator import QDTErrorMitigator




def all_methods_data(interested_qubits,
                     backend,
                     itr,
                     QDT_correlated,
                     shots_per_point=1024,
                     file_address=''):
    """Collect data for our method, Qiskit method, QDT, and standard Bayesian.

    Args:
      interested_qubits:
        an array of ints. REMEBER TO FOLLOW THE ORDER [LAST QUBIT, ..., FIRST QUBIT]
      backend:
        backend from provider.get_backend().
      itr:
        number of iterations of job submission in our method only.
      QDT_correlated:
        True if want qubits corrlected in QDT method.
      file_address:
        file address, string ends with '/' if not empty

    Returns:
      None
    """
    with open(file_address + 'interested_qubits.csv', mode='w',
              newline='') as sgm:
        param_writer = csv.writer(sgm,
                                  delimiter=',',
                                  quotechar='"',
                                  quoting=csv.QUOTE_MINIMAL)
        param_writer.writerow(interested_qubits)

    # Record data for filters (ourmethod)
    print('Our method')
    collect_filter_data(backend,
                        itr=itr,
                        shots=8192,
                        if_monitor_job=True,
                        if_write=True,
                        file_address=file_address)

    # Qiskit Method
    print('Qiskit Method')
    qr = QuantumRegister(len(interested_qubits))
    meas_calibs, state_labels = complete_meas_cal(qr=qr, circlabel='mcal')
    for i in range(len(meas_calibs)):
        meas_calibs[i] = transpile(meas_calibs[i],
                                   backend,
                                   initial_layout=interested_qubits[::-1])

    job = execute(meas_calibs,
                  backend=backend,
                  shots=8192,
                  optimization_level=0)
    job_monitor(job)

    cal_results = job.result()
    meas_fitter = CompleteMeasFitter(cal_results,
                                     state_labels,
                                     circlabel='mcal')
    meas_filter = meas_fitter.filter
    cal_matrix = meas_fitter.cal_matrix

    with open(file_address + 'cal_matrix.csv', mode='w', newline='') as sgm:
        param_writer = csv.writer(sgm,
                                  delimiter=',',
                                  quotechar='"',
                                  quoting=csv.QUOTE_MINIMAL)
        for row in cal_matrix:
            param_writer.writerow(row)
    with open(file_address + 'state_labels.csv', mode='w', newline='') as sgm:
        param_writer = csv.writer(sgm,
                                  delimiter=',',
                                  quotechar='"',
                                  quoting=csv.QUOTE_MINIMAL)
        param_writer.writerow(state_labels)

    # QDT
    print('QDT, correlation = ', QDT_correlated)
    if QDT_correlated:
        qdt_qubit_indices = interested_qubits
        qdt_probe_kets = povmtools.pauli_probe_eigenkets
        qdt_calibration_circuits = detector_tomography_circuits(
            qdt_qubit_indices, qdt_probe_kets)

        print('Number of Circuits needed is ', len(qdt_calibration_circuits))

        # We then execute them on backend prepared earlier.
        shots_number = 8192

        # Perform a noisy simulation
        job = execute(qdt_calibration_circuits,
                      backend=backend,
                      shots=shots_number,
                      optimization_level=0)
        job_monitor(job)
        result = job.result()

        dtf = DetectorTomographyFitter()
        calibration_setup = QDTCalibrationSetup.from_qiskit_results(
            [result], qdt_probe_kets)
        ml_povm_estimator = dtf.get_maximum_likelihood_povm_estimator(
            calibration_setup)

        mitigator = QDTErrorMitigator()
        mitigator.prepare_mitigator(ml_povm_estimator)
        trans_mat = mitigator.transition_matrix

        with open(file_address + 'trans_matrix.csv', mode='w',
                  newline='') as sgm:
            param_writer = csv.writer(sgm,
                                      delimiter=',',
                                      quotechar='"',
                                      quoting=csv.QUOTE_MINIMAL)
            for row in trans_mat:
                param_writer.writerow(row)
    else:
        for q in interested_qubits:
            qdt_qubit_indices = [q]
            qdt_probe_kets = povmtools.pauli_probe_eigenkets
            qdt_calibration_circuits = detector_tomography_circuits(
                qdt_qubit_indices, qdt_probe_kets)

            print('Number of Circuits needed is ',
                  len(qdt_calibration_circuits))

            # We then execute them on backend prepared earlier.
            shots_number = 8192

            # Perform a noisy simulation
            job = execute(qdt_calibration_circuits,
                          backend=backend,
                          shots=shots_number,
                          optimization_level=0)
            job_monitor(job)
            result = job.result()

            # Create Mitigator

            dtf = DetectorTomographyFitter()
            calibration_setup = QDTCalibrationSetup.from_qiskit_results(
                [result], qdt_probe_kets)
            ml_povm_estimator = dtf.get_maximum_likelihood_povm_estimator(
                calibration_setup)

            mitigator = QDTErrorMitigator()
            mitigator.prepare_mitigator(ml_povm_estimator)

            trans_mat = mitigator.transition_matrix

            with open(file_address + 'trans_matrix' + str(q) + '.csv',
                      mode='w',
                      newline='') as sgm:
                param_writer = csv.writer(sgm,
                                          delimiter=',',
                                          quotechar='"',
                                          quoting=csv.QUOTE_MINIMAL)
                for row in trans_mat:
                    param_writer.writerow(row)

    # Data for Standard Bayesian
    # Read data to for measurement error while input is |0>
    print('Write data for standard Bayesian')
    prop_dict = backend.properties().to_dict()
    with open(file_address + 'given_params.csv', mode='w', newline='') as sgm:
        param_writer = csv.writer(sgm,
                                  delimiter=',',
                                  quotechar='"',
                                  quoting=csv.QUOTE_MINIMAL)
        for q in interested_qubits:
            p0m0 = 1 - prop_dict['qubits'][q][5]['value']
            if p0m0 == 1 or p0m0 < 0.7:
                p0m0 = 0.9
            p1m1 = 1 - prop_dict['qubits'][q][4]['value']
            if p1m1 == 1 or p1m1 < 0.7:
                p1m1 = 0.9
            param_writer.writerow([p0m0, p1m1])

    with open(file_address + 'Filter_data.csv', mode='r') as measfile:
        reader = csv.reader(measfile)
        cali01 = np.asarray([row for row in reader][0])

    Data = cali01
    for q in interested_qubits:
        y = getData0(Data, itr * int(8192 / shots_per_point), q)
        with open(file_address + 'Qubit{}.csv'.format(q), mode='w',
                  newline='') as sgr:
            read_writer = csv.writer(sgr,
                                     delimiter=',',
                                     quotechar='"',
                                     quoting=csv.QUOTE_MINIMAL)
            read_writer.writerow(['x', 'y'])
            for i in range(len(y)):
                read_writer.writerow([0.5, y[i]])


def create_filters(interested_qubits,
                   QDT_correlated,
                   shots_per_point=1024,
                   seed=227,
                   show_denoised=False,
                   from_file=False,
                   file_address=''):
    """Return filters from our method, Qiskit method, QDt, and Standard Bayesian.
    
    Args:
      interested_qubits:
        an array of ints
      QDT_correlated:
        True if want qubits corrlected in QDT method.
      seed:
          seed for our method.
      file_address:
        file address, string ends with '/' if not empty

    Returns:
      our filer, qiskit method filter, QDt filter, and filter from Standard Bayesian
    """
    # Read Data from Standard Bayesian
    print('Standard Bayesian filter')
    post_dict = {}
    try:
        for q in interested_qubits:
            post_dict['Qubit{}'.format(q)] = pd.read_csv(
                file_address + 'StandPostQubit{}.csv'.format(q)).to_numpy()

        SB_filter = MeasFilterSB(interested_qubits, post_dict)
    except FileNotFoundError as e:
        raise Exception('Please run R code for Standard Bayesian Inference')

    print('Our Filter')
    mf = MeasFilter(interested_qubits, file_address=file_address)
    if from_file:
        mf.post_from_file()
    else:
        mf.inference(nPrior=40000, seed=seed,
                     show_denoised=show_denoised,
                     shots_per_point=shots_per_point)

    # Qiskit method
    print('Qiskit filter')
    cal_matrix = np.genfromtxt(file_address + 'cal_matrix.csv', delimiter=',')
    with open(file_address + 'state_labels.csv', mode='r') as sgm:
        reader = csv.reader(sgm)
        state_labels = np.asarray([row for row in reader][0])

    qiskit_filter = MeasurementFilter(cal_matrix, state_labels)
    #qiskit_filter = MeasFilterMat(cal_matrix)

    #QDT matrix
    print('QDT filter')
    if QDT_correlated:
        trans_matrix = np.genfromtxt(file_address + 'trans_matrix.csv',
                                     delimiter=',')
    else:
        trans_dict = {}
        for q in interested_qubits:
            trans_dict['Qubit{}'.format(q)] = np.genfromtxt(
                file_address + 'trans_matrix{}.csv'.format(q), delimiter=',')

        first = True
        for q in interested_qubits:
            if first:
                Mx = trans_dict['Qubit' + str(q)]
                first = False
            else:
                Mx = np.kron(Mx, trans_dict['Qubit' + str(q)])
        trans_matrix = Mx

    QDT_filter = MeasFilterQDT(trans_matrix)

    # # Read Data from Standard Bayesian
    # print('Standard Bayesian filter')
    # post_dict = {}
    # try:
    #     for q in interested_qubits:
    #         post_dict['Qubit{}'.format(q)] = pd.read_csv(file_address + 'StandPostQubit{}.csv'.format(q)).to_numpy()

    #     SB_filter = MeasFilterSB(interested_qubits,post_dict)
    # except FileNotFoundError as e:
    #     raise('Please run R code for Standard Bayesian Inference')

    return mf, qiskit_filter, QDT_filter, SB_filter


def QAOAexp(backend, file_address=''):
    """
        QAOA from https://arxiv.org/abs/1804.03719

    Parameters
    ----------
    backend : IBMQBackend
        backend.
    file_address : String, optional
        address for save data. The default is ''. Ends with "/" if not empty.

    Returns
    -------
    None.

    """
    pi = np.pi
    g1 = 0.2 * pi
    g2 = 0.4 * pi
    b1 = 0.15 * pi
    b2 = 0.05 * pi

    num = 5
    QAOA = QuantumCircuit(num, num)

    for i in range(1, 5):
        QAOA.h(i)
    QAOA.barrier()

    # k = 1
    QAOA.cx(3, 2)
    QAOA.u1(-g1, 2)
    QAOA.cx(3, 2)
    QAOA.barrier()

    QAOA.cx(4, 2)
    QAOA.u1(-g1, 2)
    QAOA.cx(4, 2)
    QAOA.barrier()

    QAOA.cx(1, 2)
    QAOA.u1(-g1, 2)
    QAOA.cx(1, 2)
    QAOA.cx(4, 3)
    QAOA.u1(-g1, 3)
    QAOA.cx(4, 3)
    QAOA.barrier()

    for i in range(1, 5):
        QAOA.u3(2 * b1, -pi / 2, pi / 2, i)
    QAOA.barrier()

    # k = 2
    QAOA.cx(3, 2)
    QAOA.u1(-g2, 2)
    QAOA.cx(3, 2)
    QAOA.barrier()

    QAOA.cx(4, 2)
    QAOA.u1(-g2, 2)
    QAOA.cx(4, 2)
    QAOA.barrier()

    QAOA.cx(1, 2)
    QAOA.u1(-g2, 2)
    QAOA.cx(1, 2)
    QAOA.cx(4, 3)
    QAOA.u1(-g2, 3)
    QAOA.cx(4, 3)
    QAOA.barrier()

    for i in range(1, 5):
        QAOA.u3(2 * b2, -pi / 2, pi / 2, i)
    QAOA.barrier()

    QAOA.barrier()
    QAOA.measure([1, 2, 3, 4], [1, 2, 3, 4])
    QAOA_trans = transpile(QAOA, backend, initial_layout=range(num))
    print('QAOA circuit depth is ', QAOA_trans.depth())

    # Run on simulator
    simulator = Aer.get_backend("qasm_simulator")
    simu_shots = 100000
    simulate = execute(QAOA, backend=simulator, shots=simu_shots)
    QAOA_results = simulate.result()
    with open(file_address + 'Count_QAOA_Simulator.csv', mode='w',
              newline='') as sgm:
        count_writer = csv.writer(sgm,
                                  delimiter=',',
                                  quotechar='"',
                                  quoting=csv.QUOTE_MINIMAL)
        for key, val in QAOA_results.get_counts().items():
            count_writer.writerow([key, val])

    # Run on real device
    shots = 8192
    job_exp = execute(QAOA_trans,
                      backend=backend,
                      shots=shots,
                      optimization_level=0)
    job_monitor(job_exp)
    exp_results = job_exp.result()
    with open(file_address + 'Count_QAOA.csv', mode='w', newline='') as sgm:
        count_writer = csv.writer(sgm,
                                  delimiter=',',
                                  quotechar='"',
                                  quoting=csv.QUOTE_MINIMAL)
        for key, val in exp_results.get_counts().items():
            count_writer.writerow([key, val])


def Groverexp(backend, file_address=''):
    """
        Gorver's search from https://arxiv.org/abs/1804.03719

    Parameters
    ----------
    backend : IBMQBackend
        backend.
    file_address : String, optional
        address for save data. The default is ''. Ends with "/" if not empty.

    Returns
    -------
    None.

    """
    num = 3
    Grover = QuantumCircuit(num, num)

    Grover.x(0)
    Grover.h(1)
    Grover.h(2)
    Grover.barrier()

    Grover.h(0)
    Grover.barrier()

    Grover.h(0)

    Grover.cx(1, 0)
    Grover.tdg(0)
    Grover.cx(2, 0)
    Grover.t(0)

    Grover.cx(1, 0)
    Grover.tdg(0)
    Grover.cx(2, 0)
    Grover.barrier()
    Grover.t(0)
    Grover.tdg(1)
    Grover.barrier()

    Grover.h(0)
    Grover.cx(2, 1)
    Grover.tdg(1)
    Grover.cx(2, 1)
    Grover.s(1)
    Grover.t(2)
    Grover.barrier()

    Grover.h(1)
    Grover.h(2)
    Grover.barrier()
    Grover.x(1)
    Grover.x(2)
    Grover.barrier()
    Grover.h(1)
    Grover.cx(2, 1)
    Grover.h(1)
    Grover.x(2)
    Grover.barrier()
    Grover.x(1)
    Grover.h(2)
    Grover.barrier()
    Grover.h(1)

    Grover.barrier()
    Grover.measure([1, 2], [1, 2])
    Grover_trans = transpile(Grover, backend, initial_layout=[0, 1, 2])
    print('Grover circuit depth is ', Grover_trans.depth())

    # Run on real device
    shots = 8192
    job_exp = execute(Grover_trans,
                      backend=backend,
                      shots=shots,
                      optimization_level=0)
    job_monitor(job_exp)
    exp_results = job_exp.result()
    with open(file_address + 'Count_Grover.csv', mode='w', newline='') as sgm:
        count_writer = csv.writer(sgm,
                                  delimiter=',',
                                  quotechar='"',
                                  quoting=csv.QUOTE_MINIMAL)
        for key, val in exp_results.get_counts().items():
            count_writer.writerow([key, val])


class MeasFilterSB:
    """Measurement error filter for standard Bayesian.

    Attributes:
        qubit_order: an array of ints, using order[LastQubit, ..., FirstQubit].
        post_dict: an dict of posteriors, {'Qubit1':[], 'Qubit2':[], ...}
    """
    def __init__(self, qubit_order, post_dict):
        self.qubit_order = qubit_order
        self.post = post_dict
        self.mat_mean = None
        self.mat_mode = None

    def mean(self):
        return np.array([
            np.mean(self.post['Qubit' + str(i)], axis=0)
            for i in self.qubit_order
        ]).reshape(len(self.qubit_order), 2)

    def mode(self):
        modes = []
        for q in self.qubit_order:
            modes.append(find_mode(self.post['Qubit' + str(q)][:, 0]))
            modes.append(find_mode(self.post['Qubit' + str(q)][:, 1]))
        return np.array(modes).reshape(len(self.qubit_order), 2)

    def filter_mean(self, counts):
        if self.mat_mean is None:
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
        if self.mat_mode is None:
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


class MeasFilterMat:
    """Measurement error filter Through a transition matrix.

    Attributes:
        trans_matrix: transition matrix
    """
    def __init__(self, trans_matrix):
        self.trans_mat = trans_matrix

    def filter(self, counts):
        shots = 0
        for key in counts:
            shots += counts[key]
        real_vec = dictToVec(len(list(counts.keys())[0]), counts) / shots

        proc_status, proc_vec = find_least_norm(
            len(list(counts.keys())[0]), nl.solve(self.trans_mat, real_vec))
        if proc_status != 'optimal':
            raise Exception('Sorry, filtering has failed')
        proc_counts = vecToDict(len(list(counts.keys())[0]), shots, proc_vec)

        return proc_counts


class MeasFilterQDT:
    """Measurement error filter Through a QDT.
       We create this class instead of using MeasFilterMat because
       the transition matrix contructed by QREM package follow the order
       [Qubit0, Qubit1, ...] instead of Qiskit register order 
       [Last Qubit, ..., First Qubit], so the key for counts should be reversed

    Attributes:
        trans_matrix: transition matrix
    """
    def __init__(self, trans_matrix):
        self.trans_mat = trans_matrix

    def filter(self, counts):
        shots = 0
        for key in counts:
            shots += counts[key]
        real_vec = dictToVec_inv(len(list(counts.keys())[0]), counts) / shots

        proc_status, proc_vec = find_least_norm(
            len(list(counts.keys())[0]), nl.solve(self.trans_mat, real_vec))
        if proc_status != 'optimal':
            raise Exception('Sorry, filtering has failed')
        proc_counts = vecToDict_inv(len(list(counts.keys())[0]), shots,
                                    proc_vec)

        return proc_counts
