import warnings
import tempfile
import os
import numpy as np
import pandas as pd
from pathlib import Path

from tvb.simulator.lab import (coupling, monitors, simulator,
                               noise, integrators, models, connectivity, cortex)
from tvb.simulator.models.epileptor import Epileptor
from tvb.simulator.models.epileptor_rs import EpileptorRestingState

def load_subject_data(root):
    pass


def simplest_gain_matrix(conn, ch_xyz):
    '''
    This is a function to recompute a new gain matrix based on xyz that moved
    G = 1 / ( 4*pi * sum(sqrt(( X - X[:, new])^2))^2)
    '''
    # NOTE IF YOU MOVE SEEGXYZ ONTO REGXYZ, YOU DIVIDE BY 0, SO THERE IS A PROBLEM
    # reg_xyz = con.centres
    dr = conn.centres - ch_xyz[:, np.newaxis]
    print("Computing simple gain mat!")
    if 0 in dr:
        print("Computing simplest gain matrix will result \
            in error when contact is directly on top of any region!\
            Dividing by 0!")

    ndr = np.sqrt((dr ** 2).sum(axis=-1))
    Vr = 1.0 / (4 * np.pi) / (1 + ndr ** 2)
    return Vr


def gain_matrix_inv_square(vertices, areas, conn, ch_xyz, region_mapping):
    '''
    Computes a gain matrix using an inverse square fall off (like a mean field model)

    Parameters
    ----------
    vertices             np.ndarray of floats of size n x 3, where n is the number of vertices
    areas                np.ndarray of floats of size n x 3
    region_mapping       np.ndarray of ints of size n
    nregions             int of the number of regions
    sensors              np.ndarray of floats of size m x 3, where m is the number of sensors

    Returns
    -------
    np.ndarray of size m x n
    '''
    nregions = conn.region_labels.shape[0]
    nverts = vertices.shape[0]
    nsens = ch_xyz.shape[0]
    reg_map_mtx = np.zeros((nverts, nregions), dtype=int)
    for i, region in enumerate(region_mapping):
        if region >= 0:
            reg_map_mtx[i, region] = 1
    gain_mtx_vert = np.zeros((nsens, nverts))
    for sens_ind in range(nsens):
        a = ch_xyz[sens_ind, :] - vertices
        na = np.sqrt(np.sum(a ** 2, axis=1))

        # original version
        gain_mtx_vert[sens_ind, :] = areas / (na ** 2)

        # To Do: Refactor to use a more physically accurate way to project source activity
        # adding a 1 in the denominator to softmax the gain matrix
        softmax_inds = np.where(na < 1)[0]
        if len(softmax_inds) > 0:
            # print("na was less than one, so softmaxing here at 1.")
            # epsilon = 1 - a
            # na = np.sqrt(np.sum(a**2, axis=1))
            gain_mtx_vert[sens_ind, softmax_inds] = areas[softmax_inds] / \
                                                    (1 + na[softmax_inds] ** 2)

    return gain_mtx_vert.dot(reg_map_mtx)


def load_subject_connectivity(source_file=None, condspeed: float = np.inf):
    if source_file is not None:
        input_kwargs = {'source_file': source_file}
    else:
        input_kwargs = {}
    conn = connectivity.Connectivity.from_file(**input_kwargs)

    condspeed = np.array(condspeed)
    conn.speed = condspeed
    conn.cortical[:] = True

    # initialize weights
    conn.weights = conn.weights / np.max(conn.weights)
    return conn


def resect_regions(conn: connectivity.Connectivity, indices):
    conn.weights[indices] = 0.
    conn.weights[:, indices] = 0.
    return conn


def setup_simulation(conn, ch_xyz, ezind=None, pzind=None,
                     ts=0.05, noise_cov=None, integration_noise=True,
                     coupling_strength=1., period=1., gain_fname=None,
                     resting_state=False, variables_of_interest=None,
                     initial_conditions=None, seeg_fname=None,
                     resected_chs=None,
                     **epileptor_kwargs):
    ####################### 2. Neural Mass Model @ Nodes ######################
    if resected_chs is None:
        resected_chs = []
    x0norm = np.array(epileptor_kwargs.get('x0norm', -2.4))
    x0ez = np.array(epileptor_kwargs.get('x0ez', -1.9))
    x0pz = np.array(epileptor_kwargs.get('x0pz', -2.04))
    r = np.array(epileptor_kwargs.get('r', 0.00035 / 7))  # Temporal scaling in the third state variable
    Ks = np.array(epileptor_kwargs.get('Ks', -0.5))  # Slow Permittivity coupling, fast to slow time scale
    Kf = np.array(epileptor_kwargs.get('Kf', 0.1))  # Permittivity coupling of spike wave events
    Kvf = np.array(epileptor_kwargs.get('Kvf', 0.0))  # Permittivity coupling of spike wave events
    tt = np.array(epileptor_kwargs.get('tt', 0.10))  # time scale of simulation
    tau = np.array(epileptor_kwargs.get('tau', 10))  # Temporal scaling coefficient in fifth st var

    if variables_of_interest is None:
        variables_of_interest = ['z', 'x2 - x1']
    if resting_state:
        epileptors = EpileptorRestingState(
            variables_of_interest=variables_of_interest, Kvf=Kvf, Kf=Kf)
    else:
        epileptors = Epileptor(
            variables_of_interest=variables_of_interest,
            Kvf=Kvf, Kf=Kf)

    # integration noise
    if noise_cov is None:
        noise_cov = np.array([0.001, 0.001, 0.,
                              0.0001, 0.0001, 0.])
        if resting_state:
            noise_cov = np.append(noise_cov, [0.0001, 0.])

    if r is not None:
        epileptors.r = r
    if Ks is not None:
        epileptors.Ks = Ks
    if tt is not None:
        epileptors.tt = tt
    if tau is not None:
        epileptors.tau = tau
    if ezind is None:
        ezind = []
    if pzind is None:
        pzind = []

    # this comes after setting all parameters
    epileptors.x0 = x0norm * np.ones(len(conn.region_labels))
    if x0ez is not None:
        try:
            epileptors.x0[ezind] = x0ez
        except AttributeError:
            warnings.warn(
                "EZ index not set yet! Do you want to proceed with simulation?")
    if x0pz is not None:
        try:
            epileptors.x0[pzind] = x0pz
        except AttributeError:
            warnings.warn(
                "pz index not set yet! Do you want to proceed with simulation?")

    if resting_state:
        # set the coupling state between the epileptor and the resting state 2D oscillator
        epileptors.p = np.ones(len(conn.region_labels)) * 0.2

        epileptors.a_rs = np.ones(len(conn.region_labels)) * 1.7402

    ####################### 3. Integrator for Models ##########################

    # define cov noise for the stochastic heun integrato
    hiss = noise.Additive(nsig=noise_cov)
    # hiss = noise.Multiplicative(nsig=noise_cov)
    # hiss = noise.Additive(nsig=noise_cov, ntau=ntau)

    # adding observation noise?
    ntau = 0.1  # color of noise?
    obs_noise_cov = np.array([0, 1.0])  # cov of noise
    obsnoise = noise.Additive(nsig=obs_noise_cov, ntau=ntau)

    if integration_noise:
        heunint = integrators.HeunStochastic(dt=ts, noise=hiss)
    else:
        heunint = integrators.HeunDeterministic(dt=ts)
    integrator = heunint

    # either use gain file, or recompute it
    usegainfile = True
    if usegainfile:
        gainfile = gain_fname
    else:
        gainfile = None

    ################## 4. Difference Coupling Between Nodes ###################
    coupling_strength = np.array(coupling_strength)
    coupl = coupling.Difference(a=coupling_strength)

    ############## 5. Import Sensor XYZ, Gain Matrix For Monitors #############
    mon_tavg = monitors.TemporalAverage(period=period)  # monitor model

    # if we are also resecting the channels
    if resected_chs != []:
        tempdir = tempfile.TemporaryDirectory()
        temp_path = Path(tempdir.name)
        temp_path.mkdir(exist_ok=True, parents=True)

        # load in the current sensors fname
        # and re-save it as a temporary file
        seeg_pd = pd.read_csv(seeg_fname, names=['x', 'y', 'z'], delim_whitespace=True)

        # load in the gain file and remove the indices
        ch_names = seeg_pd.index
        seeg_pd.drop(resected_chs, inplace=True)

        # resave the dataframe
        new_seeg_fname = os.path.join(temp_path, os.path.basename(seeg_fname))
        seeg_pd.to_csv(new_seeg_fname, sep=' ', header=None, columns=None)

        # modify the gain file
        keep_inds = [idx for idx, ch in enumerate(ch_names) if ch not in resected_chs]
        resected_inds = [idx for idx, ch in enumerate(ch_names) if ch in resected_chs]
        new_gain_file = os.path.join(temp_path, os.path.basename(gainfile))

        # load gain matrix and drop the channel indices
        gain_arr = np.loadtxt(gainfile)
        gain_arr = np.delete(gain_arr, resected_inds, axis=0)

        # resave trhe gain matrix
        np.savetxt(new_gain_file, gain_arr)

        seeg_fname = new_seeg_fname
        gainfile = new_gain_file

    if gainfile is None:
        mon_SEEG = monitors.iEEG.from_file(period=period,
                                           obsnoise=obsnoise,
                                           variables_of_interest=np.array([7, 8]),
                                           sensors_fname=str(seeg_fname),
                                           )
        # sensors_fname=self.seegfile,
        # rm_f_name=regmapfile,
        # projection_fname=gainfile)
        # gainmat = gain_matrix_inv_square()
        gainmat = simplest_gain_matrix(conn, ch_xyz)
        # self.gainmat = self.simplest_gain_matrix()
        mon_SEEG.gain = gainmat
    else:
        mon_SEEG = monitors.iEEG.from_file(period=period,
                                           variables_of_interest=np.array([7, 8]),
                                           sensors_fname=str(seeg_fname),
                                           # sensors_fname=,
                                           # rm_f_name=regmapfile,
                                           obsnoise=obsnoise,
                                           projection_fname=gainfile)

    # add scalp EEG monitor
    # mon_EEG = monitors.EEG.from_file(period=period)
    # aggregate monitors for temporal epileptor dynamics and sEEG projections
    sim_monitors = [mon_tavg, mon_SEEG]

    # initialize simulators
    print(sim_monitors)
    sim = simulator.Simulator(model=epileptors,
                              initial_conditions=initial_conditions,
                              connectivity=conn,
                              coupling=coupl,
                              integrator=integrator,
                              monitors=sim_monitors)
    if resected_chs != []:
        tempdir.cleanup()
    sim.configure()
    return sim


def runsim(sim: simulator.Simulator, sim_length: int = 60000):
    (times, epilepts), (_, seegts) = sim.run(
        simulation_length=sim_length)
    return times, epilepts, seegts
