import os
os.environ["ICTDIR"] = "/Users/user/IC"
import numpy  as np
import pandas as pd
import tables as tb
import matplotlib.pyplot as plt
import h5py
import statistics

from invisible_cities.io.channel_param_io import generator_param_reader
from invisible_cities.io.channel_param_io import subset_param_reader as spr
from invisible_cities.io.channel_param_io import basic_param_reader, parameters_and_errors
from functools import partial

# Horrible function to convert h5 file obtained from fitting script to a dataframe that i actually know how to use.
# Hidden here because Brais never wants to see it again

def obtain_stability_params(file_h5):
    if 'pmtCal' in file_h5:
        with h5py.File(file_h5, 'r') as f:

            dataset = f['/FITPARAMS/FIT_pmt_scaled_dark_pedestal']


            sensor_ids = dataset['SensorID']
            pedestal_sigma = dataset['pedestal_sigma']
            poisson_mu = dataset['poisson_mu']
            gain = dataset['gain']
            gain_sigma = dataset['gain_sigma']
            n_gaussians_chi2 = dataset['n_gaussians_chi2']
        

            final = {
                'SensorID': sensor_ids,
                'pedestal_sigma': pedestal_sigma[:, 0], 
                'u_pedestal_sigma': pedestal_sigma[:, 1],
                'poisson_mu': poisson_mu[:, 0],
                'u_poisson_mu': poisson_mu[:, 1],
                'gain': gain[:, 0],
                'u_gain': gain[:, 1],
                'gain_sigma': gain_sigma[:, 0],
                'u_gain_sigma': gain_sigma[:, 1],
                'n_gaussians': n_gaussians_chi2[:, 0],
                'chi2': n_gaussians_chi2[:, 1]
            }
    
            final_df = pd.DataFrame(final)
            return final_df
    
    if 'sipm_fit' in file_h5:
        with h5py.File(file_h5, 'r') as f:

            dataset = f['/FITPARAMS/FIT_sipm_signal']


            sensor_ids = dataset['SensorID']
            pedestal_sigma = dataset['pedestal_sigma']
            poisson_mu = dataset['poisson_mu']
            gain = dataset['gain']
            gain_sigma = dataset['gain_sigma']
            n_gaussians_chi2 = dataset['n_gaussians_chi2']
        

            final = {
                'SensorID': sensor_ids,
                'pedestal_sigma': pedestal_sigma[:, 0], 
                'u_pedestal_sigma': pedestal_sigma[:, 1],
                'poisson_mu': poisson_mu[:, 0],
                'u_poisson_mu': poisson_mu[:, 1],
                'gain': gain[:, 0],
                'u_gain': gain[:, 1],
                'gain_sigma': gain_sigma[:, 0],
                'u_gain_sigma': gain_sigma[:, 1],
                'n_gaussians': n_gaussians_chi2[:, 0],
                'chi2': n_gaussians_chi2[:, 1]
            }
    
            final_df = pd.DataFrame(final)
            return final_df

    if 'sipm_electronic' in file_h5:
        with h5py.File(file_h5, 'r') as f:

            dataset = f['/FITPARAMS/FIT_sipm_Gaussian']


            sensor_ids = dataset['SensorID']
            area = dataset['area']
            mu = dataset['mu']
            sigma = dataset['sigma']

            final = {
                'SensorID': sensor_ids,
                'area': area[:, 0], 
                'u_area': area[:, 1],
                'mu': mu[:, 0],
                'u_mu': mu[:, 1],
                'sigma': sigma[:, 0],
                'u_sigma': sigma[:, 1],
            }

            final_df = pd.DataFrame(final)

            final_df = final_df.sort_values(by='SensorID').reset_index(drop=True)

            return final_df

    if 'pmt_electronic' in file_h5:
        with h5py.File(file_h5, 'r') as f:
            dataset = f['/FITPARAMS/FIT_pmt_Gaussian']

            sensor_ids = dataset['SensorID']
            area = dataset['area']
            mu = dataset['mu']
            sigma = dataset['sigma']

            final = {
                'SensorID': sensor_ids,
                'area': area[:, 0], 
                'u_area': area[:, 1],
                'mu': mu[:, 0],
                'u_mu': mu[:, 1],
                'sigma': sigma[:, 0],
                'u_sigma': sigma[:, 1],
            }

            final_df = pd.DataFrame(final)

            final_df = final_df.sort_values(by='SensorID').reset_index(drop=True)

            return final_df
    
    if 'pmt_dark' in file_h5:
        with h5py.File(file_h5, 'r') as f:
            dataset = f['/FITPARAMS/FIT_pmt_Gaussian']

            sensor_ids = dataset['SensorID']
            area = dataset['area']
            mu = dataset['mu']
            sigma = dataset['sigma']

            final = {
                'SensorID': sensor_ids,
                'area': area[:, 0], 
                'u_area': area[:, 1],
                'mu': mu[:, 0],
                'u_mu': mu[:, 1],
                'sigma': sigma[:, 0],
                'u_sigma': sigma[:, 1],
            }

            final_df = pd.DataFrame(final)

            final_df = final_df.sort_values(by='SensorID').reset_index(drop=True)

            return final_df


# Function to calculate average gain of each dice board, given a complete list of gains.

def avg_gain_by_db(gains, errors, n_sipms=64):
    gains = np.array(gains)
    errors = np.array(errors)

    if gains.size == 0:
        return [], []

    gain_averages = []
    gain_averages_err = []
    n = len(gains)

    for i in range(0, n, n_sipms):
        dice = gains[i:i + n_sipms]
        err = errors[i:i + n_sipms]
        
        dice_average = dice.mean()
        dice_err = np.sqrt(np.sum(err**2)) / np.sqrt(n_sipms)

        gain_averages.append(dice_average)
        gain_averages_err.append(dice_err)
    
    return gain_averages, gain_averages_err


def weighted_av_std(values, weights):
    avg = np.average(values, weights=weights)
    var = np.average((values-avg)**2, weights=weights)
    # renormalize
    var = weights.sum() * var / (weights.sum()-1)
    return avg, np.sqrt(var)

def str2bool(v):
    """
    This function is added because the argparse add_argument('use_db_gain_seeds', type=bool)
    was not working in False case, everytime True was taken.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
