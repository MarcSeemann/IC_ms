import sys
import numpy             as np
import tables            as tb
import matplotlib.pyplot as plt


from scipy.signal        import find_peaks_cwt
from scipy               import stats as scs
from functools           import partial
from enum                import auto
#from core.josh_calib_functions import str2bool
from matplotlib.backends.backend_pdf import PdfPages

from   invisible_cities.core  .stat_functions  import poisson_sigma
from   invisible_cities.calib  .calib_functions import seeds_and_bounds
from   invisible_cities.calib  .calib_functions import dark_scaler
from   invisible_cities.calib  .calib_functions import SensorType
from   invisible_cities.types .ic_types        import AutoNameEnumBase
from   invisible_cities.cities.components      import get_run_number
from   invisible_cities.evm.ic_containers import FitFunction

from   invisible_cities.database import load_db as DB

import invisible_cities.calib.spe_response   as speR
import invisible_cities.core.fit_functions  as fitf
import invisible_cities.io.channel_param_io as pIO

file_name = sys.argv[1] # Input file. This should be the spectra obtained from trude.
func_name = sys.argv[2] # Name of the function. This is here because the NEW scripts allowed you to choose multiple different functions. Somewhat redundant now and should always be "signal".
n_events = sys.argv[3] # Number of events during the run. This will be used in the fit.
use_db_gain_seeds = True if 'true' in sys.argv[4] else False # Decide whether to use values from the database as seeds for gain and sigma. 
plot_option = sys.argv[5] # Choose whether to view the spectra or to simply run over all channels automatically. 'y' for yes, 'n' for no.

db_file = 'next100' # Select next100 as the relevant database.

sipmIn = tb.open_file(file_name, 'r') # Open the trude file.
run_no    = get_run_number(sipmIn) # Obtain the run number. This is used in naming subsequent files.
channs    = DB.DataSiPM(db_file, run_no).SensorID.values # Gives you the channel numbers. N.B. this is the sensor ID and not the elec ID.
sens_type = SensorType.SIPM # Select SiPM data from the trude file.

bins   = np.array(sipmIn.root.HIST.sipm_spe_bins) # Obtain bin values.
specsL = np.array(sipmIn.root.HIST.sipm_spe).sum(axis=0) # Select the light data from the trude file.

Dbins = np.array(sipmIn.root.HIST.sipm_dark_bins) # Obtain dark bin values.
specsD = np.array(sipmIn.root.HIST.sipm_dark).sum(axis=0) # Select the dark data from the trude file.


outData = [] # Create an array for the output values.
outDict = {} # Create a dictionary for the output values.
nfchans = [] # Create an array to store the channels where the fit fails.

DoutData = [] # Create an array for the dark output values.
DoutDict = {} # Create a dictionary for the dark output values.
Dnfchans = [] # Create an array to store the channels where the fit fails, for the dark spectra.

out_file_name = 'sipm_fit_R' # Prefix of the output file name
out_file      = tb.open_file(out_file_name+str(run_no)+'.h5', 'w') # Name and write the output file
param_writer  = pIO.channel_param_writer(out_file, sensor_type='sipm', func_name= 'signal', param_names=pIO.generic_params) # Imported from IC. Write the parameters to the output file.

dark_out_file_name = 'dark_sipm_fit_R' # Prefix of the output file name
dark_out_file      = tb.open_file(dark_out_file_name+str(run_no)+'.h5', 'w') # Name and write the output file
dark_param_writer  = pIO.channel_param_writer(dark_out_file, sensor_type='sipm', func_name= 'signal', param_names=pIO.generic_params) # Imported from IC. Write the parameters to the output file.

def signal( xs, bl, A, gain, sigmaq, poismu, maxpercent=0.999999999): # Define the fit function
    
    poispeaks_pos = np.arange(0, scs.poisson.ppf(maxpercent, 1)) # Poisson peak position
    realpeaks_pos = gain * poispeaks_pos + bl # Positions of n_pe peaks
    realpeaks_amp = A * scs.poisson.pmf(poispeaks_pos, poismu) # Amplitude of n_pe peaks 
    result        = np.zeros_like(xs) # Create an array of zeros of the same shape as the data given to the fit.
    
    for i in range(len(poispeaks_pos)):
        result += realpeaks_amp[i] * scs.norm.pdf(xs, loc=realpeaks_pos[i], scale=sigmaq*np.sqrt(i+1))

    return result
    
if 'y' in plot_option:

    for ich in range(len(specsL)): # Loop over all channels
        #if ich%100==0: print(ich)
        b1 = 0 # Minimum index value
        b2 = len(specsL) # Maximum index value

        outDict[pIO.generic_params[-2]] = (bins[b1], bins[min(len(bins)-1, b2)]) # Assign minimum and maximum bins to the output dictionary

        seeds = np.zeros(5) # Create an array for seeds
        bounds = np.zeros((5,2)) # Create an array for bounds

        seeds[0] = 0 # Set the baseline seed to zero
        bounds[0] = (-5,5)
        seeds[1] = n_events # The scaling factor uses the number of events as an intial guess
        bounds[1] = (0, np.inf)
        bounds[2] = (0,50)
        bounds[3] = (0,np.inf)
        if use_db_gain_seeds == True: # Condition for gain and sigma when using database as seeds
            seeds[2] = DB.DataSiPM(db_file, run_no).adc_to_pes.values[ich] # Use database gain as seed
            seeds[3] = DB.DataSiPM(db_file, run_no).Sigma.values[ich] # Use database sigma as seed
        else: # Condition for not using database gain and sigma as seeds (not ideal)
            seeds[2] = 15 # Hardcode an estimate for the gain, change as required
            seeds [3] = 2.24 # Hardcode an estimate for sigma, change as required
        seeds[4] = 1 # Guess 1 as poisson mu
        bounds[4] = (0,10)
        bounds = np.array(list(zip(*bounds)))
    
        try:
            fit = fitf.fit(signal, bins[b1:b2], specsL[ich], seeds, sigma=specsL[ich]**0.5+0.1, bounds = bounds, maxfev = 1000000)
            chi  = fit.chi2 # chi2 of the fit
            outData.append([channs[ich], fit.values, fit.errors, chi]) # Append the fit channels values

            if fit.values[2] < 1:
                fit.values[2] = 1e10
        
            if fit.values[2] > 100:
                fit.values[2] = 1e10


        except:
            print(f'I failed on {ich} iteration. ')
            nfchans.append(channs[ich]) # Append the failed fit channels
            outData.append([channs[ich], [0, 0, 0, 0], [0, 0, 0, 0], 0]) # Add fails to the output data to ensure same length
            for param in pIO.generic_params:
                outDict[param] = (1e10, 1e10)
                param_writer(ich, outDict)
            continue

        outDict[pIO.generic_params[0]] = (fit.values[1], fit.errors[1]) # Add everything to the dictionary
        outDict[pIO.generic_params[1]] = (fit.values[4], fit.errors[4])
        outDict[pIO.generic_params[2]] = (fit.values[0], fit.errors[0])
        outDict[pIO.generic_params[4]]  = (fit.values[2]  , fit.errors[2])
        outDict[pIO.generic_params[5]]  = (fit.values[3], fit.errors[3])
        outDict[pIO.generic_params[-1]] = (fit.chi2)

        param_writer(channs[ich], outDict)

        plt.errorbar(bins[b1:b2], specsL[ich], xerr=0.5*np.diff(bins)[0], yerr=0.1, fmt='b.')
        plt.plot(bins[b1:b2], signal(bins[b1:b2], *fit.values), 'r', label = 'Optimised')
        plt.plot(bins[b1:b2], signal(bins[b1:b2], *seeds), 'g', label = 'Initial')
        plt.title('Spe response fit to channel '+str(channs[ich]) + 'chi2 = ' + str(chi))
        plt.xlabel('ADC')
        plt.ylabel('AU')
        plt.legend()
        
        plt.show()
        plt.pause(1000000)

        print("Window closed.")

        plt.pause()

        print("Still alive")

        break
        plt.clf()
        plt.close()

if 'n' in plot_option:

    print('Processing light spectra...')

    with PdfPages(f'/Users/user/Calibration_Outputs/TP_Light_Runs/{run_no}/sipm_spec_{run_no}.pdf') as pdf:

        for ich in range(len(specsL)): # Loop over all channels
            #if ich%100==0: print(ich)
            b1 = 0 # Minimum index value
            b2 = len(specsL) # Maximum index value

            outDict[pIO.generic_params[-2]] = (bins[b1], bins[min(len(bins)-1, b2)]) # Assign minimum and maximum bins to the output dictionary

            seeds = np.zeros(5) # Create an array for seeds
            bounds = np.zeros((5,2)) # Create an array for bounds

            seeds[0] = 0 # Set the baseline seed to zero
            bounds[0] = (-5,5)
            seeds[1] = n_events # The scaling factor uses the number of events as an intial guess
            bounds[1] = (0, np.inf)
            bounds[2] = (0,100)
            bounds[3] = (0,np.inf)
            if use_db_gain_seeds == True: # Condition for gain and sigma when using database as seeds
                seeds[2] = DB.DataSiPM(db_file, run_no).adc_to_pes.values[ich] # Use database gain as seed
                seeds[3] = DB.DataSiPM(db_file, run_no).Sigma.values[ich] # Use database sigma as seed
            else: # Condition for not using database gain and sigma as seeds (not ideal)
                seeds[2] = 15 # Hardcode an estimate for the gain, change as required
                seeds [3] = 2.24 # Hardcode an estimate for sigma, change as required
            seeds[4] = 1 # Guess 1 as poisson mu
            bounds[4] = (0,10)
            bounds = np.array(list(zip(*bounds)))
    
            try:
                fit = fitf.fit(signal, bins[b1:b2], specsL[ich], seeds, sigma=specsL[ich]**0.5+0.1, bounds = bounds, maxfev = 1000000)
                chi  = fit.chi2 # chi2 of the fit
                outData.append([channs[ich], fit.values, fit.errors, chi]) # Append the fit channels values

                if fit.values[2] < 1:
                    fit.values[2] = 1e10
        
                if fit.values[2] > 100:
                    fit.values[2] = 1e10


            except:
                print(f'I failed on {ich} iteration. ')
                nfchans.append(channs[ich]) # Append the failed fit channels
                outData.append([channs[ich], [0, 0, 0, 0], [0, 0, 0, 0], 0]) # Add fails to the output data to ensure same length
                for param in pIO.generic_params:
                    outDict[param] = (1e10, 1e10)
                param_writer(channs[ich], outDict)
                continue

            outDict[pIO.generic_params[0]] = (fit.values[1], fit.errors[1]) # Add everything to the dictionary
            outDict[pIO.generic_params[1]] = (fit.values[4], fit.errors[4])
            outDict[pIO.generic_params[2]] = (fit.values[0], fit.errors[0])
            outDict[pIO.generic_params[4]]  = (fit.values[2]  , fit.errors[2])
            outDict[pIO.generic_params[5]]  = (fit.values[3], fit.errors[3])
            outDict[pIO.generic_params[-1]] = (fit.chi2)

            param_writer(channs[ich], outDict)

            fig, ax = plt.subplots()
            
            ax.errorbar(bins[b1:b2], specsL[ich], xerr=0.5 * np.diff(bins)[0], yerr=0.1, fmt='b.')
            ax.plot(bins[b1:b2], signal(bins[b1:b2], *fit.values), 'r', label='Optimised')
            ax.plot(bins[b1:b2], signal(bins[b1:b2], *seeds), 'g', label='Initial')
            ax.set_title(f'Spe response fit to channel {channs[ich]} chi2 = {chi}')
            ax.set_xlabel('ADC')
            ax.set_ylabel('AU')
            ax.legend()

            pdf.savefig(fig)
            plt.close(fig)

            print(f"Processing SensorID {channs[ich]}")
    
    print('Light spectra finished. Processing dark spectra...')

    with PdfPages(f'/Users/user/Calibration_Outputs/TP_Light_Runs/{run_no}/sipm_dark_{run_no}.pdf') as pdf:

        for ich in range(len(specsD)): # Loop over all channels
            b1 = 0 # Minimum index value
            b2 = len(specsD) # Maximum index value

            DoutDict[pIO.generic_params[-2]] = (Dbins[b1], Dbins[min(len(Dbins)-1, b2)]) # Assign minimum and maximum bins to the output dictionary

            seeds = np.zeros(5) # Create an array for seeds
            bounds = np.zeros((5,2)) # Create an array for bounds

            seeds[0] = 0 # Set the baseline seed to zero
            bounds[0] = (-5,5)
            seeds[1] = n_events # The scaling factor uses the number of events as an intial guess
            bounds[1] = (0, np.inf)
            bounds[2] = (0,50)
            bounds[3] = (0,np.inf)
            if use_db_gain_seeds == True: # Condition for gain and sigma when using database as seeds
                seeds[2] = DB.DataSiPM(db_file, run_no).adc_to_pes.values[ich] # Use database gain as seed
                seeds[3] = DB.DataSiPM(db_file, run_no).Sigma.values[ich] # Use database sigma as seed
            else: # Condition for not using database gain and sigma as seeds (not ideal)
                seeds[2] = 15 # Hardcode an estimate for the gain, change as required
                seeds [3] = 2.24 # Hardcode an estimate for sigma, change as required
            seeds[4] = 1 # Guess 1 as poisson mu
            bounds[4] = (0,10)
            bounds = np.array(list(zip(*bounds)))
    
            try:
                fit = fitf.fit(signal, Dbins[b1:b2], specsD[ich], seeds, sigma=specsD[ich]**0.5+0.1, bounds = bounds, maxfev = 1000000)
                chi  = fit.chi2 # chi2 of the fit
                DoutData.append([channs[ich], fit.values, fit.errors, chi]) # Append the fit channels values

                if fit.values[2] < 1:
                    fit.values[2] = 0
        
                if fit.values[2] > 100:
                    fit.values[2] = 0


            except:
                print(f'I failed on {ich} iteration. ')
                Dnfchans.append(channs[ich]) # Append the failed fit channels
                DoutData.append([channs[ich], [0, 0, 0, 0], [0, 0, 0, 0], 0]) # Add fails to the output data to ensure same length
                for param in pIO.generic_params:
                    DoutDict[param] = (0, 0)
                dark_param_writer(channs[ich], DoutDict)
                continue

            DoutDict[pIO.generic_params[0]] = (fit.values[1], fit.errors[1]) # Add everything to the dictionary
            DoutDict[pIO.generic_params[1]] = (fit.values[4], fit.errors[4])
            DoutDict[pIO.generic_params[2]] = (fit.values[0], fit.errors[0])
            DoutDict[pIO.generic_params[4]]  = (fit.values[2]  , fit.errors[2])
            DoutDict[pIO.generic_params[5]]  = (fit.values[3], fit.errors[3])
            DoutDict[pIO.generic_params[-1]] = (fit.chi2)

            dark_param_writer(channs[ich], DoutDict)

            fig, ax = plt.subplots()
            
            ax.errorbar(Dbins[b1:b2], specsD[ich], xerr=0.5 * np.diff(Dbins)[0], yerr=0.1, fmt='b.')
            ax.plot(Dbins[b1:b2], signal(Dbins[b1:b2], *fit.values), 'r', label='Optimised')
            ax.plot(Dbins[b1:b2], signal(Dbins[b1:b2], *seeds), 'g', label='Initial')
            ax.set_title(f'Dark response fit to channel {channs[ich]} chi2 = {chi}')
            ax.set_xlabel('ADC')
            ax.set_ylabel('AU')
            ax.legend()

            pdf.savefig(fig)
            plt.close(fig)

            print(f"Processing dark data, SensorID {channs[ich]}")



