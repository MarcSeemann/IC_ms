import sys
import numpy             as np
import tables            as tb
import matplotlib.pyplot as plt
import math

from functools import partial
from josh_calib_functions import weighted_av_std
from matplotlib.backends.backend_pdf import PdfPages

import invisible_cities.core.fit_functions   as fitf
import invisible_cities.calib.spe_response    as speR
import invisible_cities.io.channel_param_io  as pIO

from   invisible_cities.database             import load_db           as DB
from   invisible_cities.calib.calib_functions import seeds_and_bounds
from   invisible_cities.calib.calib_functions import dark_scaler
from   invisible_cities.calib.calib_functions import SensorType
from   invisible_cities.types.ic_types       import AutoNameEnumBase
from   invisible_cities.cities.components    import get_run_number

def main():

    file_name = sys.argv[1]
    func_name = sys.argv[2]
    min_stat  = 0
    #fix_ped = False
    # Make arguments 3 and 4 optional
    if len(sys.argv) > 3:
        use_db_gain_seeds = True if 'true' in sys.argv[3] else False
        min_stat = int(sys.argv[4])
    dats   = tb.open_file(file_name, 'r')
    bins   = np.array(dats.root.HIST.pmt_dark_bins)
    specsD = np.array(dats.root.HIST.pmt_dark).sum(axis=0)
    specsL = np.array(dats.root.HIST.pmt_spe).sum(axis=0)
    
    run_no      = get_run_number(dats)
    print(run_no)
    sensor_type = SensorType.PMT
    # Only using one function currently, the rest annoyed me
    ffuncs = {'dfunc':partial(speR.scaled_dark_pedestal, min_integral=100)}
    pOrders = {'dfunc':'norm err poismu err gain err 1peSig err'}

    fnam = {'dfunc':'scaled_dark_pedestal'}

    pOut = tb.open_file(f'pmtCalParOut_R'+str(run_no)+f'_min_{min_stat}.h5', 'w')

    param_writer = pIO.channel_param_writer(pOut,
                                            sensor_type='pmt',
                                            func_name=fnam[func_name],
                                            param_names=pIO.generic_params)
    
    outDict = {}

    failed_channels = []
    plot_option = input("Press y to view the spectra: ")
    if 'y' in plot_option:
        for ich, (dspec, lspec) in enumerate(zip(specsD, specsL)):

            try:
                b1 = 0
                b2 = len(dspec)
                if min_stat != 0:
                    valid_bins = np.argwhere(lspec>=min_stat)
                    b1 = valid_bins[0][0]
                    b2 = valid_bins[-1][0]

                outDict[pIO.generic_params[-2]] = (bins[b1], bins[min(len(bins)-1, b2)])

                gb0 = [(0, -100, 0), (1e99, 200, 10000)]
                av, rms = weighted_av_std(bins[dspec>100], dspec[dspec>100])
                sd0 = (dspec.sum(), av, rms)
                errs = np.sqrt(dspec[dspec>100])
                errs[errs==0] = 0.0001
                gfitRes = fitf.fit(fitf.gauss, bins[dspec>100], dspec[dspec>100], sd0, sigma=errs, bounds=gb0, maxfev = 1000000)
                outDict[pIO.generic_params[2]] = (gfitRes.values[1], gfitRes.errors[1])
                outDict[pIO.generic_params[3]] = (gfitRes.values[2], gfitRes.errors[2])

                scale = lspec.sum() / dspec.sum()

                respF = ffuncs[func_name](dark_spectrum=dspec[b1:b2] * scale,
                                            pedestal_mean=gfitRes.values[1],
                                            pedestal_sigma=gfitRes.values[2])

                ped_vals = np.array([gfitRes.values[0] * scale, gfitRes.values[1], gfitRes.values[2]])
                dark = dspec[b1:b2]
                scaler_func = dark_scaler(dark[bins[b1:b2]<0])



                seeds, bounds = seeds_and_bounds(sensor_type, run_no, ich, scaler_func, np.array(bins[b1:b2]),
                                                lspec[b1:b2], ped_vals, 'next100', gfitRes.errors,
                                                func='dfunc', use_db_gain_seeds=False)
            
                seeds_ = []
                seeds_.append(seeds[0])
                seeds_.append(seeds[1])
                if seeds[2]< 0:
                    seeds_.append(50)

                else:
                    seeds_.append(seeds[2])
                seeds_.append(seeds[3])
                errs = np.sqrt(lspec[b1:b2])
                #errs = np.sqrt(errs**2 + np.exp(-2*seeds[1]) * dspec[b1:b2])
                errs[errs==0] = 0.001

                rfit = fitf.fit(respF, bins[b1:b2], lspec[b1:b2], seeds_, sigma=errs, bounds=bounds, maxfev = 1000000)

                plt.errorbar(bins, lspec, xerr=0.5*np.diff(bins)[0], yerr=np.sqrt(lspec), fmt='b.')
                plt.plot(bins[b1:b2], rfit.fn(bins[b1:b2]), 'r', label = 'Optimised fit')
                plt.plot(bins[b1:b2], respF(bins[b1:b2], *seeds), 'g', label = 'Seeds')
                plt.title('Spe response fit to channel '+str(ich)+' chi2 = '+str(rfit.chi2))
                plt.xlabel('ADC')
                plt.ylabel('AU')
                plt.grid()
                plt.legend()
                plt.show(block=False)

                outDict[pIO.generic_params[0]] = (rfit.values[0], rfit.errors[0])
                outDict[pIO.generic_params[1]] = (rfit.values[1], rfit.errors[1])
                outDict[pIO.generic_params[4]] = (rfit.values[2], rfit.errors[2])
                outDict[pIO.generic_params[5]] = (rfit.values[3], rfit.errors[3])
                outDict[pIO.generic_params[-1]] = (respF.n_gaussians, rfit.chi2)
                param_writer(ich, outDict)
            except Exception as e:
                print("Error occurred in fitting for channel {}: {}".format(ich, e))
                failed_channels.append(ich)

                for param in pIO.generic_params:
                    outDict[param] = (1e10, 1e10)
            
                param_writer(ich, outDict)

        print("Channels where fit failed:", failed_channels)
        pOut.close()

    else:
        #with PdfPages(f'/Users/user/Documents/NEXT/Calibration/NEXT-100/EP/{run_no}/plots/pmt_spec_{run_no}_min_{min_stat}.pdf') as pdf:
        with PdfPages(f'/Users/user/Calibration_Outputs/EP_Light_Runs/{run_no}/pmt_spec_{run_no}_min_{min_stat}.pdf') as pdf:
            for ich, (dspec, lspec) in enumerate(zip(specsD, specsL)):
                print(ich)
                try:
                    b1 = 0
                    b2 = len(dspec)
                    if min_stat != 0:
                        valid_bins = np.argwhere(lspec>=min_stat)
                        b1 = valid_bins[0][0]
                        b2 = valid_bins[-1][0]

                    outDict[pIO.generic_params[-2]] = (bins[b1], bins[min(len(bins)-1, b2)])

                    gb0 = [(0, -100, 0), (1e99, 200, 10000)]
                    av, rms = weighted_av_std(bins[dspec>100], dspec[dspec>100])
                    sd0 = (dspec.sum(), av, rms)
                    errs = np.sqrt(dspec[dspec>100])
                    errs[errs==0] = 0.0001
                    print('Test 1 ', list(zip(sd0, *gb0)))
                    gfitRes = fitf.fit(fitf.gauss, bins[dspec>100], dspec[dspec>100], sd0, sigma=errs, bounds=gb0)# maxfev = 1000000)
                    print('Test 2 ')
                    outDict[pIO.generic_params[2]] = (gfitRes.values[1], gfitRes.errors[1])
                    outDict[pIO.generic_params[3]] = (gfitRes.values[2], gfitRes.errors[2])

                    scale = lspec.sum() / dspec.sum()

                    respF = ffuncs[func_name](dark_spectrum=dspec[b1:b2] * scale,
                                            pedestal_mean=gfitRes.values[1],
                                            pedestal_sigma=gfitRes.values[2])
                    ped_vals = np.array([gfitRes.values[0] * scale, gfitRes.values[1], gfitRes.values[2]])
                    dark = dspec[b1:b2]
                    scaler_func = dark_scaler(dark[bins[b1:b2]<0])


                    seeds, bounds = seeds_and_bounds(sensor_type, run_no, ich, scaler_func, np.array(bins[b1:b2]),
                                                lspec[b1:b2], ped_vals, 'next100', gfitRes.errors,
                                                func='dfunc', use_db_gain_seeds=use_db_gain_seeds)
                    print('Test 3 ')            
                    seeds_ = []
                    seeds_.append(seeds[0])
                    seeds_.append(seeds[1])
                    if seeds[2]< 0:
                        seeds_.append(50)

                    else:
                        seeds_.append(seeds[2])
                    seeds_.append(seeds[3])
                    errs = np.sqrt(lspec[b1:b2])
                    #errs = np.sqrt(errs**2 + np.exp(-2*seeds[1]) * dspec[b1:b2])
                    errs[errs==0] = 0.001
                    print('hiii ', list(zip(seeds_, *bounds)))
                    rfit = fitf.fit(respF, bins[b1:b2], lspec[b1:b2], seeds_, sigma=errs, bounds=bounds)#maxfev = 1000000)
                    print('hiii ')
                    outDict[pIO.generic_params[0]] = (rfit.values[0], rfit.errors[0])
                    outDict[pIO.generic_params[1]] = (rfit.values[1], rfit.errors[1])
                    outDict[pIO.generic_params[4]] = (rfit.values[2], rfit.errors[2])
                    outDict[pIO.generic_params[5]] = (rfit.values[3], rfit.errors[3])
                    outDict[pIO.generic_params[-1]] = (respF.n_gaussians, rfit.chi2)
                    param_writer(ich, outDict)

                    fig, ax = plt.subplots()

                    ax.errorbar(bins, lspec, xerr=0.5*np.diff(bins)[0], yerr=np.sqrt(lspec), fmt='b.')
                    ax.plot(bins[b1:b2], rfit.fn(bins[b1:b2]), 'r', label = 'Optimised fit')
                    ax.plot(bins[b1:b2], respF(bins[b1:b2], *seeds), 'g', label = 'Seeds')
                    ax.axvline(rfit.values[2]+rfit.values[1], color = 'y', linestyle = '--', label = 'gain')
                    ax.set_title('Spe response fit to channel '+str(ich)+' chi2 = '+str(rfit.chi2))
                    ax.set_xlabel('ADC')
                    ax.set_ylabel('AU')
                    ax.grid()
                    ax.legend()

                    pdf.savefig(fig)
                    plt.close(fig)

                except Exception as e:
                    print("Error occurred in fitting for channel {}: {}".format(ich, e))
                    failed_channels.append(ich)

                    fig, ax = plt.subplots()

                    ax.errorbar(bins, lspec, xerr=0.5*np.diff(bins)[0], yerr=np.sqrt(lspec), fmt='b.')
                    ax.set_title('Fit failed to channel '+str(ich))
                    ax.set_xlabel('ADC')
                    ax.set_ylabel('AU')
                    ax.grid()
                    ax.legend()

                    pdf.savefig(fig)
                    plt.close(fig)

                    for param in pIO.generic_params:
                        outDict[param] = (0, 0)
            
                    param_writer(ich, outDict)

            print("Channels where fit failed:", failed_channels)
            pOut.close()


if __name__ == '__main__':
    main()

