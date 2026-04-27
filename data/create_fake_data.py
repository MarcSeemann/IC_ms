import os
import sqlite3
import numpy as np
import pandas as pd
import tables as tb

def create_fake_hdf5(filename, num_events=10):
    """Creates a fake RWF HDF5 file with the requested fiber and SiPM structures."""
    
    # Parameters
    n_sipm = 832
    n_fiber = 36
    n_pmt = 12 # Minimum required to satisfy the pmt_wfs reader zip
    sipm_samples = 1600
    fiber_samples = 64000
    
    filters = tb.Filters(complevel=4, complib='blosc:lz4', bitshuffle=True)
    
    with tb.open_file(filename, "w", filters=filters) as h5out:
        # Create RD group
        rd_group = h5out.create_group(h5out.root, "RD")
        
        # Create EArrays for waveforms
        fib_hg = h5out.create_earray(rd_group, "fiberrwf_hg", tb.Int16Atom(), 
                                     shape=(0, n_fiber, fiber_samples), expectedrows=num_events)
        fib_lg = h5out.create_earray(rd_group, "fiberrwf_lg", tb.Int16Atom(), 
                                     shape=(0, n_fiber, fiber_samples), expectedrows=num_events)
        sipm   = h5out.create_earray(rd_group, "sipmrwf", tb.Int16Atom(), 
                                     shape=(0, n_sipm, sipm_samples), expectedrows=num_events)
        pmt    = h5out.create_earray(rd_group, "pmtrwf", tb.Int16Atom(), 
                                     shape=(0, n_pmt, fiber_samples), expectedrows=num_events)

        # Create Run metadata
        run_group = h5out.create_group(h5out.root, "Run")
        
        class RunInfo(tb.IsDescription):
            run_number = tb.Int32Col()
        run_info_table = h5out.create_table(run_group, "runInfo", RunInfo)
        row = run_info_table.row
        row['run_number'] = 1
        row.append()

        class EventInfo(tb.IsDescription):
            evt_number = tb.Int32Col()
            timestamp  = tb.UInt64Col()
        event_table = h5out.create_table(run_group, "events", EventInfo)

        # Create Trigger metadata
        trig_group = h5out.create_group(h5out.root, "Trigger")
        class Trigger(tb.IsDescription):
            trigger = tb.UInt32Col()
        class TriggerEvents(tb.IsDescription):
            events = tb.UInt32Col()
        
        trig_table = h5out.create_table(trig_group, "trigger", Trigger)
        trig_evt_table = h5out.create_table(trig_group, "events", TriggerEvents)

        # Fill with fake data
        for i in range(num_events):
            fib_hg.append(np.random.randint(0, 100, (1, n_fiber, fiber_samples)))
            fib_lg.append(np.random.randint(0, 10, (1, n_fiber, fiber_samples)))
            sipm.append(np.random.randint(0, 5, (1, n_sipm, sipm_samples)))
            pmt.append(np.zeros((1, n_pmt, fiber_samples)))
            
            row = event_table.row
            row['evt_number'] = i
            row['timestamp'] = 1600000000 + i
            row.append()
            
            t_row = trig_table.row
            t_row['trigger'] = 1
            t_row.append()
            
            te_row = trig_evt_table.row
            te_row['events'] = 1
            te_row.append()

    print(f"Created fake HDF5 data at: {filename}")

def create_fake_db(db_path):
    """Creates a local SQLite database with sensors and unit calibration."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    n_sipm = 832
    n_pmt, n_fiber = 12, 36

    # Create a simple 2D grid for X and Y
    side = int(np.ceil(np.sqrt(n_sipm)))
    x, y = np.meshgrid(np.arange(side) * 10, np.arange(side) * 10)
    x = x.flatten()[:n_sipm]
    y = y.flatten()[:n_sipm]

    # load_db.py expects SiPM IDs > 100 and PMT IDs < 100
    pmt_ids  = np.arange(n_pmt)
    sipm_ids = np.arange(101, 101 + n_sipm)
    fiber_ids = np.arange(1000, 1000 + n_fiber)
    all_ids  = np.concatenate([pmt_ids, sipm_ids, fiber_ids])

    # 1. ChannelPosition: Maps sensors to coordinates and labels
    df_pos = pd.DataFrame({
        'SensorID' : all_ids,
        'X'        : np.concatenate([np.zeros(n_pmt), x, np.zeros(n_fiber)]),
        'Y'        : np.concatenate([np.zeros(n_pmt), y, np.zeros(n_fiber)]),
        'Z'        : np.zeros(len(all_ids)),
        'Label'    : np.concatenate([['PMT']*n_pmt, ['SiPM']*n_sipm, ['Fiber']*n_fiber]),
        'MinRun'   : 0,
        'MaxRun'   : 999999
    })

    # 2. ChannelGain: Provides Centroid and Sigma
    df_gain = pd.DataFrame({
        'SensorID'      : all_ids,
        'Centroid'      : 1.0,
        'ErrorCentroid' : 0.0,
        'Sigma'         : 1.0,
        'ErrorSigma'    : 0.0,
        'MinRun'   : 0,
        'MaxRun'   : 999999
    })

    # 3. ChannelAmplification: Required for DataFiber
    df_amp = pd.DataFrame({
        'SensorID'      : all_ids,
        'Centroid'      : 1.0,
        'ErrorCentroid' : 0.0,
        'Sigma'         : 1.0,
        'ErrorSigma'    : 0.0,
        'MinRun'        : 0,
        'MaxRun'        : 999999
    })

    # 4. ChannelMapping: Maps SensorID to ElecID (ChannelID)
    df_map = pd.DataFrame({
        'SensorID' : all_ids,
        'ElecID'   : all_ids,
        'MinRun'   : 0,
        'MaxRun'   : 999999
    })

    # 5. PmtNoiseRms: Required for DataPMT
    df_noise = pd.DataFrame({
        'ElecID'    : pmt_ids,
        'noise_rms' : 0.5,
        'MinRun'    : 0,
        'MaxRun'    : 999999
    })

    # 6. PmtBlr: Required for DataPMT
    df_blr = pd.DataFrame({
        'ElecID'    : pmt_ids,
        'coeff_blr' : 0.001,
        'coeff_c'   : 0.0001,
        'MinRun'    : 0,
        'MaxRun'    : 999999
    })

    # 4. ChannelMask: For masking sensors (empty means all are active due to LEFT JOIN)
    df_mask = pd.DataFrame({
        'SensorID' : pd.Series(dtype='int'),
        'MinRun'   : pd.Series(dtype='int'),
        'MaxRun'   : pd.Series(dtype='int')
    })


    with sqlite3.connect(db_path) as conn:
        df_pos .to_sql("ChannelPosition", conn, index=False, if_exists="replace")
        df_gain.to_sql("ChannelGain"    , conn, index=False, if_exists="replace")
        df_amp .to_sql("ChannelAmplification", conn, index=False, if_exists="replace")
        df_map .to_sql("ChannelMapping" , conn, index=False, if_exists="replace")
        df_mask.to_sql("ChannelMask"    , conn, index=False, if_exists="replace")
        df_noise.to_sql("PmtNoiseRms"   , conn, index=False, if_exists="replace")
        df_blr  .to_sql("PmtBlr"        , conn, index=False, if_exists="replace")
    
    print(f"Created fake SQLite DB at: {db_path}")

if __name__ == "__main__":
    # Adjusted to your environment paths
    root = os.environ.get('ICTDIR', os.getcwd())
    hdf5_file = os.path.join(root, "data/fake_data.h5")
    sqlite_db = os.path.join(root, "invisible_cities", "database", "localdb.HDDEMODB.sqlite3")

    os.makedirs(os.path.join(root, "data"), exist_ok=True)
    create_fake_hdf5(hdf5_file)
    create_fake_db(sqlite_db)