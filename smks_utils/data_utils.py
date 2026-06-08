#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data processing utilities for seismic data preparation and manipulation.

This module contains functions for converting, aligning, and processing
seismic waveform data from various formats.

Created on April 26, 2026
@author: zhouyangtianli
"""

import os
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from obspy import read
from obspy.geodetics import locations2degrees
import pyproj
from obspy.signal.filter import bandpass
from .visualization_utils import plot_two_data_matrix, plot_diff_data_matrix, plot_corrcoef
import shutil
from obspy import UTCDateTime


# Initialize geodetic object
projection = pyproj.Geod(ellps='WGS84')

def read_hypo(fname):
    hypo = pd.read_csv(fname)
    source_info = {'evtid': hypo.evtid[0],
                    'evla': hypo.evla[0], 'evlo': hypo.evlo[0], 'evdp': hypo.evdp[0],
                    'otime': UTCDateTime(hypo.year[0], hypo.month[0], hypo.day[0],
                                         hypo.hour[0], hypo.minute[0], hypo.second[0])}
    return source_info


def move_repeat_data_files(data_directory):
    """
    Usage: 
        data_directory = '../Data/2011-11-22-mw66-central-bolivia-6'
        move_repeat_data_files(data_directory)
    """
    file_dir_list = glob('%s/SmKS/*R' % data_directory)
    repeat_dir = '%s/SmKS/repeat' % data_directory
    os.makedirs(repeat_dir, exist_ok=True)
    uniq_file_list = []
    for file_dir in file_dir_list:
        fname = file_dir.split('/')[-1]
        station_info = fname.split('.')
        network = station_info[0]
        station = station_info[1]
        location = station_info[2]
        if location != 'BHR':
            station_name = '%s.%s.%s' % (network, station, location)
        else:
            station_name = '%s.%s.' % (network, station)
        print(station_name)
        if station_name in uniq_file_list:
            shutil.move(file_dir, repeat_dir)
            print('move %s -> %s' % (file_dir, repeat_dir))
        else:
            uniq_file_list.append(station_name)



def matrix_align_slant(data_matrix, sample_rate, signal_begin,
                       half_window, before_pick, after_pick):

    (M, N) = data_matrix.shape
    signal_matrix = np.zeros((3*half_window*sample_rate, N))
    for i in tqdm(np.arange(N)):
        idx1 = int(signal_begin[i]*sample_rate) - int(half_window*sample_rate)
        idx2 = int(signal_begin[i]*sample_rate) + \
            int(2*half_window*sample_rate)
        signal_matrix[:, i] = data_matrix[idx1: idx2, i]
    signal_pick = np.argmax(np.abs(signal_matrix), axis=0) +\
        (signal_begin*sample_rate).astype(int) - half_window*sample_rate
    data_matrix_new = np.zeros(((before_pick+after_pick)*sample_rate, N))
    for i in tqdm(np.arange(N)):
        new_row = data_matrix[signal_pick[i]-before_pick*sample_rate:
                              signal_pick[i]+after_pick*sample_rate, i]
        if data_matrix[signal_pick[i], i] < 0:
            data_matrix_new[:, i] = -1 * new_row
        else:
            data_matrix_new[:, i] = new_row
    signal_matrix_new = data_matrix_new[(before_pick-half_window)*sample_rate:
                                        (before_pick+half_window)*sample_rate,
                                        :]
    return data_matrix_new, signal_matrix_new


def filter_and_resample(stream, lowpass_freq, resample_rate):
    """
    Apply lowpass filter and resample to a seismic stream.
    
    Args:
        stream (obspy.Stream): Input seismic waveform stream
        lowpass_freq (float): Lowpass filter frequency in Hz
        resample_rate (float): Target sampling rate in Hz
        
    Returns:
        obspy.Stream: Filtered and resampled stream
        
    Example:
        >>> stream = filter_and_resample(raw_stream, 1.0, 50.0)
    """
    stream_copy = stream.copy()
    
    if lowpass_freq > 0:
        stream_copy.filter('lowpass', freq=lowpass_freq, zerophase=True)
    
    if resample_rate > 0:
        stream_copy.resample(resample_rate)
    
    return stream_copy



def merge2_data(directory, olabel, slabel, sample_rate):
    obs_waveforms = np.load('%s/np_waveforms_%s.npy' % (directory, olabel))
    obs_stations = pd.read_csv('%s/df_stations_%s.csv' % (directory, olabel))

    simu_waveforms = np.load('%s/np_waveforms_%s.npy' % (directory, slabel))
    simu_stations = pd.read_csv('%s/df_stations_%s.csv' % (directory, slabel))

    #  N is the number of traces
    M1, N1 = obs_waveforms.shape
    M2, N2 = simu_waveforms.shape
    if M1 == M2:
        M = M1
    name_list = simu_stations['station'].to_list() + \
        obs_stations['station'].to_list()
    name_list = set(name_list)
    station_list = []
    for station in name_list:
        station_list.append(station)
    N = len(station_list)
    data_matrix = np.zeros((M, N, 2))
    df_merge = pd.DataFrame(data=np.zeros((N, 7)),
                            columns=['station',
                                     'stla', 'stlo', 'gcarc',
                                     's2ks', 's3ks', 's4ks'])
    for i in np.arange(N):
        station = station_list[i]

        row_obs = obs_stations[obs_stations.station == station]
        row_simu = simu_stations[simu_stations.station == station]
        #print(row_obs)

        try:
            # obs data at 0, simu data at 1-3
            data_matrix[:, i, 0] = obs_waveforms[:, row_obs.tridx].T
            data_matrix[:, i, 1] = simu_waveforms[:, row_simu.tridx].T
        except Exception as e:
            print(e)

        df_merge.loc[i, 'station'] = station
        df_merge.loc[i, 'tridx'] = int(i)
        df_merge.loc[i, 'stla'] = row_obs.stla.values[0]
        df_merge.loc[i, 'stlo'] = row_obs.stlo.values[0]
        df_merge.loc[i, 'gcarc'] = row_obs.gcarc.values[0]
        df_merge.loc[i, 's2ks'] = row_obs.s2ks.values[0]
        df_merge.loc[i, 's3ks'] = row_obs.s3ks.values[0]
        df_merge.loc[i, 's4ks'] = row_obs.s4ks.values[0]

    df_merge.to_csv('%s/df_stations_merge2.csv' % directory, index=False)
    data_diff = data_matrix[:, :, 1] - data_matrix[:, :, 0]
    plot_two_data_matrix(data_matrix1=data_matrix[:,:,0],
                         data_matrix2=data_matrix[:,:,1],
                         df_station=df_merge, sample_rate=10)
    plot_diff_data_matrix(data_matrix=data_diff,
                          df_station=df_merge, sample_rate=10)
    plot_corrcoef(data_matrix=data_matrix, df_station=df_merge, panel=1)
    np.save('%s/np_waveforms_merge2.npy' % directory, data_matrix)


def merge3_data(directory, olabel, slabel1, slabel2, sample_rate):
    obs_waveforms = np.load('%s/np_waveforms_%s.npy' % (directory, olabel))
    obs_stations = pd.read_csv('%s/df_stations_%s.csv' % (directory, olabel))
    simu_waveforms1 = np.load('%s/np_waveforms_%s.npy' %
                              (directory, slabel1))
    simu_stations1 = pd.read_csv('%s/df_stations_%s.csv' %
                                 (directory, slabel1))
    simu_waveforms2 = np.load('%s/np_waveforms_%s.npy' %
                              (directory, slabel2))
    simu_stations2 = pd.read_csv('%s/df_stations_%s.csv' %
                                 (directory, slabel2))

    #  N is the number of traces
    M0, N0 = obs_waveforms.shape
    M1, N1 = simu_waveforms1.shape
    M2, N2 = simu_waveforms1.shape
    if M0 == M1 and M1 == M2:
        M = M1
    name_list = simu_stations1['station'].to_list() + \
        simu_stations1['station'].to_list() + \
        obs_stations['station'].to_list()
    name_list = set(name_list)
    station_list = []
    for station in name_list:
        station_list.append(station)
    N = len(station_list)
    data_matrix = np.zeros((M, N, 3))
    df_merge = pd.DataFrame(data=np.zeros((N, 7)),
                            columns=['station',
                                     'stla', 'stlo', 'gcarc',
                                     's2ks', 's3ks', 's4ks'])
    for i in np.arange(N):
        station = station_list[i]
        print("merge %s" % station)
        row_obs = obs_stations[obs_stations.station == station]
        row_simu1 = simu_stations1[simu_stations1.station == station]
        row_simu2 = simu_stations2[simu_stations2.station == station]
        # obs data at 0, simu data at 1-3
        
        try:
            data_matrix[:, i, 0] = obs_waveforms[:, row_obs.tridx].T
            data_matrix[:, i, 1] = simu_waveforms1[:, row_simu1.tridx].T
            data_matrix[:, i, 2] = simu_waveforms2[:, row_simu2.tridx].T
        except Exception as e:
            print(e)

        df_merge.loc[i, 'station'] = station
        df_merge.loc[i, 'tridx'] = i
        df_merge.loc[i, 'stla'] = row_obs.stla.values[0]
        df_merge.loc[i, 'stlo'] = row_obs.stlo.values[0]
        df_merge.loc[i, 'gcarc'] = row_obs.gcarc.values[0]
        df_merge.loc[i, 's2ks'] = row_obs.s2ks.values[0]
        df_merge.loc[i, 's3ks'] = row_obs.s3ks.values[0]
        df_merge.loc[i, 's4ks'] = row_obs.s4ks.values[0]

    
    plot_two_data_matrix(data_matrix1=data_matrix[:, :, 0],
                         data_matrix2=data_matrix[:, :, 1],
                         df_station=df_merge, sample_rate=10)
    plot_two_data_matrix(data_matrix1=data_matrix[:, :, 0],
                         data_matrix2=data_matrix[:, :, 2],
                         df_station=df_merge, sample_rate=10)
    data_diff1 = data_matrix[:, :, 1] - data_matrix[:, :, 0]
    data_diff2 = data_matrix[:, :, 2] - data_matrix[:, :, 0]
    plot_diff_data_matrix(data_matrix=data_diff1,
                          df_station=df_merge, sample_rate=10)
    plot_diff_data_matrix(data_matrix=data_diff2,
                          df_station=df_merge, sample_rate=10)
    ccf1 = plot_corrcoef(data_matrix=data_matrix, df_station=df_merge, panel=0)
    ccf2 = plot_corrcoef(data_matrix=data_matrix, df_station=df_merge, panel=1)
    df_merge['ccf1'] = ccf1
    df_merge['ccf2'] = ccf2
    df_merge.to_csv('%s/df_stations_merge3.csv' % directory,
                    index=False, float_format='%.3f')
    np.save('%s/np_waveforms_merge3.npy' % directory, data_matrix)


def merge4_data(directory, olabel, slabel1, slabel2, slabel3,
                sample_rate, label):
    obs_waveforms = np.load('%s/np_waveforms_%s.npy' % (directory, olabel))
    obs_stations = pd.read_csv('%s/df_stations_%s.csv' % (directory, olabel))
    simu_waveforms1 = np.load('%s/np_waveforms_%s.npy' %
                              (directory, slabel1))
    simu_stations1 = pd.read_csv('%s/df_stations_%s.csv' %
                                 (directory, slabel1))
    simu_waveforms2 = np.load('%s/np_waveforms_%s.npy' %
                              (directory, slabel2))
    simu_stations2 = pd.read_csv('%s/df_stations_%s.csv' %
                                 (directory, slabel2))

    simu_waveforms3 = np.load('%s/np_waveforms_%s.npy' %
                              (directory, slabel3))
    simu_stations3 = pd.read_csv('%s/df_stations_%s.csv' %
                                 (directory, slabel3))

    #  N is the number of traces
    M0, N0 = obs_waveforms.shape
    M1, N1 = simu_waveforms1.shape
    M2, N2 = simu_waveforms1.shape
    M3, N3 = simu_waveforms1.shape
    if M0 == M1 and M1 == M2 and M2 == M3:
        M = M1
    name_list = simu_stations1['station'].to_list() + \
        simu_stations1['station'].to_list() + \
        obs_stations['station'].to_list()
    name_list = set(name_list)
    station_list = []
    for station in name_list:
        station_list.append(station)
    N = len(station_list)
    data_matrix = np.zeros((M, N, 4))
    df_merge = pd.DataFrame(data=np.zeros((N, 7)),
                            columns=['station',
                                     'stla', 'stlo', 'gcarc',
                                     's2ks', 's3ks', 's4ks'])
    for i in np.arange(N):
        station = station_list[i]
        print("merge %s" % station)
        row_obs = obs_stations[obs_stations.station == station]
        row_simu1 = simu_stations1[simu_stations1.station == station]
        row_simu2 = simu_stations2[simu_stations2.station == station]
        row_simu3 = simu_stations3[simu_stations3.station == station]
        
        try:
            data_matrix[:, i, 0] = obs_waveforms[:, row_obs.tridx].T
            data_matrix[:, i, 1] = simu_waveforms1[:, row_simu1.tridx].T
            data_matrix[:, i, 2] = simu_waveforms2[:, row_simu2.tridx].T
            data_matrix[:, i, 3] = simu_waveforms3[:, row_simu3.tridx].T
        except Exception as e:
            print(e)

        df_merge.loc[i, 'station'] = station
        df_merge.loc[i, 'tridx'] = i
        df_merge.loc[i, 'stla'] = row_obs.stla.values[0]
        df_merge.loc[i, 'stlo'] = row_obs.stlo.values[0]
        df_merge.loc[i, 'gcarc'] = row_obs.gcarc.values[0]
        df_merge.loc[i, 's2ks'] = row_obs.s2ks.values[0]
        df_merge.loc[i, 's3ks'] = row_obs.s3ks.values[0]
        df_merge.loc[i, 's4ks'] = row_obs.s4ks.values[0]

    
    plot_two_data_matrix(data_matrix1=data_matrix[:, :, 0],
                         data_matrix2=data_matrix[:, :, 1],
                         df_station=df_merge, sample_rate=10)
    plot_two_data_matrix(data_matrix1=data_matrix[:, :, 0],
                         data_matrix2=data_matrix[:, :, 2],
                         df_station=df_merge, sample_rate=10)
    plot_two_data_matrix(data_matrix1=data_matrix[:, :, 0],
                         data_matrix2=data_matrix[:, :, 3],
                         df_station=df_merge, sample_rate=10)
    data_diff1 = data_matrix[:, :, 1] - data_matrix[:, :, 0]
    data_diff2 = data_matrix[:, :, 2] - data_matrix[:, :, 0]
    data_diff3 = data_matrix[:, :, 3] - data_matrix[:, :, 0]
    plot_diff_data_matrix(data_matrix=data_diff1,
                          df_station=df_merge, sample_rate=10)
    plot_diff_data_matrix(data_matrix=data_diff2,
                          df_station=df_merge, sample_rate=10)
    plot_diff_data_matrix(data_matrix=data_diff3,
                          df_station=df_merge, sample_rate=10)
    ccf1 = plot_corrcoef(data_matrix=data_matrix, df_station=df_merge, panel=1)
    ccf2 = plot_corrcoef(data_matrix=data_matrix, df_station=df_merge, panel=2)
    ccf3 = plot_corrcoef(data_matrix=data_matrix, df_station=df_merge, panel=3)
    df_merge['ccf1'] = ccf1
    df_merge['ccf2'] = ccf2
    df_merge['ccf3'] = ccf3
    df_merge.to_csv('%s/df_stations_merge4%s.csv' % (directory, label),
                    index=False, float_format='%.3f')
    np.save('%s/np_waveforms_merge4%s.npy' % (directory, label), data_matrix)


def merge6_data(directory, olabel, slabel1, slabel2, slabel3, slabel4,
                slabel5, sample_rate, label):
    obs_waveforms = np.load('%s/np_waveforms_%s.npy' % (directory, olabel))
    obs_stations = pd.read_csv('%s/df_stations_%s.csv' % (directory, olabel))

    slabels = [slabel1, slabel2, slabel3, slabel4, slabel5]
    simu_waveforms = []
    simu_stations = []
    for slabel in slabels:
        simu_waveforms.append(np.load('%s/np_waveforms_%s.npy' %
                                      (directory, slabel)))
        simu_stations.append(pd.read_csv('%s/df_stations_%s.csv' %
                                         (directory, slabel)))

    #  N is the number of traces
    npts = [obs_waveforms.shape[0]]
    npts.extend([waveforms.shape[0] for waveforms in simu_waveforms])
    if len(set(npts)) != 1:
        raise ValueError('waveform sample counts do not match')
    M = npts[0]

    name_list = obs_stations['station'].to_list()
    for stations in simu_stations:
        name_list += stations['station'].to_list()
    name_list = set(name_list)
    station_list = []
    for station in name_list:
        station_list.append(station)
    N = len(station_list)
    data_matrix = np.zeros((M, N, 6))
    df_merge = pd.DataFrame({'station': [''] * N,
                             'stla': np.zeros(N),
                             'stlo': np.zeros(N),
                             'gcarc': np.zeros(N),
                             's2ks': np.zeros(N),
                             's3ks': np.zeros(N),
                             's4ks': np.zeros(N)})
    for i in np.arange(N):
        station = station_list[i]
        print("merge %s" % station)
        row_obs = obs_stations[obs_stations.station == station]
        rows_simu = [stations[stations.station == station]
                     for stations in simu_stations]

        try:
            data_matrix[:, i, 0] = obs_waveforms[:, row_obs.tridx].T
            for panel in np.arange(5):
                data_matrix[:, i, panel+1] = \
                    simu_waveforms[panel][:, rows_simu[panel].tridx].T
        except Exception as e:
            print(e)

        df_merge.loc[i, 'station'] = station
        df_merge.loc[i, 'tridx'] = i
        df_merge.loc[i, 'stla'] = row_obs.stla.values[0]
        df_merge.loc[i, 'stlo'] = row_obs.stlo.values[0]
        df_merge.loc[i, 'gcarc'] = row_obs.gcarc.values[0]
        df_merge.loc[i, 's2ks'] = row_obs.s2ks.values[0]
        df_merge.loc[i, 's3ks'] = row_obs.s3ks.values[0]
        df_merge.loc[i, 's4ks'] = row_obs.s4ks.values[0]

    for panel in np.arange(1, 6):
        ccf = plot_corrcoef(data_matrix=data_matrix, df_station=df_merge,
                            panel=panel, plot=False)
        df_merge['ccf%d' % panel] = ccf

    df_merge.to_csv('%s/df_stations_merge6_%s.csv' % (directory, label),
                    index=False, float_format='%.3f')
    np.save('%s/np_waveforms_merge6_%s.npy' % (directory, label), data_matrix)



# def merge2_measure(df_hypo, df_measure_ind, df_measure_syn,):
#     hypo = df_hypo.iloc[0, :]
    
#     ntrs = len(df_measure_syn)
#     df_measure = pd.DataFrame(data=np.zeros((ntrs, 20)),
#                               columns=['evtid', 'evla', 'evlo', 'evdp',
#                                        'station', 'stla', 'stlo', 'gcarc',
#                                        's2ks', 's3ks', 's4ks',
#                                        'snr',
#                                        't32_cc', 't32_pp',
#                                        'ccf_cc', 'ccf_pp',
#                                        't32_fw', 'ccf_fw',
#                                        'group', 'selected'])
#     # mearge the measurement of two method
#     for idx, row in df_measure_syn.iterrows():
#         row_ind = df_measure_ind[df_measure_ind.station == row['station']]
#         df_measure.loc[idx, 'evtid'] = hypo.evtid
#         df_measure.loc[idx, 'evla'] = hypo.evla
#         df_measure.loc[idx, 'evlo'] = hypo.evlo
#         df_measure.loc[idx, 'evdp'] = hypo.evdp
#         df_measure.loc[idx, 'station'] = row.station
#         df_measure.loc[idx, 'stla'] = row.stla
#         df_measure.loc[idx, 'stlo'] = row.stlo
#         df_measure.loc[idx, 'gcarc'] = row.gcarc
#         df_measure.loc[idx, 'ts2ks'] = row.s2ks
#         df_measure.loc[idx, 'ts3ks'] = row.s3ks
#         df_measure.loc[idx, 'ts4ks'] = row.s4ks
#         df_measure.loc[idx, 't32_fw'] = row.t32_fw
#         df_measure.loc[idx, 'ccf_fw'] = row.ccf_fw
#         single = row_ind.iloc[0, :]
#         if len(row_ind) > 0:
#             df_measure.loc[idx, 'snr'] = single.snr
#             df_measure.loc[idx, 't32_cc'] = single.t32_cc
#             df_measure.loc[idx, 't32_pp'] = single.t32_pp
#             df_measure.loc[idx, 'ccf_cc'] = single.ccf_cc
#             df_measure.loc[idx, 'ccf_pp'] = single.ccf_pp
#         else:
#             df_measure.loc[idx, 'snr'] = 0
#             df_measure.loc[idx, 't32_cc'] = 0
#             df_measure.loc[idx, 't32_pp'] = 0
#             df_measure.loc[idx, 'ccf_cc'] = 0
#             df_measure.loc[idx, 'ccf_pp'] = 0
#     return df_measure


def merge3_measure(event_directory, fn_stations, fn_measure_ind,
                   fn_measure_syn1, fn_measure_syn2, fn_output):
    fn_hypo = '%s/hypo.csv' % event_directory
    hypo = read_hypo(fn_hypo)
    df_stations = pd.read_csv('%s/%s' % (event_directory, fn_stations))

    df_measure_ind = pd.read_csv('%s/%s'
                                % (event_directory, fn_measure_ind))
    df_measure_syn1 = pd.read_csv('%s/%s'
                                % (event_directory, fn_measure_syn1))
    df_measure_syn2 = pd.read_csv('%s/%s'
                                % (event_directory, fn_measure_syn2))


    ntrs = len(df_measure_syn1)
    df_measure = pd.DataFrame(data=np.zeros((ntrs, 23)),
                              columns=['evtid', 'evla', 'evlo', 'evdp',
                                       'station', 'stla', 'stlo', 'gcarc',
                                       's2ks', 's3ks', 's4ks',
                                       'tridx', 'snr', 'ccf1', 'ccf2',
                                       't32_cc', 't32_pp',
                                       'ccf_cc', 'ccf_pp',
                                       't32_fw1', 'ccf_fw1',
                                       't32_fw2', 'ccf_fw2'])
    # mearge the measurement of two method
    for idx, row in df_stations.iterrows():
        row_ind = df_measure_ind[df_measure_ind.station == row['station']]
        row_syn1 = df_measure_syn1[
            df_measure_syn1.station == row['station']]
        row_syn2 = df_measure_syn2[
            df_measure_syn2.station == row['station']]
        
        single = row_ind.iloc[0, :]
        double1 = row_syn1.iloc[0, :]
        double2 = row_syn2.iloc[0, :]
        
        df_measure.loc[idx, 'tridx'] = row.tridx
        df_measure.loc[idx, 'ccf1'] = row.ccf1
        df_measure.loc[idx, 'ccf2'] = row.ccf2
        df_measure.loc[idx, 'evtid'] = hypo['evtid']
        df_measure.loc[idx, 'evla'] = hypo['evla']
        df_measure.loc[idx, 'evlo'] = hypo['evlo']
        df_measure.loc[idx, 'evdp'] = hypo['evdp']
        df_measure.loc[idx, 'station'] = row.station
        df_measure.loc[idx, 'stla'] = row.stla
        df_measure.loc[idx, 'stlo'] = row.stlo
        df_measure.loc[idx, 'gcarc'] = row.gcarc
        df_measure.loc[idx, 's2ks'] = row.s2ks
        df_measure.loc[idx, 's3ks'] = row.s3ks
        df_measure.loc[idx, 's4ks'] = row.s4ks
        df_measure.loc[idx, 't32_fw1'] = double1.t32_fw
        df_measure.loc[idx, 'ccf_fw1'] = double1.ccf_fw
        df_measure.loc[idx, 't32_fw2'] = double2.t32_fw
        df_measure.loc[idx, 'ccf_fw2'] = double2.ccf_fw
    
        if len(row_ind) > 0:
            df_measure.loc[idx, 'snr'] = single.snr
            df_measure.loc[idx, 't32_cc'] = single.t32_cc
            df_measure.loc[idx, 't32_pp'] = single.t32_pp
            df_measure.loc[idx, 'ccf_cc'] = single.ccf_cc
            df_measure.loc[idx, 'ccf_pp'] = single.ccf_pp
        else:
            df_measure.loc[idx, 'snr'] = 0
            df_measure.loc[idx, 't32_cc'] = 0
            df_measure.loc[idx, 't32_pp'] = 0
            df_measure.loc[idx, 'ccf_cc'] = 0
            df_measure.loc[idx, 'ccf_pp'] = 0
    df_measure.to_csv('%s/%s' % (event_directory, fn_output),
                index=False, float_format='%.3f')
    print("merge 3 measurements successfully!\n%s\n%s\n%s\n" % (
        fn_measure_ind, fn_measure_syn1, fn_measure_syn2))
    return df_measure

def merge4_measure(event_directory, fn_stations, fn_measure_ind,
                   fn_measure_syn1, fn_measure_syn2, fn_measure_syn3, fn_output):
    
    hypo = read_hypo("%s/hypo.csv" % event_directory)
    df_stations = pd.read_csv('%s/%s' % (event_directory, fn_stations))

    df_measure_ind = pd.read_csv('%s/%s'
                                % (event_directory, fn_measure_ind))
    df_measure_syn1 = pd.read_csv('%s/%s'
                                % (event_directory, fn_measure_syn1))
    df_measure_syn2 = pd.read_csv('%s/%s'
                                % (event_directory, fn_measure_syn2))
    df_measure_syn3 = pd.read_csv('%s/%s'
                                % (event_directory, fn_measure_syn3))
    ntrs = len(df_measure_syn1)
    df_measure = pd.DataFrame(data=np.zeros((ntrs, 25)),
                              columns=['evtid', 'evla', 'evlo', 'evdp',
                                       'station', 'stla', 'stlo', 'gcarc',
                                       's2ks', 's3ks', 's4ks',
                                       'tridx', 'snr',
                                        'ccf1', 'ccf2',
                                       't32_cc', 't32_pp',
                                       'ccf_cc', 'ccf_pp',
                                       't32_fw1', 'ccf_fw1',
                                       't32_fw2', 'ccf_fw2',
                                       't32_fw3', 'ccf_fw3'])
    # mearge the measurement of two method
    for idx, row in df_stations.iterrows():
        print(row)
        row_ind = df_measure_ind[df_measure_ind.station == row['station']]
        row_syn1 = df_measure_syn1[
            df_measure_syn1.station == row['station']]
        row_syn2 = df_measure_syn2[
            df_measure_syn2.station == row['station']]
        row_syn3 = df_measure_syn3[
            df_measure_syn3.station == row['station']]
        
        single = row_ind.iloc[0, :]
        double1 = row_syn1.iloc[0, :]
        double2 = row_syn2.iloc[0, :]
        double3 = row_syn3.iloc[0, :]
        
        df_measure.loc[idx, 'tridx'] = row.tridx
        df_measure.loc[idx, 'ccf1'] = row.ccf1
        df_measure.loc[idx, 'ccf2'] = row.ccf2
        df_measure.loc[idx, 'evtid'] = hypo['evtid']
        df_measure.loc[idx, 'evla'] = hypo['evla']
        df_measure.loc[idx, 'evlo'] = hypo['evlo']
        df_measure.loc[idx, 'evdp'] = hypo['evdp']
        df_measure.loc[idx, 'station'] = row.station
        df_measure.loc[idx, 'stla'] = row.stla
        df_measure.loc[idx, 'stlo'] = row.stlo
        df_measure.loc[idx, 'gcarc'] = row.gcarc
        df_measure.loc[idx, 's2ks'] = row.s2ks
        df_measure.loc[idx, 's3ks'] = row.s3ks
        df_measure.loc[idx, 's4ks'] = row.s4ks
        df_measure.loc[idx, 't32_fw1'] = double1.t32_fw
        df_measure.loc[idx, 'ccf_fw1'] = double1.ccf_fw
        df_measure.loc[idx, 't32_fw2'] = double2.t32_fw
        df_measure.loc[idx, 'ccf_fw2'] = double2.ccf_fw
        df_measure.loc[idx, 't32_fw3'] = double3.t32_fw
        df_measure.loc[idx, 'ccf_fw3'] = double3.ccf_fw
    
        if len(row_ind) > 0:
            df_measure.loc[idx, 'snr'] = single.snr
            df_measure.loc[idx, 't32_cc'] = single.t32_cc
            df_measure.loc[idx, 't32_pp'] = single.t32_pp
            df_measure.loc[idx, 'ccf_cc'] = single.ccf_cc
            df_measure.loc[idx, 'ccf_pp'] = single.ccf_pp
        else:
            df_measure.loc[idx, 'snr'] = 0
            df_measure.loc[idx, 't32_cc'] = 0
            df_measure.loc[idx, 't32_pp'] = 0
            df_measure.loc[idx, 'ccf_cc'] = 0
            df_measure.loc[idx, 'ccf_pp'] = 0
    
    df_measure.to_csv('%s/%s' % (event_directory, fn_output),
                  index=False, float_format='%.3f')
    
    return df_measure


def merge6_np2d_to_np3d(evtid):
    data0 = np.load("data/%s/np_waveforms_0.npy" % evtid) 
    data1 = np.load("data/%s/np_waveforms_1.npy" % evtid) 
    data2 = np.load("data/%s/np_waveforms_2.npy" % evtid) 
    data3 = np.load("data/%s/np_waveforms_3.npy" % evtid) 
    data4 = np.load("data/%s/np_waveforms_4.npy" % evtid) 
    data5 = np.load("data/%s/np_waveforms_5.npy" % evtid)
    data_merge6 = np.stack([data0, data1, data2, data3, data4, data5], axis=-1)
    np.save("./data/%s/np_waveforms_merge6.npy" % evtid, data_merge6)


def merge6_np3d_to_np2d(evtid):
    data_merge6 = np.load("./data/%s/np_waveforms_merge6.npy" % evtid)
    print("read")
    for i in np.arange(6):
        np.save("./data/%s/np_waveforms_%d" % (evtid, i), data_merge6[:, :, i]) 


def merge6_measure(event_directory, fn_stations, fn_measure_ind,
                   fn_measure_syn1, fn_measure_syn2, fn_measure_syn3,
                   fn_measure_syn4, fn_measure_syn5, fn_output):
    fn_hypo = '%s/hypo.csv' % event_directory
    hypo = read_hypo(fn_hypo)
    df_stations = pd.read_csv('%s/%s' % (event_directory, fn_stations))

    def load_measure(measure):
        if isinstance(measure, pd.DataFrame):
            return measure
        return pd.read_csv('%s/%s' % (event_directory, measure))

    df_measure_ind = load_measure(fn_measure_ind)
    df_measure_syn1 = load_measure(fn_measure_syn1)
    df_measure_syn2 = load_measure(fn_measure_syn2)
    df_measure_syn3 = load_measure(fn_measure_syn3)
    df_measure_syn4 = load_measure(fn_measure_syn4)
    df_measure_syn5 = load_measure(fn_measure_syn5)

    ntrs = len(df_measure_syn1)
    df_measure = pd.DataFrame(data=np.zeros((ntrs, 37)),
                              columns=['evtid', 'evla', 'evlo', 'evdp',
                                       'station', 'stla', 'stlo', 'gcarc',
                                       's2ks', 's3ks', 's4ks',
                                       'tridx', 'snr',
                                        'ccf1', 'ccf2', 'ccf3', 'ccf4', 'ccf5',
                                       't32_cc', 't32_pp',
                                       'ccf_cc', 'ccf_pp',
                                       't32_fw1', 'ccf_fw1', 'ccf_sg1',
                                       't32_fw2', 'ccf_fw2', 'ccf_sg2',
                                       't32_fw3', 'ccf_fw3', 'ccf_sg3',
                                       't32_fw4', 'ccf_fw4', 'ccf_sg4',
                                       't32_fw5', 'ccf_fw5', 'ccf_sg5'])
    # mearge the measurement of two method
    df_measure["evtid"]  = df_measure["evtid"].astype("object")
    df_measure["station"]  = df_measure["station"].astype("object")
    for idx, row in df_stations.iterrows():
        row_ind = df_measure_ind[df_measure_ind.station == row['station']]
        row_syn1 = df_measure_syn1[
            df_measure_syn1.station == row['station']]
        row_syn2 = df_measure_syn2[
            df_measure_syn2.station == row['station']]
        row_syn3 = df_measure_syn3[
            df_measure_syn3.station == row['station']]
        row_syn4 = df_measure_syn4[
            df_measure_syn4.station == row['station']]
        row_syn5 = df_measure_syn5[
            df_measure_syn5.station == row['station']]

        double1 = row_syn1.iloc[0, :]
        double2 = row_syn2.iloc[0, :]
        double3 = row_syn3.iloc[0, :]
        double4 = row_syn4.iloc[0, :]
        double5 = row_syn5.iloc[0, :]

        df_measure.loc[idx, 'tridx'] = row.tridx
        df_measure.loc[idx, 'ccf1'] = row.ccf1
        df_measure.loc[idx, 'ccf2'] = row.ccf2
        df_measure.loc[idx, 'ccf3'] = row.ccf3
        df_measure.loc[idx, 'ccf4'] = row.ccf4
        df_measure.loc[idx, 'ccf5'] = row.ccf5
        df_measure.loc[idx, 'evtid'] = hypo['evtid']
        df_measure.loc[idx, 'evla'] = hypo['evla']
        df_measure.loc[idx, 'evlo'] = hypo['evlo']
        df_measure.loc[idx, 'evdp'] = hypo['evdp']
        df_measure.loc[idx, 'station'] = row.station
        df_measure.loc[idx, 'stla'] = row.stla
        df_measure.loc[idx, 'stlo'] = row.stlo
        df_measure.loc[idx, 'gcarc'] = row.gcarc
        df_measure.loc[idx, 's2ks'] = row.s2ks
        df_measure.loc[idx, 's3ks'] = row.s3ks
        df_measure.loc[idx, 's4ks'] = row.s4ks
        df_measure.loc[idx, 't32_fw1'] = double1.t32_fw
        df_measure.loc[idx, 'ccf_fw1'] = double1.ccf_fw
        df_measure.loc[idx, 'ccf_sg1'] = double1.ccf_fw
        df_measure.loc[idx, 't32_fw2'] = double2.t32_fw
        df_measure.loc[idx, 'ccf_fw2'] = double2.ccf_fw
        df_measure.loc[idx, 'ccf_sg2'] = double1.ccf_fw
        df_measure.loc[idx, 't32_fw3'] = double3.t32_fw
        df_measure.loc[idx, 'ccf_fw3'] = double3.ccf_fw
        df_measure.loc[idx, 'ccf_sg3'] = double1.ccf_fw
        df_measure.loc[idx, 't32_fw4'] = double4.t32_fw
        df_measure.loc[idx, 'ccf_fw4'] = double4.ccf_fw
        df_measure.loc[idx, 'ccf_sg4'] = double1.ccf_fw
        df_measure.loc[idx, 't32_fw5'] = double5.t32_fw
        df_measure.loc[idx, 'ccf_fw5'] = double5.ccf_fw
        df_measure.loc[idx, 'ccf_sg5'] = double1.ccf_fw

        if len(row_ind) > 0:
            single = row_ind.iloc[0, :]
            df_measure.loc[idx, 'snr'] = single.snr
            df_measure.loc[idx, 't32_cc'] = single.t32_cc
            df_measure.loc[idx, 't32_pp'] = single.t32_pp
            df_measure.loc[idx, 'ccf_cc'] = single.ccf_cc
            df_measure.loc[idx, 'ccf_pp'] = single.ccf_pp
        else:
            df_measure.loc[idx, 'snr'] = 0
            df_measure.loc[idx, 't32_cc'] = 0
            df_measure.loc[idx, 't32_pp'] = 0
            df_measure.loc[idx, 'ccf_cc'] = 0
            df_measure.loc[idx, 'ccf_pp'] = 0

    df_measure.to_csv('%s/%s' % (event_directory, fn_output),
                index=False, float_format='%.3f')
    measure_sources = [
        '<dataframe>' if isinstance(measure, pd.DataFrame) else measure
        for measure in (fn_measure_ind, fn_measure_syn1, fn_measure_syn2,
                        fn_measure_syn3, fn_measure_syn4, fn_measure_syn5)
    ]
    print("merge 6 measurements successfully!\n%s\n%s\n%s\n%s\n%s\n%s\n" %
          tuple(measure_sources))
    return df_measure


def add_group_info(fref, ftarget, fout):
    dfref = pd.read_csv(fref)
    dftarget = pd.read_csv(ftarget)
    for idx, row, in dftarget.iterrows():
         station = row.station
         df2_select_row = dfref[dfref.station==station]
         group = df2_select_row.group.values
         select_flag = df2_select_row.selected.values
         az = df2_select_row.az.values
         print(group, select_flag, az)
         dftarget.loc[idx, 'group'] = group
         dftarget.loc[idx, 'selected'] = select_flag
         dftarget.loc[idx, 'az'] = az
    dftarget.to_csv(fout)



if __name__ == "__main__":
    print("Data processing utilities module")
    print("Functions available:")
    print("  - cal_dist_time()")
    print("  - traces_to_matrix()")
    print("  - matrix_align_slant()")
    print("  - filter_and_resample()")
