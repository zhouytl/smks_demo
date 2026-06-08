#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for calculating raypath.

This module contains functions for computing great circle raypaths
between source and receiver locations.
"""

import pandas as pd
import numpy as np
import pyproj
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees
from tqdm import tqdm


# Initialize geodetic object
projection = pyproj.Geod(ellps='WGS84')
model_prem = TauPyModel(model="prem")


def cal_raypath(latS, lonS, latR, lonR):
    raypath_ref = projection.npts(lonS, latS, lonR, latR, 300)
    raypath = [list(item) for item in raypath_ref]
    raypath = raypath + [[lonR, latR]]
    raypath = np.array(raypath)
    return raypath


def cal_az(latS, lonS, latR, lonR):
    projection = pyproj.Geod(ellps='WGS84')
    az, azbaz, dist = projection.inv(lonS, latS, lonR, latR)
    return az


def cal_dist_time(df_station, df_hypo):
    model = TauPyModel(model="prem")
    evla = df_hypo.evla[0]
    evlo = df_hypo.evlo[0]
    evdp = df_hypo.evdp[0]
    nsta = len(df_station)
    infos = np.zeros((nsta, 4))
    for idx, row in tqdm(df_station.iterrows(), total=df_station.shape[0]):
        gcarc = locations2degrees(evla, evlo, row.stla, row.stlo)
        arrivals_SKKS = model.get_travel_times(source_depth_in_km=evdp,
                                               distance_in_degree=gcarc,
                                               phase_list=["SKKS"])
        arrivals_S3KS = model.get_travel_times(source_depth_in_km=evdp,
                                               distance_in_degree=gcarc,
                                               phase_list=["SKKKS"])
        arrivals_S4KS = model.get_travel_times(source_depth_in_km=evdp,
                                               distance_in_degree=gcarc,
                                               phase_list=["SKKKKS"])
        s2ks = arrivals_SKKS[0].time
        s3ks = arrivals_S3KS[0].time
        s4ks = arrivals_S4KS[0].time
        infos[idx, :] = np.array([gcarc, s2ks, s3ks, s4ks])
    df_station[['gcarc', 's2ks', 's3ks', 's4ks']] = infos
    return df_station


def cal_pierce(latS, lonS, depS, latR, lonR, phase, depR=None):
    """
    Calculate piercing point locations of seismic ray at CMB (Core-Mantle Boundary).
    
    Parameters
    ----------
    latS : float
        Source latitude in degrees
    lonS : float
        Source longitude in degrees
    depS : float
        Source depth in km
    latR : float
        Receiver latitude in degrees
    lonR : float
        Receiver longitude in degrees
    phase : str
        Seismic phase name (e.g., 'SKKS', 'S3KS')
    depR : float, optional
        Receiver depth in km (not used but kept for backward compatibility)
    
    Returns
    -------
    tuple of float
        (latPin, lonPin, latPout, lonPout)
        - latPin, lonPin: Latitude and longitude of ray entry point at CMB
        - latPout, lonPout: Latitude and longitude of ray exit point at CMB
        Returns (np.nan, np.nan, np.nan, np.nan) if phase is not found
    """
    az_SR, baz_SR, dist_SR_km = projection.inv(lonS, latS, lonR, latR)
    gcarc = locations2degrees(latS, lonS, latR, lonR)
    pierces_prem = model_prem.get_pierce_points(source_depth_in_km=depS,
                                                distance_in_degree=gcarc,
                                                phase_list=[phase])
    if len(pierces_prem) > 0:
        path = pierces_prem[0].pierce
        pierces_points = [list(item) for item in path]
        pcmb = []
        for ppoint in pierces_points:
            if ppoint[3] == 2891:
                pcmb.append(ppoint)

        pcmb_in = pcmb[0]
        pcmb_out = pcmb[-1]
        angle_pcmb_in = 180*pcmb_in[2]/np.pi
        angle_pcmb_out = 180*pcmb_out[2]/np.pi
        dist_in_km = dist_SR_km*angle_pcmb_in/gcarc
        dist_out_km = dist_SR_km*angle_pcmb_out/gcarc

        lonPin, latPin, bazPin = projection.fwd(lonS, latS,
                                                az_SR, dist_in_km)
        lonPout, latPout, bazPout = projection.fwd(lonS, latS,
                                                   az_SR, dist_out_km)
    else:
        latPin = np.nan
        lonPin = np.nan
        latPout = np.nan
        lonPout = np.nan
    return latPin, lonPin, latPout, lonPout


def cal_pierce_and_bounce(latS, lonS, depS, latR, lonR, phase, depR=None):
    """
    Calculate all piercing points of seismic ray at CMB (Core-Mantle Boundary).
    
    Returns lists of all entry and exit points where the ray crosses the CMB,
    useful for multiple-bounce phases like SKKS (multiple core reflections).
    
    Parameters
    ----------
    latS : float
        Source latitude in degrees
    lonS : float
        Source longitude in degrees
    depS : float
        Source depth in km
    latR : float
        Receiver latitude in degrees
    lonR : float
        Receiver longitude in degrees
    phase : str
        Seismic phase name (e.g., 'SKKS', 'SKKKS')
    depR : float, optional
        Receiver depth in km (not used but kept for backward compatibility)
    
    Returns
    -------
    tuple of (list, list)
        (latPBlist, lonPBlist) - Lists of piercing point coordinates
        - latPBlist: List of latitudes of all CMB piercing points
        - lonPBlist: List of longitudes of all CMB piercing points
        Returns ([], []) if phase is not found
    """
    az_SR, baz_SR, dist_SR_km = projection.inv(lonS, latS, lonR, latR)
    gcarc = locations2degrees(latS, lonS, latR, lonR)
    
    pierces_prem = model_prem.get_pierce_points(source_depth_in_km=depS,
                                                distance_in_degree=gcarc,
                                                phase_list=[phase])
    
    latPBlist = []
    lonPBlist = []
    
    if len(pierces_prem) > 0:
        path = pierces_prem[0].pierce
        pierces_points = [list(item) for item in path]
        pcmb = []
        
        # Find all points at CMB depth (2891 km)
        for ppoint in pierces_points:
            if ppoint[3] == 2891:
                pcmb.append(ppoint)
        
        # Calculate coordinates for each piercing point
        for bpidx in np.arange(len(pcmb)):
            pb = pcmb[bpidx]
            angle = 180 * pb[2] / np.pi
            dist_in_km = dist_SR_km * angle / gcarc
            lonB, latB, bazB = projection.fwd(lonS, latS, az_SR, dist_in_km)
            latPBlist.append(latB)
            lonPBlist.append(lonB)
    
    return latPBlist, lonPBlist


def cal_coverage(df_selected_traces, saven=None):
    raypath_array_lon = np.zeros((301, len(df_selected_traces)))
    raypath_array_lat = np.zeros((301, len(df_selected_traces)))
    total_traces = len(df_selected_traces)
    for idx, row in tqdm(df_selected_traces.iterrows(), total=total_traces):
        latS = row.evla
        lonS = row.evlo
        depS = row.evdp
        latR = row.stla
        lonR = row.stlo
        try: 
            latPin_s3ks, lonPin_s3ks, latPout_s3ks, lonPout_s3ks = cal_pierce(
                latS, lonS, depS, latR, lonR, phase='SKKKS')
            raypath = cal_raypath(latPin_s3ks, lonPin_s3ks,
                                latPout_s3ks, lonPout_s3ks)
            raypath_array_lon[:, idx] = raypath[:, 0]
            raypath_array_lat[:, idx] = raypath[:, 1]
        except Exception as e:
            print(e)

    df_raypath = pd.DataFrame({'longitude': raypath_array_lon.flatten(),
                            'latitude': raypath_array_lat.flatten()})
    df_raypath.to_csv('output/raypath_selected.csv', index=False)
    df_raypath = pd.read_csv('output/raypath_selected.csv')
    lon_list = np.arange(-180, 181, 2)
    lat_list = np.arange(-90, 91, 2)
    nlon = len(lon_list)
    nlat = len(lat_list)
    X, Y = np.meshgrid(lon_list, lat_list)
    Z = np.zeros((len(lat_list), len(lon_list)))
    xx = X.flatten()
    yy = Y.flatten()
    npts = len(xx)

    radius = 2
    for i in tqdm(np.arange(npts)):
        xmin = xx[i] - radius
        xmax = xx[i] + radius
        ymin = yy[i] - radius
        ymax = yy[i] + radius
        df_nearby = df_raypath[(df_raypath.longitude > xmin) &
                            (df_raypath.longitude < xmax) &
                            (df_raypath.latitude > ymin) &
                            (df_raypath.latitude < ymax)]
        if len(df_nearby) > 0:
            ix = int(i // 181)
            iy = int(i % 181)
            Z[ix, iy] = 1
    total_percentage = len(Z[Z>0])/npts*100
    print('coverage_selected = %.2f %%' % total_percentage)
    output_array = np.zeros((nlat, nlon, 3))
    output_array[:, :, 0] = X
    output_array[:, :, 1] = Y
    output_array[:, :, 2] = Z
    if saven is not None:
        np.save(saven, output_array)
    return output_array