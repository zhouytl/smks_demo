#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization utilities for seismic data and analysis.

This module contains functions for plotting and visualizing seismic data,
including waveforms, maps, and station/event distributions.

Created on April 26, 2026
@author: zhouyangtianli
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from cartopy import crs as ccrs
from obspy.imaging.beachball import beach
from obspy.geodetics import locations2degrees
from .raypath_utils import cal_raypath, cal_pierce

from obspy.taup.velocity_model import VelocityModel
from obspy.taup import plot_travel_times
from obspy.taup import plot_ray_paths
from obspy.taup import TauPyModel
from obspy.taup.taup_create import build_taup_model
model_prem = TauPyModel(model="prem")

DPI = 200


def plot_smks_features(gcarc, depth):

    gcarcs = np.arange(120, 170, 1)
    gcarcN = len(gcarcs)

    ## calculate theoretical PREM

    evdp = depth
    radius = 6371

    dist_list = range(120, 180)
    skks_time_list = []
    for dist in range(120, 180):
        travel_time = model_prem.get_travel_times(source_depth_in_km=evdp, distance_in_degree=dist, phase_list=['SKKS'])
        skks_time_list.append(travel_time[0].time)

    s3ks_time_list = []
    for dist in range(120, 180):
        travel_time = model_prem.get_travel_times(source_depth_in_km=evdp, distance_in_degree=dist, phase_list=['SKKKS'])
        s3ks_time_list.append(travel_time[0].time)

    fig = plt.figure(figsize=(4,3), dpi=DPI)
    plt.plot(dist_list, skks_time_list, 'r', 4, label='S2KS')
    plt.plot(dist_list, s3ks_time_list, 'g', 4, label='S3KS')
    plt.xlim(120, 170)
    plt.ylim(1500, 2000)
    plt.grid()
    plt.xlabel('Distance (degree)')
    plt.ylabel('Travel Time (s)')
    plt.tight_layout()


    skks_ray_path = model_prem.get_ray_paths(source_depth_in_km=evdp, distance_in_degree=gcarc, phase_list=['SKKS'])
    skks_path = skks_ray_path[0].path
    skks_path = [list(item) for item in skks_path]
    skks_path = np.array(skks_path)
    skks_theta = skks_path[:, 2]
    skks_rho = radius-skks_path[:,3]

    s3ks_ray_path = model_prem.get_ray_paths(source_depth_in_km=evdp, distance_in_degree=gcarc, phase_list=['SKKKS'])
    s3ks_path = s3ks_ray_path[0].path
    s3ks_path = [list(item) for item in s3ks_path]
    s3ks_path = np.array(s3ks_path)
    s3ks_theta = s3ks_path[:, 2]
    s3ks_rho = radius-s3ks_path[:,3]

    rad = np.linspace(0, 2*np.pi, 100)
    crust = np.ones(100) * radius
    cmb = np.ones(100) * (radius-2891)
    icb = np.ones(100) * (radius-5150.5)
    eprm = np.ones(100) * (radius-2891-300)

    fig, ax = plt.subplots(figsize=(3, 3), dpi=600,
                        subplot_kw={'projection': 'polar'})
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    # ax.set_yticks([radius-5150.5, radius-2891], ['ICB', 'CMB'])
    ax.plot(skks_theta, skks_rho, 'k', 4, label='S2KS', zorder=2)
    ax.plot(s3ks_theta, s3ks_rho, 'k', 4, label='S3KS', zorder=2)
    ax.scatter(0, radius-500, s=120, marker='*', c='r', edgecolor='k', zorder=3)
    ax.scatter(np.deg2rad(gcarc), radius+100, s=50, marker='^',
            c='green', edgecolor='k', zorder=3)
    ax.plot(rad, crust, 'k', linewidth=0.5, zorder=0)
    ax.plot(rad, cmb, 'k', linewidth=1, zorder=0)
    ax.plot(rad, icb, 'k', linewidth=0.5, zorder=0)
    ax.plot(rad, eprm, 'k', linewidth=1, linestyle=':', zorder=0)
    ax.fill_between(rad, eprm, cmb, color='grey', alpha=0.5, linewidth=0, zorder=0)
    # ax.fill_between(rad, icb, eprm, color='grey', alpha=0.3, zorder=0)
    plt.show()


def plot_SmKS_travel_time_table(depth):
    model = model_prem

    gcarcList = range(120, 181)
    npts = len(gcarcList)

    PPP1 = np.zeros(npts)
    pPPP1 = np.zeros(npts)
    sPPP1 = np.zeros(npts)
    PcPPKP1 = np.zeros(npts)
    SKKS1 = np.zeros(npts)
    S3KS1 = np.zeros(npts)
    PKKS1 = np.zeros(npts)

    PPP2 = np.zeros(npts)
    pPPP2 = np.zeros(npts)
    sPPP2 = np.zeros(npts)
    PcPPKP2 = np.zeros(npts)
    SKKS2 = np.zeros(npts)
    S3KS2 = np.zeros(npts)
    PKKS2 = np.zeros(npts)

    PKS1 = np.zeros(npts)
    PKS2 = np.zeros(npts)
    SKS1 = np.zeros(npts)
    SKS2 = np.zeros(npts)


    for i in np.arange(npts):
        
        gcarc = gcarcList[i]
        
        arrivals_SKKS = model.get_travel_times(source_depth_in_km=depth,
                                            distance_in_degree=gcarc,
                                            phase_list=['SKKS'])
        if len(arrivals_SKKS) > 0:
            SKKS1[i] = arrivals_SKKS[0].time
            SKKS2[i] = arrivals_SKKS[-1].time
            
            
            
        arrivals_S3KS = model.get_travel_times(source_depth_in_km=depth,
                                            distance_in_degree=gcarc,
                                            phase_list=['SKKKS'])
        if len(arrivals_S3KS) > 0:
            S3KS1[i] = arrivals_S3KS[0].time
            S3KS2[i] = arrivals_S3KS[-1].time
        
        

        arrivals_PPP = model.get_travel_times(source_depth_in_km=depth,
                                            distance_in_degree=gcarc,
                                            phase_list=['PKIIIIKP'])
        if len(arrivals_PPP) > 0:
            PPP1[i] = arrivals_PPP[0].time
            PPP2[i] = arrivals_PPP[-1].time
        
        arrivals_pPPP = model.get_travel_times(source_depth_in_km=depth,
                                            distance_in_degree=gcarc,
                                            phase_list=['pPPP'])
        if len(arrivals_pPPP) > 0:
            pPPP1[i] = arrivals_pPPP[0].time
            pPPP2[i] = arrivals_pPPP[-1].time
            
            
        arrivals_sPPP = model.get_travel_times(source_depth_in_km=depth,
                                            distance_in_degree=gcarc,
                                            phase_list=['sPPP'])
        if len(arrivals_sPPP) > 0:
            sPPP1[i] = arrivals_sPPP[0].time
            sPPP2[i] = arrivals_sPPP[-1].time
        
        
        arrivals_PcPPKP = model.get_travel_times(source_depth_in_km=depth,
                                            distance_in_degree=gcarc,
                                            phase_list=['PcPPKP'])
        if len(arrivals_PcPPKP) > 0:
            PcPPKP1[i] = arrivals_PcPPKP[0].time
            PcPPKP2[i] = arrivals_PcPPKP[-1].time

        arrivals_PKKS = model.get_travel_times(source_depth_in_km=depth,
                                            distance_in_degree=gcarc,
                                            phase_list=['PPPPP'])
        if len(arrivals_PKKS) > 0:
            PKKS1[i] = arrivals_PKKS[0].time
            PKKS2[i] = arrivals_PKKS[-1].time

        arrivals_PKS = model.get_travel_times(source_depth_in_km=depth,
                                            distance_in_degree=gcarc,
                                            phase_list=['PPPP'])
        if len(arrivals_PKS) > 0:
            PKS1[i] = arrivals_PKS[0].time
            PKS2[i] = arrivals_PKS[-1].time
        
        arrivals_SKS = model.get_travel_times(source_depth_in_km=depth,
                                            distance_in_degree=gcarc,
                                            phase_list=['SKS'])
        if len(arrivals_SKS) > 0:
            SKS1[i] = arrivals_SKS[0].time
            SKS2[i] = arrivals_SKS[-1].time
    
    
    fig = plt.figure(figsize=(3, 4), dpi=200)
    plt.plot(PPP1-SKKS1, gcarcList, c='tab:blue', label='PPP')
    plt.plot(pPPP1-SKKS1, gcarcList, c='tab:orange', label='pPPP')
    plt.plot(PcPPKP1-SKKS1, gcarcList, c='tab:green', label='PcPPKP')
    plt.plot(sPPP1-SKKS1, gcarcList, c='tab:red', label='sPPP')
    plt.plot((PKKS1-SKKS1), gcarcList, c='tab:purple', label='PPPPP')
    plt.plot(SKKS1-SKKS1, gcarcList, c='k', label='SKKS')
    plt.plot(S3KS1-SKKS1, gcarcList, c='k', linestyle='--', label='S3KS')
    plt.plot(PKS1-SKKS1, gcarcList, c='tab:pink', label='PPPP')
    plt.plot(SKS1-SKKS1, gcarcList, c='tab:olive', linestyle='--', label='SKS')


    plt.plot(PPP2-SKKS1, gcarcList, c='tab:blue')
    plt.plot(pPPP2-SKKS1, gcarcList, c='tab:orange')
    plt.plot(PcPPKP2-SKKS1, gcarcList, c='tab:green')
    plt.plot(sPPP2-SKKS1, gcarcList, c='tab:red')
    plt.plot(PKKS2-SKKS1, gcarcList, c='tab:purple')
    plt.plot(SKKS2-SKKS1, gcarcList, c='k')
    plt.plot(S3KS2-SKKS1, gcarcList, c='k', linestyle='--')
    plt.plot(PKS2-SKKS1, gcarcList, c='tab:pink')
    plt.plot(SKS2-SKKS1, gcarcList, c='tab:olive', linestyle='--')
    plt.xlim(-50, 150)
    plt.ylim(120, 180)
    plt.legend()
    plt.show()




def plot_data_matrix(data_matrix, df_station, sample_rate):
    M, N = data_matrix.shape
    time = np.arange(1, M+1)/sample_rate
    gcarc = df_station.gcarc.values
    fig = plt.figure(figsize=(3, 6), dpi=DPI)
    ax = fig.add_subplot(111)
    for i in np.arange(N):
        ax.plot(time, gcarc[i]+1.0*data_matrix[:, i], 'g', linewidth=0.2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance (°)')
    plt.show()


def plot_eq_map(df_station, df_hypo):
    df_hypo = df_hypo.iloc[0, :]
    evla = df_hypo.evla
    evlo = df_hypo.evlo

    fig = plt.figure(num=None, figsize=(6, 5), dpi=200, edgecolor='k')
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=0))
    ax.coastlines()
    ax.set_global()

    for idx, row in df_station.iterrows():
        raypath = cal_raypath(evla, evlo, row.stla, row.stlo)
        raypath_west = raypath[raypath[:, 0] < 0]
        raypath_east = raypath[raypath[:, 0] > 0]
        if len(raypath_west) > 0:
            ax.plot(raypath_west[:, 0], raypath_west[:, 1], c='k', alpha=0.2,
                    transform=ccrs.PlateCarree(central_longitude=0))
        if len(raypath_east) > 0:
            ax.plot(raypath_east[:, 0], raypath_east[:, 1], c='k', alpha=0.2,
                    transform=ccrs.PlateCarree(central_longitude=0))
        ax.scatter(row.stlo, row.stla, s=5, c='k', marker='.',
                   transform=ccrs.PlateCarree(central_longitude=0))

    lon_list = np.linspace(-180, 180, 20)
    lat_list = np.linspace(-90, 90, 20)
    X, Y = np.meshgrid(lon_list, lat_list)
    Z = np.zeros((20, 20))
    for i in np.arange(0, 20):
        for j in np.arange(0, 20):
            Z[i, j] = locations2degrees(evla, evlo, Y[i, j], X[i, j])

    ax.contour(X, Y, Z, transform=ccrs.PlateCarree(central_longitude=0),
               levels=[120, 180])
    ctr = ax.contourf(X, Y, Z, transform=ccrs.PlateCarree(central_longitude=0),
                      cmap='Pastel1_r', levels=np.arange(0, 185, 20))
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    cbar = fig.colorbar(mappable=ctr, ax=ax,
                        orientation='horizontal',
                        location='bottom', pad=0.05)
    cbar.set_label('Distance(degree)')
    ax.scatter(df_station.stlo, df_station.stla,
               transform=ccrs.PlateCarree(central_longitude=0),
               s=20, c='w', marker='^', edgecolor='k',
               label='station', zorder=10)

    fm = [df_hypo.m11, df_hypo.m22, df_hypo.m33,
          df_hypo.m12, df_hypo.m13, df_hypo.m23]
    beachball = beach(fm, xy=(evlo, evla), facecolor='r', width=10)
    ax.add_collection(beachball)
    plt.tight_layout()
    plt.show()


def plot_data_matrix_with_color_label(data_matrix, df_station, sample_rate):

    M, N = data_matrix.shape
    time = np.arange(1, M+1)/sample_rate
    gcarc = df_station.gcarc.values
    fig = plt.figure(figsize=(6, 8), dpi=600)
    ax = fig.add_subplot(111)
    for i in np.arange(N-1):
        if df_station.loc[i, 'selected'] > 0:
            gcarc = df_station.loc[i, 'gcarc']
            ax.plot(time, gcarc+1.0*data_matrix[:, i], 'g', linewidth=0.2)
        else:
            gcarc = df_station.loc[i, 'gcarc']
            ax.plot(time, gcarc+1.0*data_matrix[:, i], 'r', alpha=0.2,
                    linewidth=0.2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance (°)')
    plt.show()


def plot_eq_map_with_color_label(df_station, df_hypo):
    df_hypo = df_hypo.iloc[0, :]
    evla = df_hypo.evla
    evlo = df_hypo.evlo

    fig = plt.figure(num=None, figsize=(6, 5), dpi=600, edgecolor='k')
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=0))
    ax.coastlines()
    ax.set_global()

    for idx, row in df_station.iterrows():
        raypath = cal_raypath(evla, evlo, row.stla, row.stlo)
        raypath_west = raypath[raypath[:, 0] < 0]
        raypath_east = raypath[raypath[:, 0] > 0]
        if row.selected > 0:
            if len(raypath_west) > 0:
                ax.plot(raypath_west[:, 0], raypath_west[:, 1],
                        c='g', alpha=0.2,
                        transform=ccrs.PlateCarree(central_longitude=0))
            if len(raypath_east) > 0:
                ax.plot(raypath_east[:, 0], raypath_east[:, 1],
                        c='g', alpha=0.2,
                        transform=ccrs.PlateCarree(central_longitude=0))
        else:
            if len(raypath_west) > 0:
                ax.plot(raypath_west[:, 0], raypath_west[:, 1],
                        c='r', alpha=0.1,
                        transform=ccrs.PlateCarree(central_longitude=0))
            if len(raypath_east) > 0:
                ax.plot(raypath_east[:, 0], raypath_east[:, 1],
                        c='r', alpha=0.1,
                        transform=ccrs.PlateCarree(central_longitude=0))

    lon_list = np.linspace(-180, 180, 20)
    lat_list = np.linspace(-90, 90, 20)
    X, Y = np.meshgrid(lon_list, lat_list)
    Z = np.zeros((20, 20))
    for i in np.arange(0, 20):
        for j in np.arange(0, 20):
            Z[i, j] = locations2degrees(evla, evlo, Y[i, j], X[i, j])

    ax.contour(X, Y, Z, transform=ccrs.PlateCarree(central_longitude=0),
               levels=[120, 180])
    ctr = ax.contourf(X, Y, Z, transform=ccrs.PlateCarree(central_longitude=0),
                      cmap='Pastel1_r', levels=np.arange(0, 185, 20))
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    cbar = fig.colorbar(mappable=ctr, ax=ax,
                        orientation='horizontal',
                        location='bottom', pad=0.05)
    cbar.set_label('Distance(degree)')
    ax.scatter(df_station.stlo, df_station.stla,
               transform=ccrs.PlateCarree(central_longitude=0),
               s=20, c='w', marker='^', edgecolor='k',
               label='station', zorder=10)

    fm = [df_hypo.m11, df_hypo.m22, df_hypo.m33,
          df_hypo.m12, df_hypo.m13, df_hypo.m23]
    beachball = beach(fm, xy=(evlo, evla), facecolor='r', width=10)
    ax.add_collection(beachball)
    plt.tight_layout()


# def hex_to_kml_color(hex_color, alpha='ff'):
#     """
#     Convert hex color to KML's AABBGGRR format.
    
#     KML uses a specific color format (Alpha, Blue, Green, Red) which differs
#     from standard hex color notation. This function converts a standard hex
#     color to KML's required format.
    
#     Args:
#         hex_color (str): Hex color code in format '#RRGGBB' or 'RRGGBB'
#         alpha (str, optional): Alpha channel value in hex format (default: 'ff')
#             - 'ff' = fully opaque
#             - '00' = fully transparent
#             - '80' = 50% transparent
        
#     Returns:
#         str: KML color in AABBGGRR format (e.g., 'ffff0000' for opaque red)
        
#     Example:
#         >>> hex_to_kml_color('#FF0000')
#         'ffff0000'
#         >>> hex_to_kml_color('#FF0000', alpha='80')
#         '80ff0000'
#     """
#     hex_color = hex_color.lstrip('#')
#     bb = hex_color[4:6]  # Blue
#     gg = hex_color[2:4]  # Green
#     rr = hex_color[0:2]  # Red
#     return f"{alpha}{bb}{gg}{rr}"


def plot_measurement_results(df_measure, figdir=None):
    """
    Plot measurement results including residuals and quality metrics.
    
    Creates a figure showing measured time shifts, correlation coefficients,
    and signal-to-noise ratios.
    
    Args:
        df_measure (pd.DataFrame): Measurement results DataFrame with columns:
            - gcarc (float): Epicentral distance
            - residual (float): Residual value
            - cc_max (float): Maximum correlation coefficient
            - snr (float): Signal-to-noise ratio
        figdir (str, optional): Directory to save figure (default: None, no save)
        
    Returns:
        None: Displays plot
        
    Example:
        >>> plot_measurement_results(results_df, figdir='./figures/')
    """
    fig = plt.figure(figsize=(12, 8), dpi=DPI)
    
    # Subplot 1: Residuals vs Distance
    ax1 = fig.add_subplot(221)
    ax1.scatter(df_measure.gcarc, df_measure.residual, alpha=0.6)
    ax1.set_xlabel('Epicentral Distance (°)')
    ax1.set_ylabel('Residual (s)')
    ax1.set_title('Residuals vs Distance')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Correlation Coefficient
    ax2 = fig.add_subplot(222)
    ax2.scatter(df_measure.gcarc, df_measure.cc_max, alpha=0.6, c='orange')
    ax2.set_xlabel('Epicentral Distance (°)')
    ax2.set_ylabel('Correlation Coefficient')
    ax2.set_title('Correlation Quality vs Distance')
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: SNR
    ax3 = fig.add_subplot(223)
    ax3.scatter(df_measure.gcarc, df_measure.snr, alpha=0.6, c='green')
    ax3.set_xlabel('Epicentral Distance (°)')
    ax3.set_ylabel('SNR (dB)')
    ax3.set_title('Signal-to-Noise Ratio vs Distance')
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Distribution
    ax4 = fig.add_subplot(224)
    ax4.hist(df_measure.residual, bins=20, alpha=0.7, edgecolor='k')
    ax4.set_xlabel('Residual (s)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Residuals')
    
    plt.tight_layout()
    
    if figdir:
        fig.savefig(f'{figdir}/measurement_results.png', dpi=600, bbox_inches='tight')
    
    plt.show()

def plot_coverage(X, Y, Z):    
    fig = plt.figure(num=None, figsize=(8, 6), dpi=100, edgecolor='k')
    levels = np.linspace(0.0, 1.0, 4)
    ax = plt.axes(projection=ccrs.Mollweide(central_longitude=180))
    ax.coastlines()
    ax.set_global()
    levels = np.linspace(0.0, 1.0, 4)
    ctr = ax.contourf(X, Y, Z, cmap='grey', alpha=0.8, levels=levels,
                    transform=ccrs.PlateCarree(central_longitude=0))
    plt.show()    


def plot_two_data_matrix(data_matrix1, data_matrix2, df_station, sample_rate):
    M, N = data_matrix1.shape
    time = np.arange(1, M+1)/sample_rate
    gcarc = df_station.gcarc.values
    fig = plt.figure(figsize=(3, 8), dpi=600)
    ax = fig.add_subplot(111)
    for i in np.arange(N):
        ax.plot(time, gcarc[i]+1.0*data_matrix1[:, i],
                c='tab:blue', linewidth=0.1)
        ax.plot(time, gcarc[i]+1.0*data_matrix2[:, i],
                c='tab:red', linewidth=0.1)
    plt.ylim(120, 180)


def plot_diff_data_matrix(data_matrix, df_station, sample_rate):
    M, N = data_matrix.shape
    time = np.arange(1, M+1)/sample_rate
    gcarc = df_station.gcarc.values
    fig = plt.figure(figsize=(3, 8), dpi=600)
    ax = fig.add_subplot(111)
    for i in np.arange(N):
        ax.plot(time, gcarc[i]+1.0*data_matrix[:, i],
                c='tab:green', linewidth=0.1)
    plt.ylim(120, 180)
    plt.show()


def plot_corrcoef(data_matrix, df_station, panel, plot=True):
    M, N, _ = data_matrix.shape
    gcarc = df_station.gcarc.values
    coef_list1 = []
    for i in np.arange(N):
        coef_list1.append(np.corrcoef(data_matrix[:, i, 0],
                                      data_matrix[:, i, panel])[0, 1])
    if plot:
        plt.figure()
        plt.scatter(gcarc, coef_list1, c='orange', marker='.', alpha=1.0)
        plt.xlim(120, 180)
        plt.ylim(-0.2, 1.0)
    return coef_list1


def plot_waveform_and_raypath(df_station, df_hypo, data_matrix,
                              sample_rate, title, figdir, layer):
    df_hypo = df_hypo.iloc[0, :]
    evla = df_hypo.evla
    evlo = df_hypo.evlo
    M, N, _ = data_matrix.shape
    time = np.arange(1, M+1)/sample_rate

    fig = plt.figure(figsize=(10, 4), dpi=600)
    fig.suptitle(title)
    gs = fig.add_gridspec(nrows=1, ncols=4)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1:4],
                          projection=ccrs.PlateCarree(central_longitude=0))
    ax2.coastlines()
    ax2.set_global()

    for idx, row in df_station.iterrows():
        tridx = int(row.tridx)
        gcarc = row.gcarc
        tr_obs = data_matrix[:, tridx, 0]
        tr_real = data_matrix[:, tridx, layer]
        ax1.plot(time, gcarc+1.0*tr_obs/np.max(np.abs(tr_obs)),
                 c='k', linewidth=0.5)
        ax1.plot(time, gcarc+1.0*tr_real/np.max(np.abs(tr_real)),
                 c='r', linewidth=0.5)
        raypath = cal_raypath(evla, evlo, row.stla, row.stlo)
        raypath_west = raypath[raypath[:, 0] < 0]
        raypath_east = raypath[raypath[:, 0] > 0]

        if row.selected > 0:
            if len(raypath_west) > 0:
                ax2.plot(raypath_west[:, 0], raypath_west[:, 1], c='k',
                         transform=ccrs.PlateCarree(central_longitude=0))
            if len(raypath_east) > 0:
                ax2.plot(raypath_east[:, 0], raypath_east[:, 1], c='k',
                         transform=ccrs.PlateCarree(central_longitude=0))
        else:
            if len(raypath_west) > 0:
                ax2.plot(raypath_west[:, 0], raypath_west[:, 1], c='k',
                         transform=ccrs.PlateCarree(central_longitude=0))
            if len(raypath_east) > 0:
                ax2.plot(raypath_east[:, 0], raypath_east[:, 1], c='k',
                         transform=ccrs.PlateCarree(central_longitude=0))
        ax2.scatter(df_station.stlo, df_station.stla,
                    transform=ccrs.PlateCarree(central_longitude=0),
                    s=20, c='w', marker='^', edgecolor='k',
                    label='station', zorder=10)
        fm = [df_hypo.m11, df_hypo.m22, df_hypo.m33,
              df_hypo.m12, df_hypo.m13, df_hypo.m23]
        beachball = beach(fm, xy=(evlo, evla), facecolor='r', width=10)
        ax2.add_collection(beachball)
        plt.tight_layout()

    ax1.set_ylim(119, 180)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Distance (°)')
    plt.tight_layout()
    fig.savefig(figdir)


def plot_residual_and_rss(dfdir, column, ax, error_bar=True):
    ax.plot([115, 180], [0, 0], c='k')
    df = pd.read_csv(dfdir)
    gcarc = df.gcarc
    residual = df[column]
    ax.scatter(gcarc, residual, s=20, marker='o',
               edgecolor='darkblue', facecolor='#FF5910', zorder=1) # "#FFBC91"
    if error_bar:
        residual_std = df["%s_std" % column]
        ax.errorbar(x=gcarc, y=residual,
                    yerr=residual_std, c='darkblue',
                    ls='none', zorder=0)
    mean = np.mean(np.array(residual))
    # uncertainty = np.median(np.array(residual_std))
    rss = np.sum(np.power(np.array(residual), 2))
    print("%s: RSS=%.2f" %(column, rss))
    ax.text(155, 2.5, "%.2f" % rss)
    ax.grid()
    ax.set_ylim(-3.0, 3.0)
    ax.set_xlim(118, 180)
    ax.set_title(column)
    ax.set_xlabel('Distance (°)')
    ax.set_ylabel('Residual (s)')



if __name__ == "__main__":
    print("Visualization utilities module")
    print("Functions available:")
    print("  - plot_data_matrix()")
    print("  - plot_eq_map()")
    print("  - hex_to_kml_color()")
    print("  - plot_measurement_results()")
