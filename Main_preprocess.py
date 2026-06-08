#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Yangtianli Zhou 06/2026
"""

# %% SmKS Intro
# this cell can plot the basic feature of SmKS, like arrival, sample depth, etc.
from smks_utils.visualization_utils import plot_smks_features, plot_SmKS_travel_time_table
plot_smks_features(gcarc=120, depth=560)
plot_SmKS_travel_time_table(depth=560)


# %% measure smks solo residuals
"""
This cell can measure the SmKS travel time residual and save .pdf file for every single traces. 
The measured figures are not saved in the github repo due to folder size limitation. 
Set plot=False if you prefer a fast run without plot. 
"""
from smks_utils.measure_smks_utils import measure_merge6_altogether
from glob import glob
evtdir_list = glob('./data/20*')
for evtdir in evtdir_list:
        evtid = evtdir.split('/')[-1]
        measure_merge6_altogether(evtid=evtid, fname='df_stations_merge6.csv', sample_rate=10, plot=True)
        # out df_measured.csv

# %% Post Processing
"""
This cell can autoselect and cluster traces into groups, 
and then summarize every group and plot their distance and residual. 
residualc: individual cross-correlation measurement
residualp: individual pick arrival measurement
residual1: waveform simulation (PREM)
residual2: waveform simulation (PREM + SP12RTS)
residual3: waveform simulation (PREM + SP12RTS + Torus)
residual4: waveform simulation (PREM + SP12RTS + Global)
residual5: waveform simulation (PREM + SP12RTS + subLLSVP)
"""
from smks_utils.selection_utils import auto_select
from smks_utils.selection_utils import cluster_traces
from smks_utils.selection_utils import sum_groups_merge6
from smks_utils.visualization_utils import plot_residual_and_rss
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob

evtdir_list = glob('./data/20*')
selection_args = [['test1', 3.0, 0.4, 0.6], # my choice
                  ['test2', 4.0, 0.6, 0.8]]

"""
selection are:
    1. testid
    2. minimum SmKS SNR
    3. cross-correlation coefficient between observed and synthetic 200s SmKS signal
    4. cross-correlation coefficient between observed SKKS and S3KS,
        same as cross-correlation coefficient between observed and synthetic S3KS.
"""

for args in selection_args:
    testid = args[0] 
    min_snr = args[1]
    min_ccf = args[2]
    min_sig_ccf = args[3]
    for evtdir in evtdir_list:
        evtid = evtdir.split('/')[-1]
        auto_select(evtid, fname='df_measured.csv',
                     min_snr=min_snr, min_ccf=min_ccf, min_sig_ccf=min_sig_ccf)
        # output df_selected.csv
        cluster_traces(evtid, fname='df_selected.csv', plot=False) 
        # output df_clustered.csv
        sum_groups_merge6(evtid, fname='df_clustered.csv')
        # output df_groups.csv
    dfs = [pd.read_csv(f) for f in glob("data/20*/df_groups.csv")]
    df_all = pd.concat(dfs, ignore_index=True)
    df_all.to_csv('data/df_all_groups.csv')
    print(testid + "result:")
    fig, axes = plt.subplots(3, 2)
    plot_residual_and_rss(dfdir='data/df_all_groups.csv', column='residualc', ax=axes[0, 0])
    plot_residual_and_rss(dfdir='data/df_all_groups.csv', column='residual1', ax=axes[0, 1])
    plot_residual_and_rss(dfdir='data/df_all_groups.csv', column='residual2', ax=axes[1, 0])
    plot_residual_and_rss(dfdir='data/df_all_groups.csv', column='residual3', ax=axes[1, 1])
    plot_residual_and_rss(dfdir='data/df_all_groups.csv', column='residual4', ax=axes[2, 0])
    plot_residual_and_rss(dfdir='data/df_all_groups.csv', column='residual5', ax=axes[2, 1])
    plt.tight_layout()
    plt.show()

