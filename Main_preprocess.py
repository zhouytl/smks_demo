#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Yangtianli Zhou 06/2026
"""

# %% SmKS Intro
# from smks_utils.visualization_utils import plot_smks_features, plot_SmKS_travel_time_table
# plot_smks_features(gcarc=120, depth=560)
# plot_SmKS_travel_time_table(depth=560)

# %%
# from smks_utils.data_utils import merge6_np3d_to_np2d
# from glob import glob
# evtdir_list = glob('./data/20*')

# for evtdir in evtdir_list:
#     evtid = evtdir.split('/')[-1]
#     merge6_np3d_to_np2d(evtid)

# exit()

# %% measure smks solo residuals
from smks_utils.measure_smks_utils import measure_merge6_altogether
from glob import glob
# evtdir_list = ['./data/20100724_0535', './data/20110101_0957', './data/20200706_2254','./data/20230301_0536']
evtdir_list = glob('./data/20*')
#  glob('./data/20110101_0957')


for evtdir in evtdir_list:
        evtid = evtdir.split('/')[-1]
        measure_merge6_altogether(evtid=evtid, fname='df_stations_merge6.csv', sample_rate=10, plot=True)
        # out df_measured.csv


exit()
# %% Post Processing
from smks_utils.selection_utils import auto_select
from smks_utils.selection_utils import cluster_traces
from smks_utils.selection_utils import sum_groups_merge6
from smks_utils.visualization_utils import plot_residual_and_rss
import matplotlib.pyplot as plt
import pandas as pd
import glob

# evtdir_list = glob('./data/20100724_0535')
evtdir_list = ['./data/20100724_0535', './data/20110101_0957', './data/20200706_2254','./data/20230301_0536']


selection_args = [['test1', 2.0, 0.3, 0.5],
                  ['test2', 3.0, 0.4, 0.6], # my choice
                  ['test3', 4.0, 0.5, 0.7],
                  ['test4', 5.0, 0.6, 0.8]]

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
    dfs = [pd.read_csv(f) for f in glob.glob("data/20*/df_groups.csv")]
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

# %%
