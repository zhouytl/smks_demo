#!/usr/bin/env python3

import os
import numpy as np
import shutil
import pandas as pd
from glob import glob
from .raypath_utils import cal_az
import matplotlib.pyplot as plt
import pyproj
from sklearn.cluster import DBSCAN
from .visualization_utils import plot_waveform_and_raypath

def auto_select(evtid, fname, min_snr, min_ccf, min_sig_ccf):

    evtdir = 'data/%s' % evtid
    fn_measure = '%s/%s' % (evtdir, fname)
    df_measure = pd.read_csv(fn_measure)
    df_measure['auto_selected'] = (
        (df_measure.snr > min_snr)
        & (df_measure.ccf2 > min_ccf)
        & (df_measure.ccf_fw2 > min_sig_ccf)
        & (df_measure.ccf_cc > min_sig_ccf)
    ).astype(int)
    df_measure.to_csv("%s/df_selected.csv" % evtdir, index=False)

    opt_list = glob("%s/Figures/*" % evtdir)
    for option in opt_list:
        objgood = '%s/selected' % option
        objbad = '%s/unselected' % option

        if not os.path.exists(objgood):
            os.makedirs(objgood)
        if not os.path.exists(objbad):
            os.makedirs(objbad)

        for idx, row in df_measure.iterrows():
            figname = '%s.png' % row.station
            figure_in_objdir = '%s/%s' % (option, figname)
            try:
                if row.auto_selected == 1:
                    shutil.move(figure_in_objdir, objgood)
                else:
                    shutil.move(figure_in_objdir, objbad)
            except:
                print ("already moved %s" % row.station)



def sync_dirs(refdir, objdir):
    refgood = '%s/good' % refdir
    refbad = '%s/bad' % refdir
    objgood = '%s/good' % objdir
    objbad = '%s/bad' % objdir

    if not os.path.exists(objgood):
        os.makedirs(objgood)
    if not os.path.exists(objbad):
        os.makedirs(objbad)

    objgood_list = glob('%s/*' % objgood)
    objbad_list = glob('%s/*' % objbad)
    refgood_list = glob('%s/*' % refgood)
    refbad_list = glob('%s/*' % refbad)
    
    print(objgood_list)
    if len(objgood_list) > 0:
        for fname in objgood_list:
            print(fname)
            shutil.move(fname, objdir)
    if len(objbad_list) > 0:
        for fname in objbad_list:
            shutil.move(fname, objdir)

    for fdir in refgood_list:
        figname = fdir.split('/')[-1]
        figure_in_objdir = '%s/%s' % (objdir, figname)
        shutil.move(figure_in_objdir, objgood)

    for fdir in refbad_list:
        figname = fdir.split('/')[-1]
        figure_in_objdir = '%s/%s' % (objdir, figname)
        print(figure_in_objdir)
        shutil.move(figure_in_objdir, objbad)


def catagorize_figures(fn_input, ref_directory, fn_output):
    df_measure = pd.read_csv(fn_input)
    good_dir = '%s/good' % ref_directory
    bad_dir = '%s/bad' % ref_directory
    select_list = []
    for idx, row in df_measure.iterrows():
        figure_name = '%s.png' % row['station']
        print('%s/%s' % (good_dir, figure_name))
        if os.path.exists('%s/%s' % (good_dir, figure_name)):
            select_list.append(1)
        elif os.path.exists('%s/%s' % (bad_dir, figure_name)):
            select_list.append(0)
        else:
            select_list.append(-1)
    df_measure['selected'] = select_list
    df_measure.to_csv(fn_output, index=False)
    return df_measure


def cluster_traces(evtid, fname, plot=True):
    df = pd.read_csv("data/%s/%s" % (evtid, fname))
    df['group'] = np.ones(len(df)) * (-1)
    np_az = np.zeros(len(df))
    for idx, row in df.iterrows():
        az = cal_az(row.evla, row.evlo, row.stla, row.stlo)
        np_az[idx] = az
    df['az'] = np.round(np_az, 3)
    
    mask = df.auto_selected > 0
    data = df.loc[mask, ['az', 'gcarc']].values
    labels = DBSCAN(eps=8, min_samples=2).fit_predict(data)
    df.loc[mask, 'group'] = labels

    df.to_csv("data/%s/df_clustered.csv" % evtid, index=False)

    df = df.loc[mask]
    if plot:
        fig = plt.figure(figsize=(8, 8), dpi=600)
        fig.suptitle(evtid)
        gs = fig.add_gridspec(nrows=8, ncols=6)

        ax1 = fig.add_subplot(gs[0:2, 0:2])
        ax1.hist(df.t32_fw2, bins=10,
                color='gold', edgecolor='lime', alpha=0.8,
                density=True, zorder=1)
        ax1.plot([0, 0], [0, 100], c='k', linewidth=0.5, zorder=0)
        ax1.set_ylim(0, 1)
        ax1.set_xlim(-3, 3)
        ax1.grid()

        ax2 = fig.add_subplot(gs[0:2, 2:6])
        ax2.scatter(df.gcarc, df.t32_fw2, marker='o', s=8,
                    color='gold', edgecolor='lime',
                    linewidth=0.2, alpha=0.8, zorder=1)
        ax2.plot([120, 180], [0, 0], c='k', linewidth=0.5, zorder=0)
        ax2.set_xlim(120, 180)
        ax2.set_ylim(-3, 3)
        ax2.grid()

        ax3 = fig.add_subplot(gs[2:6, 0:3], projection='polar')
        ax3.scatter(df.az*np.pi/180, df.gcarc, c=df.t32_fw2,
                    marker='o', s=5, cmap='rainbow', vmin=-0.5, vmax=0.5)
        ax3.set_yticks(range(120, 181, 20), range(120, 181, 20))
        ax3.set_rmin(100)
        ax3.set_rmax(180)

        ax4 = fig.add_subplot(gs[2:6, 3:6], projection='polar')
        ax4.scatter(df.az*np.pi/180, df.gcarc,
                    marker='o', s=5,
                    c=labels, cmap='tab20b')
        ax4.set_yticks(range(120, 181, 20), range(120, 181, 20))
        ax4.set_rmin(100)
        ax4.set_rmax(180)

        ax6 = fig.add_subplot(gs[6:8, 2:6])
        t32_list = []
        for lab in set(labels):
            df_select = df[df.group == lab]
            if lab > -1:
                gcarc = np.mean(df_select.gcarc.values)
                t32 = np.mean(df_select.t32_fw2.values)
                t32_list.append(t32)
                std = np.std(df_select.t32_fw2.values)
                dist_range = np.std(df_select.gcarc.values)
                ax6.errorbar(x=gcarc, y=t32, xerr=dist_range, yerr=std,
                            c='k', zorder=0)
                ax6.scatter(x=gcarc, y=t32, marker='s', s=10,
                            color='w', edgecolor='k', zorder=1)

        ax6.plot([120, 180], [0, 0], c='k', linewidth=1, zorder=0)
        ax6.set_xlim(120, 180)
        ax6.set_ylim(-3, 3)
        ax6.grid()

        ax5 = fig.add_subplot(gs[6:8, 0:2])
        ax5.hist(t32_list, bins=3, color='gold', edgecolor='lime', alpha=0.8,
                density=True, zorder=1)
        ax5.set_ylim(0, 1)
        ax5.set_xlim(-3, 3)
        ax5.grid()

        plt.tight_layout()

        os.makedirs( "data/%s/Figures/group" % evtid, exist_ok=True)
        figdir = "data/%s/Figures/group/grouping.png" % evtid
        fig.savefig(figdir)


def sum_groups_merge3(dfname, fdata, fhypo, figdir, fgrp):

    if not os.path.exists(figdir):
        os.makedirs(figdir)
    data_matrix = np.load(fdata)
    df = pd.read_csv(dfname)
    df_hypo = pd.read_csv(fhypo)
    group_list = list(set(df.group.values))
    ngrp = len(group_list)
    df_group = pd.DataFrame(data=np.zeros((0, 19)),
                            columns=['group', 'ntr', 
                                     'evla', 'evlo', 'evdp',
                                     'stla', 'stlo', 'gcarc', 'gcarc_std',
                                     'residualp', 'residual_stdp',
                                     'residualc', 'residual_stdc',
                                     'residual1', 'residual_std1',
                                     'residual3', 'residual_std3',
                                     'fiji', 'new'])

    for grpid in group_list:
        if grpid >=0 :
            df_select = df[df.group == grpid]
            ntr = len(df_select)
            evla = np.mean(df_select.evla)
            evlo = np.mean(df_select.evlo)
            evdp = np.mean(df_select.evdp)
            stla = np.mean(df_select.stla)
            stlo = np.mean(df_select.stlo)
            gcarc = np.mean(df_select.gcarc)
            std_gcarc = np.std(df_select.gcarc)
            t32_p = np.mean(df_select.t32_pp)
            std_t32_p = np.std(df_select.t32_pp)
            t32_c = np.mean(df_select.t32_cc)
            std_t32_c = np.std(df_select.t32_cc)
            t32_1 = np.mean(df_select.t32_fw1)
            std_t32_1 = np.std(df_select.t32_fw1)
            t32_3 = np.mean(df_select.t32_fw2)
            std_t32_3 = np.std(df_select.t32_fw2)
            # s4ks_select = 0.0
            # t42_3 = np.mean(df_select.t42_fw2)
            # std_t42_3 = np.std(df_select.t32_fw2)
            if std_gcarc > 0.1:
                df_row = pd.DataFrame(data=[[grpid, ntr,
                                             evla, evlo, evdp, stla, stlo,
                                             gcarc, std_gcarc,
                                             t32_p, std_t32_p,
                                             t32_c, std_t32_c,
                                             t32_1, std_t32_1,
                                             t32_3, std_t32_3,
                                             #s4ks_select,
                                             #t42_3, std_t42_3,
                                             0, 0]],
                                      columns=['group', 'ntr',
                                               'evla', 'evlo', 'evdp',
                                               'stla', 'stlo',
                                               'gcarc', 'gcarc_std',
                                               'residualp', 'residual_stdp',
                                               'residualc', 'residual_stdc',
                                               'residual1', 'residual_std1',
                                               'residual3', 'residual_std3',
                                               #'residual_s4ks', 'residual_std_s4ks',
                                               'fiji', 'new'])
                df_group = pd.concat([df_group, df_row])
                ftitle = 'group: %03d; trace number: %04d; '\
                         'gcarc=%.1f±%.1f°; residual=%.1f±%.1fs' %\
                         (grpid, ntr, gcarc, std_gcarc, t32_3, std_t32_3)
                fname = '%s/%03d.pdf' % (figdir, grpid)
                plot_waveform_and_raypath(df_select, df_hypo, data_matrix,
                                          sample_rate=10,
                                          title=ftitle, figdir=fname, layer=1)
            else:
                df.loc[df['group'] == grpid, 'group'] = -1
    df_single = df_select = df[df.group == -1]
    ftitle_single = 'group: -1; trace number: %04d; '\
                    'residual=%.1f±%.1fs' %\
                    (len(df_single), np.mean(df_single.t32_fw2),
                     np.std(df_single.t32_fw2))
    fname_single = '%s/singles.pdf' % figdir
    plot_waveform_and_raypath(df_select, df_hypo, data_matrix,
                              sample_rate=10,
                              title=ftitle_single, figdir=fname_single, layer=1)
    df_group.to_csv(fgrp, index=False)
    df.to_csv(dfname, index=False)


def sum_groups_merge4(dfname, fdata, fhypo, figdir, fgrp):
    if not os.path.exists(figdir):
        os.makedirs(figdir)
    data_matrix = np.load(fdata)
    df = pd.read_csv(dfname)
    df_hypo = pd.read_csv(fhypo)
    group_list = list(set(df.group.values))
    ngrp = len(group_list)
    new_residual = 'residuals'
    new_std = 'residual_stds'
    df_group = pd.DataFrame(data=np.zeros((0, 21)),
                            columns=['group', 'ntr', 
                                     'evla', 'evlo', 'evdp',
                                     'stla', 'stlo', 'gcarc', 'gcarc_std',
                                     'residualpp', 'residual_stdpp',
                                     'residualcc', 'residual_stdcc',
                                     'residual1', 'residual_std1',
                                     'residual3', 'residual_std3',
                                     new_residual, new_std,
                                     'fiji', 'new'])

    for grpid in group_list:
        if grpid >=0 :
            df_select = df[df.group == grpid]
            ntr = len(df_select)
            evla = np.mean(df_select.evla)
            evlo = np.mean(df_select.evlo)
            evdp = np.mean(df_select.evdp)
            stla = np.mean(df_select.stla)
            stlo = np.mean(df_select.stlo)
            gcarc = np.mean(df_select.gcarc)
            std_gcarc = np.std(df_select.gcarc)
            t32_p = np.mean(df_select.t32_pp)
            std_t32_p = np.std(df_select.t32_pp)
            t32_c = np.mean(df_select.t32_cc)
            std_t32_c = np.std(df_select.t32_cc)
            t32_1 = np.mean(df_select.t32_fw1)
            std_t32_1 = np.std(df_select.t32_fw1)
            t32_3 = np.mean(df_select.t32_fw2)
            std_t32_3 = np.std(df_select.t32_fw2)
            # s4ks_select = 0.0
            t32_s = np.mean(df_select.t32_fw3)
            std_t32_s = np.std(df_select.t32_fw3)
            # if std_gcarc > 0.1:
            df_row = pd.DataFrame(data=[[grpid, ntr,
                                            evla, evlo, evdp, stla, stlo,
                                            gcarc, std_gcarc,
                                            t32_p, std_t32_p,
                                            t32_c, std_t32_c,
                                            t32_1, std_t32_1,
                                            t32_3, std_t32_3,
                                            # s4ks_select,
                                            t32_s, std_t32_s,
                                            0, 0]],
                                    columns=['group', 'ntr',
                                            'evla', 'evlo', 'evdp',
                                            'stla', 'stlo',
                                            'gcarc', 'gcarc_std',
                                            'residualpp', 'residual_stdpp',
                                            'residualcc', 'residual_stdcc',
                                            'residual1', 'residual_std1',
                                            'residual3', 'residual_std3',
                                            new_residual, new_std,
                                            'fiji', 'new'])
            df_group = pd.concat([df_group, df_row])
            ftitle = 'group: %03d; trace number: %04d; '\
                        'gcarc=%.1f±%.1f°; residual=%.1f±%.1fs' %\
                        (grpid, ntr, gcarc, std_gcarc, t32_s, std_t32_s)
            fname = '%s/%03d.pdf' % (figdir, grpid)
            plot_waveform_and_raypath(df_select, df_hypo, data_matrix,
                                        sample_rate=10,
                                        title=ftitle, figdir=fname, layer=3)
        #else:
        #    df.loc[df['group'] == grpid, 'group'] = -1
    df_single = df_select = df[df.group == -1]
    ftitle_single = 'group: -1; trace number: %04d; '\
                    'residual=%.1f±%.1fs' %\
                    (len(df_single), np.mean(df_single.t32_fw3),
                     np.std(df_single.t32_fw3))
    fname_single = '%s/singles.pdf' % figdir
    plot_waveform_and_raypath(df_select, df_hypo, data_matrix,
                              sample_rate=10,
                              title=ftitle_single, figdir=fname_single, layer=3)
    df_group.to_csv(fgrp, index=False)
    df.to_csv(dfname, index=False)
            #else:
            #    df.loc[df['group'] == grpid, 'group'] = -1
    # df_single = df_select = df[df.group == -1]
    df_group.to_csv(fgrp, index=False)
    # df.to_csv(dfname, index=False)


def sum_groups_merge6(evtid, fname):
    df = pd.read_csv('data/%s/%s' % (evtid, fname))
    group_list = list(set(df.group.values))
    df_group = pd.DataFrame(data=np.zeros((0, 23)),
                            columns=['group', 'ntr', 
                                     'evla', 'evlo', 'evdp',
                                     'stla', 'stlo', 'gcarc', 'gcarc_std',
                                     'residualp', 'residualp_std',
                                     'residualc', 'residualc_std',
                                     'residual1', 'residual1_std',
                                     'residual2', 'residual2_std',
                                     'residual3', 'residual3_std',
                                     'residual4', 'residual4_std',
                                     'residual5', 'residual5_std'])

    for grpid in group_list:
        if grpid >=0 :
            df_select = df[df.group == grpid]
            std_gcarc = float(np.std(df_select.gcarc))
            std_residual2 = np.std(df_select.t32_fw2)
            if std_gcarc <= 0.1 or std_residual2 > 2.0:
                print(std_gcarc, std_residual2)
                df.loc[df["group"] == grpid, "group"] = -1
                continue
            ntr = len(df_select)
            evla = np.mean(df_select.evla)
            evlo = np.mean(df_select.evlo)
            evdp = np.mean(df_select.evdp)
            stla = np.mean(df_select.stla)
            stlo = np.mean(df_select.stlo)
            gcarc = np.mean(df_select.gcarc)
            gcarc_std = np.std(df_select.gcarc)
            residualp = np.mean(df_select.t32_pp)
            residualp_std = np.std(df_select.t32_pp)
            residualc = np.mean(df_select.t32_cc)
            residualc_std = np.std(df_select.t32_cc)
            residual1 = np.mean(df_select.t32_fw1)
            residual1_std = np.std(df_select.t32_fw1)
            residual2 = np.mean(df_select.t32_fw2)
            residual2_std= np.std(df_select.t32_fw2)
            residual3 = np.mean(df_select.t32_fw3)
            residual3_std = np.std(df_select.t32_fw3)
            residual4 = np.mean(df_select.t32_fw4)
            residual4_std = np.std(df_select.t32_fw4)
            residual5 = np.mean(df_select.t32_fw5)
            residual5_std = np.std(df_select.t32_fw5)
            df_row = pd.DataFrame(data=[[grpid, ntr,
                                            evla, evlo, evdp, stla, stlo,
                                            gcarc, gcarc_std,
                                            residualp, residualp_std,
                                            residualc, residualc_std,
                                            residual1, residual1_std,
                                            residual2, residual2_std,
                                            residual3, residual3_std,
                                            residual4, residual4_std,
                                            residual5, residual5_std]],
                                    columns=['group', 'ntr',
                                            'evla', 'evlo', 'evdp',
                                            'stla', 'stlo',
                                            'gcarc', 'gcarc_std',
                                            'residualp', 'residualp_std',
                                            'residualc', 'residualc_std',
                                            'residual1', 'residual1_std',
                                            'residual2', 'residual2_std',
                                            'residual3', 'residual3_std',
                                            'residual4', 'residual4_std',
                                            'residual5', 'residual5_std',])
            df_group = pd.concat([df_group, df_row])
    df_group = df_group.round(3)
    df.to_csv("data/%s/df_clustered_updated.csv" % evtid, index=False)
    df_group.to_csv("data/%s/df_groups.csv" % evtid, index=False)
