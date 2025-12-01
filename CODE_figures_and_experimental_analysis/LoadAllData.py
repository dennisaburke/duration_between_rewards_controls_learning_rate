# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 11:24:14 2025

@author: DeBurke
"""
import os
import sys
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import statsmodels
import pingouin
import scipy.stats as stats
import scipy.io as sio
import mat73
from pathlib import Path
sys.path.append(r'D:\DATA\OneDrive - UCSF\AnalysisScripts\\multi_project_modules')
import read_nwb_files
import default_configs as dc
import figure_functions_refactored as ff
import LickPhotoFunctions_choppedUp as lpf
import plotting_functions as pf
import stats_functions_DB as sf
import data_wrangling as dw
import Utils as util

import functions.load_preprocess  as lp
import functions.default_configs as dc
import functions.figure_functions as ff
import functions.lick_photo_functions as lpf
import functions.stats_functions as sf
import functions.data_wrangling as dw

#set working directory to script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

"""
set paths to data and outputs
"""
nwb_dir_path = r'..\DATA_experimental\001632'
figure_path_root = r'..\FIGURES'
pickle_path = r'..\DATA_experimental\pickled_dfs'

fig_path_Figure1_ExtDataFig12 = os.path.join(figure_path_root, r'Figure1_ExtDataFig12')
fig_path_Figure2_ExtDataFig4 = os.path.join(figure_path_root, r'Figure2_ExtDataFig4')
fig_path_Figure3_ExtDataFig6ae = os.path.join(figure_path_root, r'Figure3_ExtDataFig6ae')
fig_path_Figure4_ExtDataFig6fp = os.path.join(figure_path_root, r'Figure4_ExtDataFig6fp')
fig_path_Figure5_ExtDataFig7 = os.path.join(figure_path_root, r'Figure5_ExtDataFig7')
fig_path_Figure6_ExtData8af9 =  os.path.join(figure_path_root, r'Figure6_ExtData8af9')
fig_path_Figure7_ExtDataFig10 = os.path.join(figure_path_root, r'Figure7_ExtDataFig10')
fig_path_Figure8 = os.path.join(figure_path_root, r'Figure8')
fig_path_ExtDataFig8fk = os.path.join(figure_path_root, r'ExtDataFig8fk')
fig_path_ExtDataFig5 =  os.path.join(figure_path_root, r'ExtDataFig5')

Path(fig_path_Figure1_ExtDataFig12).mkdir(parents=True, exist_ok=True)
Path(fig_path_Figure2_ExtDataFig4).mkdir(parents=True, exist_ok=True)
Path(fig_path_Figure3_ExtDataFig6ae).mkdir(parents=True, exist_ok=True)
Path(fig_path_Figure4_ExtDataFig6fp).mkdir(parents=True, exist_ok=True)
Path(fig_path_Figure5_ExtDataFig7).mkdir(parents=True, exist_ok=True)
Path(fig_path_Figure6_ExtData8af9).mkdir(parents=True, exist_ok=True)
Path(fig_path_Figure7_ExtDataFig10).mkdir(parents=True, exist_ok=True)
Path(fig_path_Figure8).mkdir(parents=True, exist_ok=True)
Path(fig_path_ExtDataFig8fk).mkdir(parents=True, exist_ok=True)
Path(fig_path_ExtDataFig5).mkdir(parents=True, exist_ok=True)

#uncomment below if prefer to save all figs in one directory
# fig_path_all = ''
# (fig_path_Figure1_ExtDataFig12 = fig_path_Figure2_ExtDataFig4 = fig_path_Figure3_ExtDataFig6ae
#  = fig_path_Figure4_ExtDataFig6fp = fig_path_Figure5_ExtDataFig7 = fig_path_Figure6_ExtData8af9
#  = fig_path_Figure7_ExtDataFig10 = fig_path_Figure8 = fig_path_ExtDataFig8fk = fig_path_ExtDataFig5
#  = fig_path_all)

save_figs = True
save_stats = True
#%%

"""
load and prepare data
"""

nwb_file_info_df =  lp.get_all_nwb_files_by_condition(nwb_dir_path,  'all', dandi_naming_convention = True)
#%%
all_trial_data_df, df, all_session_data_df = lp.make_trial_df_from_nwb(nwb_file_info_df,
                                                               total_time_window_s = 37,
                                                               baseline_length_s = 7,
                                                               return_session_df = True)
#%%
# lpf.savePickle(all_trial_data_df, os.path.join(pickle_path, 'all_TRIAL_data_df_fromNWB.pkl'))
# lpf.savePickle(all_session_data_df,  os.path.join(pickle_path, 'all_SESSION_data_df_fromNWB.pkl'))
# lpf.savePickle(df,  os.path.join(pickle_path, 'processed_df_fromNWB.pkl'))
# #%%
# all_trial_data_df = lpf.openPickle(os.path.join(pickle_path, 'all_TRIAL_data_df_fromNWB.pkl'))
# all_session_data_df = lpf.openPickle(os.path.join(pickle_path, 'all_SESSION_data_df_fromNWB.pkl'))
# df = lpf.openPickle(os.path.join(pickle_path, 'processed_df_fromNWB.pkl'))
#%%

df_behavior_days_CSplus = lp.get_behavior_days_CSplus_df(df)
df_behavior_trials_CSplus = lp.get_behavior_trials_CSplus_df(df)
df_behavior_trials_CSplus_learners = lp.get_behavior_trials_CSplus_learners_df(df)
nonlearners_list = lpf.get_nonlearners(df_behavior_days_CSplus, conditions_to_exclude = ['60s-10%'])

df_dlight_days_CSplus = lp.subset_dopamine_animals(lp.get_behavior_days_CSplus_df(df))
df_dlight_trials_CSplus = lp.subset_dopamine_animals(df_behavior_trials_CSplus)
df_dlight_trials_CSplus_learners = lp.subset_dopamine_animals(df_behavior_trials_CSplus_learners)
df_dlight_trials_CSplus_learners_full3600 = lp.subset_dopamine_animals(lp.get_behavior_trials_CSplus_learners_df(df, full3600 = True))
df_dlight_trials_CSplus_nonlearners = lp.subset_dopamine_animals(lp.get_behavior_trials_CSplus_nonlearners_df(df))
df_dlight_trials_CSplus_learners_by_trialtype = df_dlight_trials_CSplus_learners.copy()
df_dlight_trials_CSplus_learners_by_trialtype =  lpf.cumLickTrialCount(df_dlight_trials_CSplus_learners_by_trialtype,
                                                                       grouping_var = ['animal',
                                                                                       'cue_type',
                                                                                       'trial_type',],
                                                                       )
df_rewards = df_behavior_trials_CSplus[df_behavior_trials_CSplus['trial_type'] == 'reward'].copy()
df_rewards = lpf.cumLickTrialCount(df_rewards,
                                   grouping_var = ['animal',
                                                   'cue_type',
                                                   'trial_type'],
                                   )
df_first_40_trials = df_behavior_trials_CSplus[df_behavior_trials_CSplus['cue_trial_num'] <=40]
df_first_80_trials = df_behavior_trials_CSplus[df_behavior_trials_CSplus['cue_trial_num'] <=80]
excluded_sorted_df_CSminus_days = lp.get_all_trials_before_40th_CSplus(df)
df_CSminus_new = lp.get_CSminus_renamed_df(df)
df_behavior_trials_CSplus_10percentexcl = df_behavior_trials_CSplus_learners[~df_behavior_trials_CSplus_learners['animal'].isin(['60s-10%D_M2'])] #outlier dopamine 60s-10% see extdata10j
df_dlight_trials_CSplus_10percentexcl = lp.subset_dopamine_animals(df_behavior_trials_CSplus_10percentexcl)