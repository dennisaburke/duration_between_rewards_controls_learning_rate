# -*- coding: utf-8 -*-
"""
@author: Dennis A. Burke (dennis.burke@ucsf.edu ; permanent address: dennis.a.burke AT gmail)

from Burke et al. - Duration between rewards controls the rate of behavioral and dopaminergic learning

FIGURE 8: Emergence of dopamine dip to reward omission is inconsistent with canonical reward prediction error signaling
"""


"""
imports
"""
import os
import matplotlib.pyplot as plt

import functions.load_preprocess  as lp
import functions.default_configs as dc
import functions.figure_functions as ff
import functions.lick_photo_functions as lpf



"""
set paths to data and outputs
"""

#set working directory to script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

nwb_dir_path = r'..\DATA_experimental\001632'
figure_path_root = r'..\FIGURES'
fig_path_Figure8 = os.path.join(figure_path_root, r'Figure8')

save_figs = True
save_stats = True

#%%
"""
load and prepare data
"""
nwb_file_info_df =  lp.get_all_nwb_files_by_condition(nwb_dir_path,
                                                                  ['60s-50%',
                                                                    ],
                                                                  )
all_trial_data_df, df = lp.make_trial_df_from_nwb(nwb_file_info_df,
                                                               total_time_window_s = 37,
                                                               baseline_length_s = 7,
                                                               )
#%%

df_dlight_trials_CSplus_learners = lp.subset_dopamine_animals(lp.get_behavior_trials_CSplus_learners_df(df))
df_dlight_trials_CSplus_learners_by_trialtype = df_dlight_trials_CSplus_learners.copy()
df_dlight_trials_CSplus_learners_by_trialtype =  lpf.cumLickTrialCount(df_dlight_trials_CSplus_learners_by_trialtype, grouping_var = ['animal', 'cue_type', 'trial_type'])

#%%
"""
FIGURE 8
 - PANEL B: EXAMPLE 60S-50% RASTER PLOT AND DA REWARD AND OMISSION HEATMAP
 - PANEL C: EXAMPLE 60S-50% REWARDED AND OMITTED LICK AND DA ACROSS LEARNING
"""
#PANEL B
save_figs = False
omission_example_heatmap_raster = ff.plot_omission_heatmap_raster(df_dlight_trials_CSplus_learners_by_trialtype,
                                                                  animal = dc.cumsum_examples_DA['60s-50%'],
                                                                  axsize = (0.5, 2),
                                                                  show_yticklabels = True,
                                                                  save_fig = save_figs,
                                                                  fig_path  = fig_path_Figure8,)
#PANEL C

ff.plot_example_reward_and_omission_PSTHs_by_trial_bins(df,
                                     dc.cumsum_examples_DA['60s-50%'],
                                     save_fig = save_figs,
                                     fig_path = fig_path_Figure8)

#%%

"""
FIGURE 8
 - PANEL D:
     LEFT: AVERAGE CUE AND OMISSION DA RESPONSE ACROSS OMISSION TRIALS WITH SIGMOID FIT
     RIGHT: CUMSUM CUE AND OMISSION DA RESPONSE ACROSS OMISSION TRIALS
"""
#CUMSUM
save_figs = False
plot_cumsum = True
plot_DA_cue = True
linewidth_DA_cue = 1
linestyle_DA_cue = (0, (4.5, 1.5))
linewidth_0_lick = 1
ylim_DA = [-0.1, 0.232]
conditions_to_plot = ['60s-50%']
df_omissions_trials_subset = df_dlight_trials_CSplus_learners_by_trialtype[df_dlight_trials_CSplus_learners_by_trialtype['cue_trial_num'] <=265].copy()
df_omissions_trials_subset = lpf.cumLickTrialCount(df_omissions_trials_subset, grouping_var = ['animal', 'cue_type', 'trial_type'])
fig_lick_avg_trials_both2, ax_lick_avg_trials_both2, ax_cumsum = ff.plotDALickOverTime(df_omissions_trials_subset,
                                                                          conditions_to_plot = conditions_to_plot,
                                                                          colors_for_conditions = dc.colors_for_conditions,
                                                                          plot_cumsum = plot_cumsum,
                                                                          omissions_and_reward_plot_cue_and_lick = 'omission',
                                                                          linewidth_DA_cue =linewidth_DA_cue,
                                                                          plot_DA_omissions = True,
                                                                          trials_or_rewards = 'reward',
                                                                          trial_norm = True,
                                                                          plot_DA_cue = plot_DA_cue,
                                                                          linewidth_0_lick = linewidth_0_lick,
                                                                          colors_for_conditions_DA = dc.colors_for_conditions,
                                                                          axsize = dc.axsize_timecourse,
                                                                          save_figs = save_figs,
                                                                          ylim_DA = ylim_DA,
                                                                          title = '',
                                                                          fig_path  = fig_path_Figure8,
                                                                          )
#AVERAGE WITH SIGMOID FIT
plot_cumsum = False
plot_DA_cue = True
linewidth_DA_cue = 0.25
linestyle_DA_cue = (0, (4.5, 1.5))
ylim_DA = [-0.1, 0.232]
ylim_DA = [None,None]

cue_means, omission_means, cue_omissions_ax = ff.plotDALickOverTime(df_omissions_trials_subset,
                                                conditions_to_plot = conditions_to_plot,
                                                colors_for_conditions = dc.colors_for_conditions,
                                                plot_cumsum = plot_cumsum,
                                                omissions_and_reward_plot_cue_and_lick = 'omission',
                                                linewidth_DA_cue = linewidth_DA_cue,
                                                linewidth_DA_reward = linewidth_DA_cue,
                                                plot_DA_omissions = True,
                                                trials_or_rewards = 'reward',
                                                trial_norm = True,
                                                plot_DA_cue = plot_DA_cue,
                                                linewidth_0_lick = linewidth_0_lick,
                                                colors_for_conditions_DA = dc.colors_for_conditions,
                                                axsize = dc.axsize_timecourse,
                                                save_figs = False,
                                                ylim_DA = ylim_DA,
                                                title = '',
                                                fig_path  = fig_path_Figure8,
                                                )
x_data_cue = list(cue_means.index)
x_data_omission = list(omission_means.index)
y_data_cue = list(cue_means)
y_data_omission = list(omission_means)
y_fit_cue = lpf.fit_sigmoid_to_data(x_data_cue, y_data_cue,)
y_fit_omission = lpf.fit_sigmoid_to_data(x_data_omission, y_data_omission, curve_goes_negative = True)

cue_omissions_ax.plot(x_data_cue, y_fit_cue['y_fit'], '-', label='fit', color = 'k', linewidth = 0.75)
cue_omissions_ax.plot(x_data_omission, y_fit_omission['y_fit'], '-', label='fit', color = 'k', linewidth = 0.75)
cue_omissions_ax.axvline(y_fit_cue['half_rise_trial'], color = 'k', linewidth = 0.5)
cue_omissions_ax.axvline(y_fit_omission['half_rise_trial'],color = 'k', linewidth = 0.5)
cue_omissions_ax.axvline(y_fit_cue['95%_rise_trial'], color = 'k', linewidth = 0.5)

fig = plt.gcf()
if save_figs:
    fig.savefig(os.path.join(fig_path_Figure8, 'omissionsCueDAwithFitLines.pdf'),
                        bbox_inches = 'tight',
                        transparent = True
                        )

#%%
"""
FIGURE 8
 - PANEL E: INDIVIDUAL CUMSUM CUE AND OMISSION
 - PANEL F: PAIRED BAR GRAPH CUE VS OMISSION LEARNED TRIAL
"""
#PANEL E
save_fig = False
save_stats = False
ylim_lick = [-0.16, None]
ylim_DA = [-0.16, 0.275]
conditions_to_plot = ['60s-50%']

learned_trial_data = ff.getCumSumLearnedTrialsAndPlot(df_dlight_trials_CSplus_learners_by_trialtype,
                                                      conditions_to_plot=conditions_to_plot,
                                                      colors_for_conditions_DA = dc.colors_for_conditions_DA,
                                                      plot_all_individuals = True,
                                                      get_DA_learned_trial = True,
                                                      use_trial_normalized_y = True,
                                                      get_omission_learned_trial = True,
                                                      trial_or_reward_or_omission = 'omission',
                                                      plot_omission = True,
                                                      plot_lick = False,
                                                      plot_DA_reward= False,
                                                      lick_left = False,
                                                      plot_on_2_lines = False,
                                                      sharey_DA = True,
                                                      ylim_lick = ylim_lick,
                                                      ylim_DA = ylim_DA,
                                                      sharex = True,
                                                      sharey = True,
                                                      xlim = [None, 300],
                                                      axsize = dc.axsize_cumsum_all_individuals_DA,
                                                      save_fig = save_fig,
                                                      fig_path = fig_path_Figure8,
                                                      )
#PANEL F
pairs_labels = ['learned_trial_DA', 'learned_trial_DA_omission']
paired_cue_omission_learned_trial_dict = {x: learned_trial_data[x]['60s-50%']
                                          for x in pairs_labels
                                          }
colors_for_cue_omission_bars = {'learned_trial_DA': '#5a9c43', 'learned_trial_DA_omission': '#ad2472'}
omissions_vs_cue = ff.plot_paired_bars_from_dicts_or_list(paired_cue_omission_learned_trial_dict,
                                condition_colors = colors_for_cue_omission_bars,
                                labels = ['learned_trial_DA', 'learned_trial_DA_omission'],
                                ylabel ='omission trials to learn',
                                save_fig = save_fig,
                                fig_path = fig_path_Figure8,
                                title='cue DA vs omissions',
                                axsize = dc.axsize_bars_2,
                                save_stats = save_stats,
                                )