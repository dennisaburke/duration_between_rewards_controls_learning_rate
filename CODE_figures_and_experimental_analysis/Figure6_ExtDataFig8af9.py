# -*- coding: utf-8 -*-
"""
@author: Dennis A. Burke (dennis.burke@ucsf.edu ; permanent address: dennis.a.burke AT gmail)

from Burke et al. - Duration between rewards controls the rate of behavioral and dopaminergic learning

FIGURE 6: Learning rate scaling is not explained by number of experiences per day, context extinction, overall rate of auditory cues, or overall rate of rewards.
"""


"""
imports
"""
import os
import pingouin
import numpy as np
import statsmodels

import functions.load_preprocess  as lp
import functions.default_configs as dc
import functions.figure_functions as ff
import functions.lick_photo_functions as lpf
import functions.stats_functions as sf
import functions.data_wrangling as dw



#%%
"""
set paths to data and outputs
"""

#set working directory to script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

nwb_dir_path = r'..\DATA_experimental\001632'
figure_path_root = r'..\FIGURES'
fig_path_Figure6_ExtData8af9 = os.path.join(figure_path_root, r'Figure6_ExtData8af9')

save_figs = True
save_stats = True
#%%
"""
load and prepare data

"""
nwb_file_info_df =  lp.get_all_nwb_files_by_condition(nwb_dir_path,
                                                                  [ '30s',
                                                                    '60s',
                                                                    '300s',
                                                                    '600s',
                                                                    '60s-few',
                                                                    '60s-few-ctxt',
                                                                    '60s-CSminus',
                                                                    '600s-bgdmilk',
                                                                    ],
                                                                  )
all_trial_data_df, df = lp.make_trial_df_from_nwb(nwb_file_info_df,
                                                               total_time_window_s = 37,
                                                               baseline_length_s = 7,
                                                               )
#%%
df_behavior_trials_CSplus = lp.get_behavior_trials_CSplus_df(df)
df_behavior_days_CSplus = lp.get_behavior_days_CSplus_df(df)
df_behavior_trials_CSplus_learners = lp.get_behavior_trials_CSplus_learners_df(df)
df_dlight_trials_CSplus = lp.subset_dopamine_animals(df_behavior_trials_CSplus)
df_first_40_trials = df_behavior_trials_CSplus[df_behavior_trials_CSplus['cue_trial_num'] <=40]
df_dlight_days_CSplus = lp.subset_dopamine_animals(lp.get_behavior_days_CSplus_df(df))


#%%
"""
FIGURE 6:
 - PANEL B & C: 60s-FEW TRIALS VS 60 & 600s ITI FIRST 40 TRIALS LICK
"""

conditions_to_plot = ['60s',
                      '600s',
                      '60s-few',
                      ]
plot_lick = True
fig_lick_avg_trials_both_first40, ax_lick_avg_trials_both_first40 = ff.plotDALickOverTime(df_first_40_trials,
                                                                                          conditions_to_plot = conditions_to_plot,
                                                                                          colors_for_conditions = dc.colors_for_conditions,
                                                                                          plot_lick = plot_lick,
                                                                                          axsize = dc.axsize_timecourse,
                                                                                          save_figs = save_figs,
                                                                                          fig_path = fig_path_Figure6_ExtData8af9,
                                                                                          )
range_to_plot = {'600s': [36, 40],
                 '60s': [36, 40],
                 '60s-few':[36, 40]}
order_to_plot = ['60s',
                '60s-few',
                '600s',
                ]
fig_lickratemeans, ax_lickratemeans = ff.compare_asymptote_bars(df_behavior_trials_CSplus,
                                                               conditions_to_plot = order_to_plot,
                                                               range_to_plot = range_to_plot,
                                                               condition_colors = dc.colors_for_conditions,
                                                               ylabel ='lick rate to cue (Hz)\ntrials 36 - 40',
                                                               save_fig = save_figs,
                                                               fig_path = fig_path_Figure6_ExtData8af9,
                                                               ylim = [None, None],
                                                               axsize = dc.axsize_bars_3,
                                                               )

#%%
"""
FIGURE 6:
 - PANELS E & F: 60s-FEW W/ CONTEXT EXTINCTION VS 60 & 600s ITI FIRST 40 TRIALS
"""
conditions_to_plot = ['60s',
                      '600s',
                      '60s-few-ctxt',
                      ]
plot_lick = True
fig_lick_avg_trials_both_first40, ax_lick_avg_trials_both_first40 = ff.plotDALickOverTime(df_first_40_trials,
                                                                                          conditions_to_plot = conditions_to_plot,
                                                                                          colors_for_conditions = dc.colors_for_conditions,
                                                                                          plot_lick = plot_lick,
                                                                                          axsize = dc.axsize_timecourse,
                                                                                          save_figs = save_figs,
                                                                                          fig_path = fig_path_Figure6_ExtData8af9,
                                                                                          )
range_to_plot = {'60s': [36, 40],
                 '60s-few-ctxt': [36, 40],
                 '600s': [36, 40],
                 }
order_to_plot = ['60s',
                '60s-few-ctxt',
                '600s',
                ]
fig_lickratemeans, ax_lickratemeans =  ff.compare_asymptote_bars(df_behavior_trials_CSplus,
                                                               conditions_to_plot = order_to_plot,
                                                               range_to_plot = range_to_plot,
                                                               condition_colors = dc.colors_for_conditions,
                                                               ylabel ='lick rate to cue (Hz)\ntrials 36 - 40',
                                                               save_fig = save_figs,
                                                               fig_path = fig_path_Figure6_ExtData8af9,
                                                               axsize = dc.axsize_bars_3,
                                                               )

#%%
"""
FIGURE 6:
 - PANELS H & I: 60s-CSMINUS VS 60 & 600s ITI FIRST 40 TRIALS
"""
conditions_to_plot =['60s',
                     '600s',
                     '60s-CSminus',
                     ]
plot_lick = True
fig_lick_avg_trials_both_first40, ax_lick_avg_trials_both_first40 = ff.plotDALickOverTime(df_first_40_trials,
                                                                                          conditions_to_plot = conditions_to_plot,
                                                                                          colors_for_conditions = dc.colors_for_conditions,
                                                                                          plot_lick = plot_lick,
                                                                                          axsize = dc.axsize_timecourse,
                                                                                          save_figs = save_figs,
                                                                                          fig_path = fig_path_Figure6_ExtData8af9,
                                                                                          )
range_to_plot = {'600s': [36, 40],
                 '60s': [36, 40],
                 '60s-CSminus': [36, 40],
                 }

order_to_plot = ['60s',
                '60s-CSminus',
                '600s',
                ]
fig_lickratemeans, ax_lickratemeans = ff.compare_asymptote_bars(df_behavior_trials_CSplus,
                                                                conditions_to_plot = order_to_plot,
                                                                range_to_plot = range_to_plot,
                                                                condition_colors = dc.colors_for_conditions,
                                                                ylabel ='lick rate to cue (Hz)\ntrials 36 - 40',
                                                                save_fig = save_figs,
                                                                fig_path = fig_path_Figure6_ExtData8af9,
                                                                axsize = dc.axsize_bars_3
                                                                )
#%%
"""
FIGURE 6:
 - PANELS K & L: 600s W/ BACKGROUND MILK VS 60 & 600s ITI FIRST 40 TRIALS
"""

conditions_to_plot = ['600s',
                      '60s',
                      '600s-bgdmilk',
                      ]
plot_lick = True
df_behavior_trials_CSplusfig_lick_avg_trials_both_first40, ax_lick_avg_trials_both_first40 = ff.plotDALickOverTime(df_first_40_trials,
                                                                                          conditions_to_plot = conditions_to_plot,
                                                                                          colors_for_conditions = dc.colors_for_conditions,
                                                                                          plot_lick = plot_lick,
                                                                                          axsize = dc.axsize_timecourse,
                                                                                          save_figs = save_figs,
                                                                                          fig_path = fig_path_Figure6_ExtData8af9,
                                                                                          )
range_to_plot = {'600s': [36, 40],
                 '60s': [36, 40],
                 '600s-bgdmilk': [36, 40],
                 }

order_to_plot = ['60s',
                '600s-bgdmilk',
                '600s',
                ]
fig_lickratemeans, ax_lickratemeans =  ff.compare_asymptote_bars(df_behavior_trials_CSplus,
                                                               conditions_to_plot = order_to_plot,
                                                               range_to_plot = range_to_plot,
                                                               condition_colors = dc.colors_for_conditions,
                                                               ylabel ='lick rate to cue (Hz)\ntrials 36 - 40',
                                                               save_stats = save_stats,
                                                               save_fig = save_figs,
                                                               fig_path = fig_path_Figure6_ExtData8af9,
                                                               axsize = dc.axsize_bars_3
                                                               )
#%%
"""
FIGURE 6:
 - STATS FOR C, F, I, L
"""
conditions_to_subset = ['60s',
                      '600s',
                      '60s-few',
                      '60s-few-ctxt',
                      '60s-CSminus',
                      '600s-bgdmilk',
                      ]

range_to_plot = {'600s': [36, 40],
                 '60s': [36, 40],
                 '60s-few': [36, 40],
                 '60s-few-ctxt': [36, 40],
                 '60s-CSminus': [36, 40],
                 '600s-bgdmilk': [36, 40],
                 }
plot_lick_or_DA = 'lick'

condition_means_controls_dict  = dw.get_mean_values_from_trial_range(df_behavior_trials_CSplus,
                                            conditions_to_subset = conditions_to_subset,
                                            range_to_subset = range_to_plot,
                                            data_to_return = plot_lick_or_DA,
                                            return_as_regular_dict = True,
                                            return_as_df = False,
                                            return_as_nested_dict = False)
test_conditions = ['60s-few',
                   '60s-few-ctxt',
                   '60s-CSminus',
                   '600s-bgdmilk',
                   ]
pvals_all = []
pvals_vs_60 = []
pvals_vs_600 = []
for condition in test_conditions:
    t_test_60 = sf.t_test_from_dict_list_or_df(condition_means_controls_dict,
                                            ['60s', condition],
                                            assume_equal_var = False,
                                            data_is_regular_dict = True,
                                            )
    pvals_all.append(t_test_60['p'])
    pvals_vs_60.append(t_test_60['p'])
    t_test_600 = sf.t_test_from_dict_list_or_df(condition_means_controls_dict,
                                            ['600s', condition],
                                            assume_equal_var = False,
                                            data_is_regular_dict = True,
                                            )
    pvals_all.append(t_test_600['p'])
    pvals_vs_600.append(t_test_600['p'])

corrected_ps = statsmodels.stats.multitest.multipletests(pvals_all,
                                                         method= 'fdr_by',
                                                         )

#%%
"""
EXTENDED DATA FIGURE 8:
 - PANEL A & B: 60s-FEW TRIALS VS 60 & 600s ITI FIRST 40 TRIALS CUE DA
 - PANEL C & D: 60s-FEW TRIALS VS 60 & 600s ITI FIRST 6 TRIALS CUE DA AVERAGE AND CUMSUM
"""
#PANEL A

linewidth_DA_cue = 0.5
conditions_to_plot = ['60s',
                      '600s',
                      '60s-few',
                      ]
plot_lick = False
plot_DA_cue = True
fig_cueDA_avg_trials_all3, ax_cueDA_avg_trials_all3 = ff.plotDALickOverTime(df_dlight_trials_CSplus,
                                                                          conditions_to_plot = conditions_to_plot,
                                                                          colors_for_conditions = dc.colors_for_conditions,
                                                                          plot_lick = plot_lick,
                                                                          plot_DA_cue =plot_DA_cue,
                                                                          linewidth_DA_cue =linewidth_DA_cue,
                                                                          linewidth_0_lick = 1,
                                                                          axsize = dc.axsize_timecourse,
                                                                          save_figs = save_figs,
                                                                          title = 'cue',
                                                                          fig_path  = fig_path_Figure6_ExtData8af9,
                                                                          ylim_DA = [-0.2, None],
                                                                          xlim_DA = [0.9, 40],
                                                                          x_MulitpleLocator = 10,
                                                                          )
#PANEL B
plot_lick_or_DA = 'DA'
range_to_plot = {'600s': [36, 40],
                 '60s': [36, 40],
                 '60s-few':[36, 40]}
order_to_plot = ['60s',
                '60s-few',
                '600s',
                ]
fig_lickratemeans, ax_lickratemeans = ff.compare_asymptote_bars(df_dlight_trials_CSplus,
                                                               conditions_to_plot = order_to_plot,
                                                               range_to_plot = range_to_plot,
                                                               condition_colors = dc.colors_for_conditions,
                                                               plot_lick_or_DA = plot_lick_or_DA,
                                                               ylabel ='cue DA\ntrials 36 - 40',
                                                               save_stats = save_stats,
                                                               save_fig = save_figs,
                                                               fig_path = fig_path_Figure6_ExtData8af9,
                                                               ylim = [None, 0.8],
                                                               axsize = dc.axsize_bars_3,
                                                               )
#stats for B
plot_lick_or_DA = 'DA'

DA_first40_dict  = dw.get_mean_values_from_trial_range(df_dlight_trials_CSplus,
                                            conditions_to_subset = order_to_plot,
                                            range_to_subset = range_to_plot,
                                            data_to_return = plot_lick_or_DA,
                                            return_as_regular_dict = True,
                                            return_as_df = False,
                                            return_as_nested_dict = False)
t_test_60vsfew = sf.t_test_from_dict_list_or_df(DA_first40_dict,
                                        ['60s', '60s-few',],
                                        assume_equal_var = False,
                                        data_is_regular_dict = True,
                                        )

t_test_600vsfew = sf.t_test_from_dict_list_or_df(DA_first40_dict,
                                        ['600s', '60s-few',],
                                        assume_equal_var = False,
                                        data_is_regular_dict = True,
                                        )
#PANEL C
fig_cueDA_avg_trials_all3, ax_cueDA_avg_trials_all3 = ff.plotDALickOverTime(df_dlight_trials_CSplus,
                                                                          conditions_to_plot = conditions_to_plot,
                                                                          colors_for_conditions = dc.colors_for_conditions,
                                                                          plot_lick = plot_lick,
                                                                          plot_DA_cue = plot_DA_cue,
                                                                          linewidth_DA_cue = linewidth_DA_cue,
                                                                          linewidth_0_lick = 1,
                                                                          axsize = dc.axsize_timecourse,
                                                                          save_figs = save_figs,
                                                                          title = 'cue-first6',
                                                                          fig_path  = fig_path_Figure6_ExtData8af9,
                                                                          ylim_DA = [-0.13, 0.5],
                                                                          xlim_DA = [0.9, 6],
                                                                          x_MulitpleLocator = 2,
                                                                          )
#PANEL D
fig_cueDA_avg_trials_all3, ax_cueDA_avg_trials_all3 = ff.plotDALickOverTime(df_dlight_trials_CSplus,
                                                                          conditions_to_plot = conditions_to_plot,
                                                                          colors_for_conditions = dc.colors_for_conditions,
                                                                          plot_cumsum = True,
                                                                          plot_lick = plot_lick,
                                                                          plot_DA_cue =plot_DA_cue,
                                                                          linewidth_DA_cue =linewidth_DA_cue,
                                                                          linewidth_0_lick = 1,
                                                                          axsize = dc.axsize_timecourse,
                                                                          save_figs = save_figs,
                                                                          title = 'cue-first6_cumsum',
                                                                          fig_path  = fig_path_Figure6_ExtData8af9,
                                                                          ylim_DA = [-0.366, 1.865],
                                                                          xlim_DA = [0.9, 6],
                                                                          x_MulitpleLocator = 2,
                                                                          )

#%%
"""
EXTENDED DATA FIGURE 8:
 - PANEL E: 60s-FEW TRIALS VS 60 & 600s ITI FIRST 6 TRIALS PER DAY CUE DA AND LICK
"""
plot_lick = True
plot_DA_cue = False
plot_days = True
plot_symbols = True
shaded_error = False
conditions_to_plot = ['60s',
                      '600s',
                      '60s-few',
                      ]
df_dlight_trials_CSplus_first6trialsday = df_dlight_days_CSplus[df_dlight_days_CSplus['trial_num'] <=6].copy()
df_behavior_trials_CSplus_first6trialday = df_behavior_days_CSplus[df_behavior_days_CSplus['trial_num'] <=6].copy()
#normalize da with all conditioning
DA_normalization_all = dw.get_DA_normalization_cue_reward(df_dlight_days_CSplus,
                                                         )
#TOP PLOT: CUE DA
plot_lick = False
plot_DA_cue = True
fig_cueDA_avg_trials_all3, ax_cueDA_avg_trials_all3 = ff.plotDALickOverTime(df_dlight_trials_CSplus_first6trialsday,
                                                                          conditions_to_plot = conditions_to_plot,
                                                                          colors_for_conditions = dc.colors_for_conditions,
                                                                          plot_lick = plot_lick,
                                                                          plot_DA_cue =plot_DA_cue,
                                                                          plot_days = plot_days,
                                                                          shaded_error =shaded_error,
                                                                          plot_symbols = plot_symbols,
                                                                          linewidth_DA_cue = linewidth_DA_cue,
                                                                          axsize = dc.axsize_timecourse,
                                                                          save_figs = save_figs,
                                                                          title = 'first 6 a day cue DA',
                                                                          fig_path  = fig_path_Figure6_ExtData8af9,
                                                                          ylim_lick = [None, 6],
                                                                          DA_normalization_precalced = DA_normalization_all,
                                                                          )
#BOTTOM PLOT: LICKING
plot_lick = True
plot_DA_cue = False
fig_lick_avg_trials_all3, ax_lick_avg_trials_all3 = ff.plotDALickOverTime(df_behavior_trials_CSplus_first6trialday,
                                                                          conditions_to_plot = conditions_to_plot,
                                                                          colors_for_conditions = dc.colors_for_conditions,
                                                                          plot_lick = plot_lick,
                                                                          plot_DA_cue =plot_DA_cue,
                                                                          plot_days = plot_days,
                                                                          shaded_error= shaded_error,
                                                                          plot_symbols = plot_symbols,
                                                                          axsize = dc.axsize_timecourse,
                                                                          save_figs = save_figs,
                                                                          title = 'first 6 a day all behavior',
                                                                          fig_path  = fig_path_Figure6_ExtData8af9,
                                                                          )
#%%
"""
EXTENDED DATA FIGURE 8:
 - PANEL F (LEFT): REWARD LICK BOUT DURATION AND PSTH 30S ITI AND 60S ITI
 (bar graphs for right side of this panel generated in 'ExtendedDataFig8fk.py' script)
"""
#TOP
condition_to_plot = '60s'
ff.plot_consumption_PSTH_beginning_end(df_behavior_days_CSplus,
                                    condition_to_plot,
                                    num_to_subset = 6,
                                    color_lick_PSTH= {'beginning':'k',
                                                      'end': dc.colors_for_conditions[condition_to_plot]},
                                    save_fig = save_figs,
                                    fig_path = fig_path_Figure6_ExtData8af9)
#BOTTOM
condition_to_plot = '30s'
ff.plot_consumption_PSTH_beginning_end(df_behavior_days_CSplus,
                                    condition_to_plot,
                                    num_to_subset = 6,
                                    color_lick_PSTH= {'beginning':'k',
                                                      'end': dc.colors_for_conditions[condition_to_plot]},
                                    save_fig = save_figs,
                                    fig_path = fig_path_Figure6_ExtData8af9)
#%%
"""
EXTENDED DATA FIGURE 9:
 - PANEL A: INDIVIDUAL CUMSUMS 60s-CSminus CS+ TRIALS
 - PANEL B: TRIALS TO LEARN 60S-CSMINUS VS 60 & 600S ITI
 - PANEL C: INDIVIDUAL CUMSUMS 60s-CSminus CS- TRIALS
 - PANEL D: CS+ vs CS- LICKS ACROSS DAYS
"""
#PANEL A
supplment_all_CSminus_cumsum = ff.getCumSumLearnedTrialsAndPlot(df_behavior_trials_CSplus,
                                                                  conditions_to_plot = ['60s-CSminus'],
                                                                  colors_for_conditions = dc.colors_for_conditions,
                                                                  linewidth_learned_trial = 0.25,
                                                                  plot_all_individuals = True,
                                                                  axsize = dc.axsize_cumsum_all_individuals,
                                                                  sharex = 'row',
                                                                  save_fig = save_figs,
                                                                  fig_path = fig_path_Figure6_ExtData8af9,
                                                                  )
#PANEL B
conditions_to_plot =['60s',
                     '60s-CSminus',
                     '600s',
                     ]

learned_trial_data = ff.getCumSumLearnedTrialsAndPlot(df_behavior_trials_CSplus_learners,
                                                      conditions_to_plot= conditions_to_plot,
                                                      )
figax_learned_trial = ff.plotBarsFromDict(learned_trial_data['learned_trial_lick'],
                                           condition_colors = dc.colors_for_conditions,
                                           ylabel ='trials to learn',
                                           data_is_nested_dict = True,
                                           plot_individuals = True,
                                           plot_stats = True,
                                           save_stats = False,
                                           plot_sem = True,
                                           logscale = True,
                                           axsize = dc.axsize_bars_3,
                                           save_fig = save_figs,
                                           fig_path = fig_path_Figure6_ExtData8af9,
                                           )
#stats for B
t_test_60vsCSminus = sf.t_test_from_dict_list_or_df(learned_trial_data['learned_trial_lick'],
                                        ['60s', '60s-CSminus',],
                                        assume_equal_var = False,
                                        data_is_nested_dict = True,
                                        )

t_test_600vsCSminus = sf.t_test_from_dict_list_or_df(learned_trial_data['learned_trial_lick'],
                                        ['600s', '60s-CSminus',],
                                        assume_equal_var = False,
                                        data_is_nested_dict = True,
                                        )
#PANEL C
excluded_sorted_df_CSminus_days = lp.get_all_trials_before_40th_CSplus(df)
supplment_all_CSminus_cumsum = ff.getCumSumLearnedTrialsAndPlot(excluded_sorted_df_CSminus_days,
                                                                  conditions_to_plot = ['60s-CSminus'],
                                                                  colors_for_conditions = dc.colors_for_conditions,
                                                                  linewidth_learned_trial = 0.25,
                                                                  use_trial_normalized_y = True,
                                                                  plot_all_individuals = True,
                                                                  axsize = dc.axsize_cumsum_all_individuals,
                                                                  sharex = True,
                                                                  save_fig = save_figs,
                                                                  fig_path = fig_path_Figure6_ExtData8af9,
                                                                  cue_type = 'CS_minus',
                                                                  )

#PANEL D
df_CSminus_new = lp.get_CSminus_renamed_df(df)

colors_CSminus = {'CS_plus': '#45b97c',
               'CS_minus': '#45b97c'}

conditions_to_plot = ['CS_minus', 'CS_plus',]
plot_days = True

plot_cumsum = False
plot_lick = True
linewidth_lick = 0.5
plot_days = True
shaded_error = False
plot_symbols = True
linewidth_0_lick = 1
markersize = 5
fig_CSminus_days, ax_CSminus_days = ff.plotDALickOverTime(df_CSminus_new,
                                                          conditions_to_plot = conditions_to_plot,
                                                          colors_for_conditions = colors_CSminus,
                                                          plot_cumsum = plot_cumsum,
                                                          plot_lick = plot_lick,
                                                          plot_days = plot_days,
                                                          shaded_error = shaded_error,
                                                          markersize = markersize,
                                                          plot_symbols = plot_symbols,
                                                          linewidth_lick = linewidth_lick,
                                                          linewidth_0_lick = linewidth_0_lick,
                                                          axsize = dc.axsize_timecourse,
                                                          save_figs = save_figs,
                                                          fig_path = fig_path_Figure6_ExtData8af9,
                                                          )

#%%
"""
EXTENDED DATA FIGURE 9:
 - PANEL E: INDIVIDUAL CUE LICK CUMSUMS 600S-BGDMILK
"""
supplment_all_bgdmilk_cumsum = ff.getCumSumLearnedTrialsAndPlot(df_behavior_trials_CSplus,
                                                                  conditions_to_plot = ['600s-bgdmilk'],
                                                                  colors_for_conditions = dc.colors_for_conditions,
                                                                  linewidth_learned_trial = 0.25,
                                                                  plot_all_individuals = True,
                                                                  axsize = dc.axsize_cumsum_all_individuals,
                                                                  sharex = 'row',
                                                                  save_fig = save_figs,
                                                                  fig_path = fig_path_Figure6_ExtData8af9,
                                                                  )
#%%
"""
EXTENDED DATA FIGURE 9:
 - PANEL F: PREDICTED TRIALS TO LEARN 600S-BGDMILK
 - PANEL G: TRIALS TO LEARN 600S-BGDMILK VS 600S ITI
"""
#PANEL F

#get fit line to get predicted trial to learn for bgdmilk
conditions_fitline = ['30s',
                        '60s',
                        '300s',
                        '600s',
                        ]
conditions_to_plot = ['30s',
                        '60s',
                        '300s',
                        '600s',
                        '600s-bgdmilk',
                        ]
learned_trial_data = ff.getCumSumLearnedTrialsAndPlot(df_behavior_trials_CSplus_learners,
                                                      conditions_to_plot=conditions_to_plot,
                                                      )
learned_trial_lick_fitline = learned_trial_data['learned_trial_lick']
ylim = [None, 320]
xlim = [10, 1001]

IRI_vs_trials_to_learn = ff.plot_IRI_vs_learned_trial_scatter(learned_trial_lick_fitline,
                                      colors_for_conditions = dc.colors_for_conditions,
                                      nested_dict = True,
                                      conditions_to_plot = conditions_to_plot,
                                      conditions_for_fitline = conditions_fitline,
                                      alpha = 0.5,
                                      xlim = xlim,
                                      ylim = ylim,
                                      axsize = dc.axsize_timecourse,
                                      save_fig = False,
                                      fig_path = fig_path_Figure6_ExtData8af9,
                                      )
IRI_to_predict_allrewards = 204.25
linear_Y_predict_bgdmilk = lpf.predict_trials_to_learn_from_IRI(IRI_to_predict_allrewards, IRI_vs_trials_to_learn['fit_line'])

IRI_to_predict_sucroseonly = 604.25
linear_Y_predict_bgdmilk_sucroseonly = lpf.predict_trials_to_learn_from_IRI(IRI_to_predict_sucroseonly, IRI_vs_trials_to_learn['fit_line'])
IRI_vs_trials_to_learn['ax'].axhline(linear_Y_predict_bgdmilk,
                                    linestyle = (2,(4,4)),
                                    color = 'k',
                                    linewidth = 0.5,
                                    alpha = 1,
                                    )
IRI_vs_trials_to_learn['ax'].axhline(linear_Y_predict_bgdmilk_sucroseonly,
                                    linestyle = (2,(4,4)),
                                    color = 'k',
                                    linewidth = 0.5,
                                    alpha = 1,
                                    )
if save_figs:
    IRI_vs_trials_to_learn['fig'].savefig(os.path.join(fig_path_Figure6_ExtData8af9,
                         'bgdMilk-IRIvsLearnedTrialScatter_predicted_hlines.pdf'),
                bbox_inches = 'tight',
                transparent = True,
                )
#PANEL G
conditions_to_plot_learned_trial =['600s',
                                   '600s-bgdmilk',
                                   ]

learned_trial_lick_subset = {k: v for k, v in learned_trial_lick_fitline.items() if k in conditions_to_plot_learned_trial}
figax_learned_trial = ff.plotBarsFromDict(learned_trial_lick_subset,
                                           condition_colors = dc.colors_for_conditions,
                                           ylabel ='trials to learn - no predicted line',
                                           order_to_plot = ['600s-bgdmilk',
                                                            '600s',],
                                           ylim = [None, 20],
                                           ax_to_plot = None,
                                           axsize = dc.axsize_bars_2,
                                           save_stats = save_stats,
                                           save_fig = save_figs,
                                           fig_path = fig_path_Figure6_ExtData8af9,
                                           )
#stats for F

trials_to_learn_bgdmilk = list(learned_trial_lick_subset['600s-bgdmilk'].values())
linear_Y_observed_bgdmilk = np.mean(trials_to_learn_bgdmilk)
one_sample_ttestbgdmilk_from_predicted_allrewards = pingouin.ttest(trials_to_learn_bgdmilk, linear_Y_predict_bgdmilk)
one_sample_ttestbgdmilk_from_predicted_sucroseonly = pingouin.ttest(trials_to_learn_bgdmilk, linear_Y_predict_bgdmilk_sucroseonly)

#%%
"""
EXTENDED DATA FIGURE 9:
 - PANEL H: REWARD CONSUMPTION LICK PSTH 600S-BGDMILK SUCROSE VS MILK
"""
ax_reward_psth_comparison = ff.plot_cued_reward_and_background_reward_consumption_PSTH(df,
                                                                                        color_lick_PSTH= {'bgd_reward':'black',
                                                                                                          'reward': '#605656'},
                                                                                        axsize = (1.69, 1),
                                                                                        save_fig = False,
                                                                                        fig_path ='')