# -*- coding: utf-8 -*-
"""
@author: Dennis A. Burke (dennis.burke@ucsf.edu ; permanent address: dennis.a.burke AT gmail)

from Burke et al. - Duration between rewards controls the rate of behavioral and dopaminergic learning

FIGURE 4: Learning rate increases, but not proportionally, with extreme trial spacing.

"""
"""
imports
"""
import os
import numpy as np
import pingouin
import matplotlib.pyplot as plt
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
fig_path_Figure4_ExtDataFig6fp = os.path.join(figure_path_root, r'Figure4_ExtDataFig6fp')

save_figs = False
save_stats = False
#%%
"""
load and prepare data

"""

nwb_dir_path = r'E:\NWB_output'
nwb_file_info_df =  lp.get_all_nwb_files_by_condition(nwb_dir_path,  ['30s', '60s', '300s', '600s', '3600s'])
all_trial_data_df, df = lp.make_trial_df_from_nwb(nwb_file_info_df,
                                                               total_time_window_s = 37,
                                                               baseline_length_s = 7,
                                                               )
#df_dlight_trials_CSplus = lp.get_behavior_trials_CSplus_df(df)
df_behavior_trials_CSplus = lp.get_behavior_trials_CSplus_df(df)
df_behavior_trials_CSplus_learners = lp.get_behavior_trials_CSplus_learners_df(df)
df_dlight_trials_CSplus_learners = lp.subset_dopamine_animals(df_behavior_trials_CSplus_learners)
df_dlight_trials_CSplus_learners_full3600 = lp.subset_dopamine_animals(lp.get_behavior_trials_CSplus_learners_df(df, full3600 = True))


#%%
"""
FIGURE 4:
 - PANEL B: EXAMPLE 3600s ITI LICK RASTER PSTH HEATMAP
 - PANEL C: EXAMPLE 3600s CUMSUM
"""
plot_lick = True
plot_DA = True
days = [1,4]

fig_ex_3600s, ax_ex_3600s = ff.plotPSTHbyDay(all_trial_data_df,
                                            dc.cumsum_examples_DA['3600s'],
                                            color_lick_PSTH = dc.colors_for_conditions['3600s'],
                                            color_DA_PSTH = dc.colors_for_conditions_DA['3600s'],
                                            days_to_plot = days,
                                            plot_lick_raster = plot_lick,
                                            plot_lick_PSTH = plot_lick,
                                            plot_DA_heatmap = plot_DA,
                                            plot_DA_PSTH = plot_DA,
                                            axsize = dc.axsize_raster_PSTH_panel,
                                            fig_path = fig_path_Figure4_ExtDataFig6fp,
                                            save_fig = save_figs,
                                            )
use_trial_normalized_y = True
ylim_left = [None, 2.25]
ylim_right = [None, None]
lick_left = True
fig_ex_3600s_cumsum, ax_ex_3600s_cumsum = ff.plotExampleCumsumLearnedTrial(df_behavior_trials_CSplus,
                                                                                animal = dc.cumsum_examples_DA['3600s'],
                                                                                color_lick = dc.colors_for_conditions['3600s'],
                                                                                color_da = dc.colors_for_conditions_DA['3600s'],
                                                                                plot_lick = plot_lick,
                                                                                plot_da = plot_DA,
                                                                                use_trial_normalized_y = use_trial_normalized_y,
                                                                                axsize = dc.axsize_cumsum_examples,
                                                                                ylim_left = ylim_left,
                                                                                ylim_right = ylim_right,
                                                                                lick_left = lick_left,
                                                                                save_fig = save_figs,
                                                                                fig_path = fig_path_Figure4_ExtDataFig6fp,
                                                                                )
#%%
"""
FIGURE 4:
 - PANEL D: CUE DA AND LICK 3600s ITI
"""
plot_lick = True
plot_DA_cue = True
plot_cumsum = True
peak_or_auc = 'auc'
linewidth_lick = 1
linewidth_DA_cue = 1
lick_left = True
xlim_lick = [1,8.4]
trial_norm = True
fig_cueDA_lick_3600, ax_cueDA_lick_3600 = ff.plotDALickOverTime(df_behavior_trials_CSplus,
                                                                conditions_to_plot = ['3600s'],
                                                                colors_for_conditions = dc.colors_for_conditions,
                                                                plot_cumsum = plot_cumsum,
                                                                plot_lick = plot_lick,
                                                                plot_DA_cue = plot_DA_cue,
                                                                trial_norm = trial_norm,
                                                                peak_or_auc = peak_or_auc,
                                                                linewidth_lick = linewidth_lick,
                                                                linewidth_DA_cue = linewidth_DA_cue,
                                                                axsize = dc.axsize_timecourse,
                                                                xlim_lick = xlim_lick,
                                                                lick_left = lick_left,
                                                                save_figs = save_figs,
                                                                fig_path  = fig_path_Figure4_ExtDataFig6fp,
                                                                )
#%%
"""
FIGURE 4:
 - PANEL E: 3600S TRIALS TO LEARN LICK
 - PANEL H: 3600S TRIALS TO LEARN CUE DA
 - PANEL F: TRIALS TO LEARN VS IRI SCATTER WITH FIT LINE INC 3600 IN SCATTER BUT NOT FIT

"""
#PANEL E
#predict trials to learn for 3600s ITI mice
conditions_to_plot = ['30s',
                      '60s',
                      '300s',
                      '600s',
                      ]
learned_trial_data = ff.getCumSumLearnedTrialsAndPlot(df_behavior_trials_CSplus_learners,
                                                      conditions_to_plot=conditions_to_plot,
                                                      )
IRI_vs_trials_to_learn = ff.plot_IRI_vs_learned_trial_scatter(learned_trial_data['learned_trial_lick'],
                                      conditions_to_plot = conditions_to_plot,
                                      plot_fig = False,
                                      )
ITI_to_predict = 3600
IRI_to_predict = ITI_to_predict + 4.25 #trial + consumption period in s
linear_Y_predict_3600 = lpf.predict_trials_to_learn_from_IRI(IRI_to_predict,
                                                        IRI_vs_trials_to_learn['fit_line'],
                                                        )
conditions_to_plot = ['3600s',]
learned_trial_data = ff.getCumSumLearnedTrialsAndPlot(df_behavior_trials_CSplus_learners,
                                                      conditions_to_plot=conditions_to_plot,
                                                      get_DA_learned_trial = True,
                                                      )
learned_trial_lick = learned_trial_data['learned_trial_lick']
figax_learned_trial = ff.plotBarsFromDict(learned_trial_lick,
                                           condition_colors = dc.colors_for_conditions,
                                           ylabel ='trials to learn',
                                           axsize = dc.axsize_bars_1,
                                           save_fig = False,
                                           fig_path = fig_path_Figure4_ExtDataFig6fp,
                                           )
figax_learned_trial['ax'].axhline(linear_Y_predict_3600, color = 'black',
                                  linestyle = (0, (4,2)),
                                  linewidth = 0.5
                                  )
if save_figs:
    figax_learned_trial['fig'].savefig(os.path.join(fig_path_Figure4_ExtDataFig6fp, 'trials_to_learn_with_predicted_line'+ '_'.join(conditions_to_plot) +'.pdf'),
                    transparent = True,
                    bbox_inches = 'tight',
                    )
trials_to_learn_3600 = list(learned_trial_lick['3600s'].values())
linear_Y_observed = np.mean(trials_to_learn_3600)
one_sample_ttest3600_from_predicted = pingouin.ttest(trials_to_learn_3600, linear_Y_predict_3600)

#PANEL H
learned_trial_DA = learned_trial_data['learned_trial_DA']
figax_learned_trial = ff.plotBarsFromDict(learned_trial_DA,
                                           condition_colors = dc.colors_for_conditions,
                                           ylabel ='DA learned trial',
                                           save_stats = save_stats,
                                           ylim = figax_learned_trial['ax'].get_ylim(),
                                           axsize = dc.axsize_bars_1,
                                           save_fig = save_figs,
                                           fig_path = fig_path_Figure4_ExtDataFig6fp,
                                           )

#PANEL F
conditions_to_plot = ['30s',
                      '60s',
                      '300s',
                      '600s',
                      '3600s',
                      ]
conditions_for_fitline = ['30s',
                          '60s',
                          '300s',
                          '600s',
                          ]
xlim = [10, 5000]
ylim = [0.8, 320]
learned_trial_data = ff.getCumSumLearnedTrialsAndPlot(df_behavior_trials_CSplus_learners,
                                                      conditions_to_plot=conditions_to_plot,
                                                      )
learned_trial_lick = learned_trial_data['learned_trial_lick']
fig, ax = plt.subplots()

ax.axhline(linear_Y_predict_3600,
            linestyle = (2,(4,4)),
            color = 'k',
            linewidth = 0.5,
            alpha = 1)
ax.axhline(linear_Y_observed,
            linestyle = (2,(4,4)),
            color = '#4362AD',
            linewidth = 0.5,
            alpha = 1)
IRI_vs_trials_to_learn = ff.plot_IRI_vs_learned_trial_scatter(learned_trial_lick,
                                      colors_for_conditions = dc.colors_for_conditions,
                                      conditions_to_plot = conditions_to_plot,
                                      conditions_for_fitline = conditions_for_fitline,
                                      alpha = 0.5,
                                      xlim = xlim,
                                      ylim = ylim,
                                      ax_to_plot = ax,
                                      axsize = dc.axsize_timecourse,
                                      save_fig = save_figs,
                                      fig_path = fig_path_Figure4_ExtDataFig6fp,
                                      )
#%%
"""
FIGURE 4:
 - PANEL G: SCALED LICKING ALL ITIS TOGETHER INCL 3600
"""
plot_lick = True
scaled = True
scaled_align_to_1 = True
linewidth_0_lick = 1
plot_cumsum = False
ylim_lick = [None, 6]
conditions_to_plot = ['60s',
                      '600s',
                      '300s',
                      '30s',
                      '3600s',]
fig_lick_trials_scaled_all, ax_lick_trials_scaled_all = ff.plotDALickOverTime(df_behavior_trials_CSplus,
                                                                          conditions_to_plot = conditions_to_plot,
                                                                          colors_for_conditions = dc.colors_for_conditions,
                                                                          plot_cumsum = plot_cumsum,
                                                                          plot_lick = plot_lick,
                                                                          linewidth_0_lick = linewidth_0_lick,
                                                                          ylim_lick = ylim_lick,
                                                                          scaled = scaled,
                                                                          scaled_align_to_1 = scaled_align_to_1,
                                                                          axsize = dc.axsize_timecourse,
                                                                          save_figs = save_figs,
                                                                          fig_path = fig_path_Figure4_ExtDataFig6fp,
                                                                          )
#%%
"""
FIGURE 4:
 - PANEL I: SCALED CUE DA CUMSUM TIMECOURSE 60 600 3600
"""
scaled = True
scaled_align_to_1 = True
plot_lick = False
plot_DA_cue = True
plot_cumsum = True
peak_or_auc = 'auc'
linewidth_0_lick = 1
linewidth_DA_cue = 0.5
fig_cueDA_cumsum_scale, ax_cueDA_cumsum_scale = ff.plotDALickOverTime(df_behavior_trials_CSplus,
                                                                      conditions_to_plot = ['600s',
                                                                                            '60s',
                                                                                            '3600s'],
                                                                      colors_for_conditions = dc.colors_for_conditions,
                                                                      plot_cumsum = plot_cumsum,
                                                                      plot_lick = plot_lick,
                                                                      plot_DA_cue = plot_DA_cue,
                                                                      peak_or_auc = peak_or_auc,
                                                                      linewidth_DA_cue = linewidth_DA_cue,
                                                                      linewidth_0_lick = linewidth_0_lick,
                                                                      scaled = scaled,
                                                                      scaled_align_to_1 = scaled_align_to_1,
                                                                      axsize = dc.axsize_timecourse,
                                                                      save_figs = save_figs,
                                                                      fig_path  = fig_path_Figure4_ExtDataFig6fp,
                                                                      xlim_DA = [1, None]
                                                                      )

#%%
"""
PANEL J: CUE DA ASYMPTOTE MEANS BARS 60 600 3600

"""
conditions_to_plot = ['60s',
                      '600s',
                      '3600s',]
range_to_plot = {'600s': [31, 40],
                 '60s': [301, 400],
                 '3600s': [15, 16]}
plot_lick_or_DA = 'DA'
fig_cueDA_means, ax_cueDA_means = ff.compare_asymptote_bars(df_dlight_trials_CSplus_learners_full3600,
                                                            conditions_to_plot = conditions_to_plot,
                                                            range_to_plot = range_to_plot,
                                                            condition_colors = dc.colors_for_conditions,
                                                            plot_lick_or_DA = plot_lick_or_DA,
                                                            ylim =[None,None],
                                                            ylabel ='cue DA \nlast 2, 10, or 100 trials',
                                                            axsize = dc.axsize_bars_3,
                                                            save_fig = save_figs,
                                                            fig_path = fig_path_Figure4_ExtDataFig6fp,
                                                            )
#stats
conditions_to_plot = ['60s',
                      '600s',
                      '3600s',]
range_to_plot = {'600s': [31, 40],
                 '60s': [301, 400],
                 '3600s': [15, 16]}
plot_lick_or_DA = 'DA'
condition_means_controls_dict  = dw.get_mean_values_from_trial_range(df_dlight_trials_CSplus_learners_full3600,
                                            conditions_to_subset = conditions_to_plot,
                                            range_to_subset = range_to_plot,
                                            data_to_return = plot_lick_or_DA,
                                            return_as_regular_dict = True,
                                            return_as_df = False,
                                            return_as_nested_dict = False)


pvals_all = []

t_test_60_3600 = sf.t_test_from_dict_list_or_df(condition_means_controls_dict,
                                        ['60s', '3600s'],
                                        assume_equal_var = False,
                                        data_is_regular_dict = True,
                                        )
pvals_all.append(t_test_60_3600['p'])
t_test_600_3600 = sf.t_test_from_dict_list_or_df(condition_means_controls_dict,
                                        ['600s', '3600s'],
                                        assume_equal_var = False,
                                        data_is_regular_dict = True,
                                        )
pvals_all.append(t_test_600_3600['p'])
t_test_60_600 = sf.t_test_from_dict_list_or_df(condition_means_controls_dict,
                                        ['60s', '600s'],
                                        assume_equal_var = False,
                                        data_is_regular_dict = True,
                                        )
pvals_all.append(t_test_60_600['p'])
corrected_ps = statsmodels.stats.multitest.multipletests(pvals_all,
                                                         method= 'fdr_bh',
                                                         )

#%%
"""
EXTENDED DATA FIGURE 6:
 - PANEL F: INDIVIDUAL 3600S ANIMALS CUMSUM LICK + DA
"""

conditions_to_plot = ['3600s',
                     ]
learned_trial_lick_DA = ff.getCumSumLearnedTrialsAndPlot(df_behavior_trials_CSplus_learners,
                                                      conditions_to_plot=conditions_to_plot,
                                                      colors_for_conditions = dc.colors_for_conditions,
                                                      colors_for_conditions_DA = dc.colors_for_conditions_DA,
                                                      plot_all_individuals = True,
                                                      get_DA_learned_trial = True,
                                                      use_trial_normalized_y = True,
                                                      plot_examples = False,
                                                      condition_examples = dc.cumsum_examples_DA,
                                                      save_fig = save_figs,
                                                      fig_path = fig_path_Figure4_ExtDataFig6fp,
                                                      plot_on_2_lines = False,
                                                      sharey_DA = True,
                                                      sharex = False,
                                                      axsize = dc.axsize_cumsum_all_individuals_DA,
                                                      )
#%%
"""
EXTENDED DATA FIGURE 6:
 - PANEL G: TRIALS TO LEARN 3600 VS 600 lick
 - PANEL J: TRIALS TO LEARN 3600 VS 600 DA


"""
#PANEL G
conditions_to_plot = ['600s',
                      '3600s',]
learned_trial_data = ff.getCumSumLearnedTrialsAndPlot(df_behavior_trials_CSplus_learners,
                                                      conditions_to_plot=conditions_to_plot,
                                                      get_DA_learned_trial = True,
                                                      )
figax_learned_trial = ff.plotBarsFromDict(learned_trial_data['learned_trial_lick'],
                                           condition_colors = dc.colors_for_conditions,
                                           ylabel ='trials to learn',
                                           save_stats = save_stats,
                                           axsize = dc.axsize_bars_2,
                                           save_fig = save_figs,
                                           fig_path = fig_path_Figure4_ExtDataFig6fp,
                                           )
#PANEL J
figax_learned_trial = ff.plotBarsFromDict(learned_trial_data['learned_trial_DA'],
                                           condition_colors = dc.colors_for_conditions,
                                           ylabel ='trials to learn DA',
                                           save_stats = save_stats,
                                           axsize = dc.axsize_bars_2,
                                           save_fig = save_figs,
                                           fig_path = fig_path_Figure4_ExtDataFig6fp,
                                           )
#%%

"""
EXTENDED DATA FIGURE 6:
 - PANEL H: Asymptotic lick 3600 alone

"""
save_fig = True
conditions_to_plot = ['3600s',
                     ]
range_to_plot = {'3600s': [14, 16],
                 }
plot_lick_or_DA = 'lick'
fig_cueDA_means, ax_cueDA_means = ff.compare_asymptote_bars(df_dlight_trials_CSplus_learners_full3600,
                                                            conditions_to_plot = conditions_to_plot,
                                                            range_to_plot = range_to_plot,
                                                            condition_colors = dc.colors_for_conditions,
                                                            plot_lick_or_DA = plot_lick_or_DA,
                                                            plot_individuals = True,
                                                            plot_stats = False,
                                                            save_stats = False,
                                                            ylim =[None,8],
                                                            plot_sem = True,
                                                            ylabel ='lick rate to cue (Hz)\ntrials 14 - 16',
                                                            axsize = dc.axsize_bars_1,
                                                            save_fig = save_figs,
                                                            fig_path = fig_path_Figure4_ExtDataFig6fp,
                                                            )

#%%
"""
EXTENDED DATA FIGURE 6:
 - PANEL I: DA SCATTER IRI VS TRIALS TO LEARN FOR DA WITH 3600s AND FIT LINE FOR DA
"""
conditions_to_plot = ['60s',
                      '600s',
                      '3600s',
                      ]

xlim = [10, 5000]
ylim = [0.2, 100]
learned_trial_data = ff.getCumSumLearnedTrialsAndPlot(df_dlight_trials_CSplus_learners,
                                                      conditions_to_plot = conditions_to_plot,
                                                      plot_all_individuals = False,
                                                      get_DA_learned_trial = True,
                                                      use_trial_normalized_y = True,
                                                      )
learned_trial_DA = learned_trial_data['learned_trial_DA']



IRI_vs_trials_to_learn_DA = ff.plot_IRI_vs_learned_trial_scatter(learned_trial_DA,
                                      colors_for_conditions = dc.colors_for_conditions,
                                      nested_dict = True,
                                      conditions_to_plot = conditions_to_plot,
                                      conditions_for_fitline = conditions_for_fitline,
                                      plot_fig = True,
                                      plot_line = True,
                                      linewidth_fitline = 1,
                                      linestyle_fitline = 'solid', #(0,(4,4)),
                                      alpha = 0.5,
                                      error = 'std',
                                      xlim = xlim,
                                      ylim = ylim,
                                      ax_to_plot = None,
                                      axsize = dc.axsize_timecourse,
                                      save_fig = False,
                                      fig_path = fig_path_Figure4_ExtDataFig6fp
                                      )
trials_to_learn_3600_DA_all = list(learned_trial_DA['3600s'].values())
trials_to_learn_3600_DA = np.mean(trials_to_learn_3600_DA_all)
ITI_to_predict = 3600
IRI_to_predict = ITI_to_predict + 4.25 #trial + consumption period
linear_Y_predict_3600_DA = lpf.predict_trials_to_learn_from_IRI(IRI_to_predict, IRI_vs_trials_to_learn_DA['fit_line'])

IRI_vs_trials_to_learn_DA['ax'].axhline(linear_Y_predict_3600_DA,
                            linestyle = (2,(4,4)),
                            color = 'k',
                            linewidth = 0.5,
                            alpha = 1)

IRI_vs_trials_to_learn_DA['ax'].axhline(trials_to_learn_3600_DA,
                            linestyle = (2,(4,4)),
                            color = '#4362AD',
                            linewidth = 0.5,
                            alpha = 1)
if save_figs:
    IRI_vs_trials_to_learn_DA['fig'].savefig(os.path.join(fig_path_Figure4_ExtDataFig6fp,
                                                         'IRIvsDALearnedTrialScatter_fitlinewith3600spredicted.pdf'),
                                            bbox_inches = 'tight',
                                            transparent = True,
                                            )

one_sample_ttest3600_from_predicted_DA = pingouin.ttest(trials_to_learn_3600_DA_all, linear_Y_predict_3600_DA)

#%%
"""
EXTENDED DATA FIGURE 6:
 - PANEL K: CUE DA AND LICK 3600s ITI full 16 trials
 - PANEL M: CUE DA AND REWARD DA 3600s ITI full 16 trials
 - PANEL P: PEAK CUE DA AND REWARD DA 3600s ITI full 16 trials
"""
linewidth_lick = 1
linewidth_0_lick = 1
linewidth_DA_cue = 1
lick_left = True
xlim_lick = [1,None]

#PANEL K
plot_lick = True
plot_DA_cue = True
plot_DA_reward = False
plot_cumsum = False
peak_or_auc = 'auc'
fig_cueDA_lick_3600, ax_cueDA_lick_3600 = ff.plotDALickOverTime(df_dlight_trials_CSplus_learners_full3600,
                                                                conditions_to_plot = ['3600s'],
                                                                colors_for_conditions = dc.colors_for_conditions,
                                                                colors_for_conditions_DA = dc.colors_for_conditions,
                                                                plot_cumsum = plot_cumsum,
                                                                plot_lick = plot_lick,
                                                                plot_DA_cue = plot_DA_cue,
                                                                plot_DA_reward = plot_DA_reward,
                                                                peak_or_auc = peak_or_auc,
                                                                linewidth_lick = linewidth_lick,
                                                                linewidth_0_lick = linewidth_0_lick,
                                                                linewidth_DA_cue = linewidth_DA_cue,
                                                                axsize = dc.axsize_timecourse,
                                                                xlim_lick = xlim_lick,
                                                                lick_left = lick_left,
                                                                save_figs = save_figs,
                                                                fig_path  = fig_path_Figure4_ExtDataFig6fp,
                                                             )
#PANEL M
plot_lick = False
plot_DA_cue = True
plot_DA_reward = True
plot_cumsum = False
peak_or_auc = 'auc'
fig_cueDA_lick_3600, ax_cueDA_lick_3600 = ff.plotDALickOverTime(df_dlight_trials_CSplus_learners_full3600,
                                                                conditions_to_plot = ['3600s'],
                                                                colors_for_conditions = dc.colors_for_conditions,
                                                                colors_for_conditions_DA = dc.colors_for_conditions,
                                                                plot_cumsum = plot_cumsum,
                                                                plot_lick = plot_lick,
                                                                plot_DA_cue = plot_DA_cue,
                                                                plot_DA_reward = plot_DA_reward,
                                                                peak_or_auc = peak_or_auc,
                                                                linewidth_lick = linewidth_lick,
                                                                linewidth_DA_cue = linewidth_DA_cue,
                                                                linewidth_0_lick =linewidth_0_lick,
                                                                axsize = dc.axsize_timecourse,
                                                                xlim_DA = xlim_lick,
                                                                lick_left = lick_left,
                                                                save_figs = save_figs,
                                                                fig_path  = fig_path_Figure4_ExtDataFig6fp,
                                                              )
#PANEL P
plot_lick = False
plot_DA_cue = True
plot_DA_reward = True
plot_cumsum = False
peak_or_auc = 'peak'
fig_cueDA_lick_3600, ax_cueDA_lick_3600 = ff.plotDALickOverTime(df_dlight_trials_CSplus_learners_full3600,
                                                                conditions_to_plot = ['3600s'],
                                                                colors_for_conditions = dc.colors_for_conditions,
                                                                colors_for_conditions_DA = dc.colors_for_conditions,
                                                                plot_cumsum = plot_cumsum,
                                                                plot_lick = plot_lick,
                                                                plot_DA_cue = plot_DA_cue,
                                                                plot_DA_reward = plot_DA_reward,
                                                                peak_or_auc = peak_or_auc,
                                                                linewidth_lick = linewidth_lick,
                                                                linewidth_DA_cue = linewidth_DA_cue,
                                                                linewidth_0_lick = linewidth_0_lick,
                                                                axsize = dc.axsize_timecourse,
                                                                xlim_DA = xlim_lick,
                                                                lick_left = lick_left,
                                                                save_figs = save_figs,
                                                                fig_path  = fig_path_Figure4_ExtDataFig6fp,
                                                                )
#%%
"""
EXTENDED DATA FIGURE 6:
 - PANEL L: CUE DA AVERAGE TIMECOURSE 60 600 3600
 - PANEL N: PEAK CUE DA CUMSUM TIMECOURSE 60 600 3600
 - PANEL O: PEAK CUE DA AVERAGE TIMECOURSE 60 600 3600
"""
scaled = True
scaled_align_to_1 = True
plot_lick = False
plot_DA_cue = True
linewidth_0_lick = 1
linewidth_DA_cue = 0.5
ylim_DA = [-0.1, None]

peak_or_auc = 'auc'
plot_cumsum = False
fig_cueDA_cumsum_scale, ax_cueDA_cumsum_scale = ff.plotDALickOverTime(df_dlight_trials_CSplus_learners,
                                                                      conditions_to_plot = ['600s',
                                                                                            '60s',
                                                                                            '3600s'],
                                                                      colors_for_conditions = dc.colors_for_conditions,
                                                                      plot_cumsum = plot_cumsum,
                                                                      plot_lick = plot_lick,
                                                                      plot_DA_cue = plot_DA_cue,
                                                                      peak_or_auc = peak_or_auc,
                                                                      linewidth_DA_cue = linewidth_DA_cue,
                                                                      linewidth_0_lick = linewidth_0_lick,
                                                                      scaled = scaled,
                                                                      scaled_align_to_1 = scaled_align_to_1,
                                                                      axsize = dc.axsize_timecourse,
                                                                      ylim_DA = ylim_DA,
                                                                      save_figs = save_figs,
                                                                      fig_path  = fig_path_Figure4_ExtDataFig6fp,
                                                                      xlim_DA = [0.9, None]
                                                                      )


plot_cumsum = False
peak_or_auc = 'peak'
fig_cueDA_cumsum_scale, ax_cueDA_cumsum_scale = ff.plotDALickOverTime(df_dlight_trials_CSplus_learners,
                                                                      conditions_to_plot = ['600s',
                                                                                            '60s',
                                                                                            '3600s'],
                                                                      colors_for_conditions = dc.colors_for_conditions,
                                                                      plot_cumsum = plot_cumsum,
                                                                      plot_lick = plot_lick,
                                                                      plot_DA_cue = plot_DA_cue,
                                                                      peak_or_auc = peak_or_auc,
                                                                      linewidth_DA_cue = linewidth_DA_cue,
                                                                      linewidth_0_lick = linewidth_0_lick,
                                                                      scaled = scaled,
                                                                      scaled_align_to_1 = scaled_align_to_1,
                                                                      axsize = dc.axsize_timecourse,
                                                                      ylim_DA = ylim_DA,
                                                                      save_figs = save_figs,
                                                                      fig_path  = fig_path_Figure4_ExtDataFig6fp,
                                                                      xlim_DA = [0.9, None]
                                                                      )


plot_cumsum = True
peak_or_auc = 'peak'
fig_cueDA_cumsum_scale, ax_cueDA_cumsum_scale = ff.plotDALickOverTime(df_dlight_trials_CSplus_learners,
                                                                      conditions_to_plot = ['600s',
                                                                                            '60s',
                                                                                            '3600s'],
                                                                      colors_for_conditions = dc.colors_for_conditions,
                                                                      plot_cumsum = plot_cumsum,
                                                                      plot_lick = plot_lick,
                                                                      plot_DA_cue = plot_DA_cue,
                                                                      peak_or_auc = peak_or_auc,
                                                                      linewidth_DA_cue = linewidth_DA_cue,
                                                                      linewidth_0_lick = linewidth_0_lick,
                                                                      scaled = scaled,
                                                                      scaled_align_to_1 = scaled_align_to_1,
                                                                      axsize = dc.axsize_timecourse,
                                                                      ylim_DA = ylim_DA,
                                                                      save_figs = save_figs,
                                                                      fig_path  = fig_path_Figure4_ExtDataFig6fp,
                                                                      xlim_DA = [0.9, None]
                                                                      )