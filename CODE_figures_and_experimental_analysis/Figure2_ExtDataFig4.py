# -*- coding: utf-8 -*-
"""
@author: Dennis A. Burke (dennis.burke@ucsf.edu ; permanent address: dennis.a.burke AT gmail)

from Burke et al. - Duration between rewards controls the rate of behavioral and dopaminergic learning

FIGURE 2: Dopaminergic learning in one-tenth the experiences with ten times the trial spacing.

"""
"""
imports
"""
import os

import functions.load_preprocess  as lp
import functions.default_configs as dc
import functions.figure_functions as ff
import functions.lick_photo_functions as lpf

#%%
"""
set paths to data and outputs
"""
#set working directory to script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

nwb_dir_path = r'..\DATA_experimental\001632'
figure_path_root = r'..\FIGURES'
fig_path_Figure2_ExtDataFig4 = os.path.join(figure_path_root, r'Figure2_ExtDataFig4')

save_figs = False
save_stats = False

#%%
"""
load and prepare data

"""
nwb_file_info_df =  lp.get_all_nwb_files_by_condition(nwb_dir_path,  ['600s', '60s'], dopamine_only = True)
#%%
all_trial_data_df, df = lp.make_trial_df_from_nwb(nwb_file_info_df,
                                                               total_time_window_s = 37,
                                                               baseline_length_s = 7,
                                                               )
df_dlight_trials_CSplus_learners = lp.subset_dopamine_animals(lp.get_behavior_trials_CSplus_learners_df(df))


#%%
"""
FIGURE 2:
 - PANEL C: EXAMPLE ANIMAL RASTER, PSTH, AND HEATMAP 60S AND 600S
"""

plot_lick = True
plot_DA = True

fig_ex_dlight_600s, ax_ex_dlight_600s = ff.plotPSTHbyDay(all_trial_data_df,
                                                            dc.cumsum_examples_DA['600s'],
                                                            color_lick_PSTH = dc.colors_for_conditions_DA['600s'],
                                                            color_DA_PSTH = dc.colors_for_conditions_DA['600s'],
                                                            plot_lick_raster = plot_lick,
                                                            plot_lick_PSTH = plot_lick,
                                                            plot_DA_heatmap = plot_DA,
                                                            plot_DA_PSTH = plot_DA,
                                                            axsize = dc.axsize_raster_PSTH_panel,
                                                            fig_path =fig_path_Figure2_ExtDataFig4,
                                                            save_fig = save_figs,
                                                            )
fig_ex_dlight_60s, ax_ex_dlight_60s = ff.plotPSTHbyDay(all_trial_data_df,
                                                            dc.cumsum_examples_DA['60s'],
                                                            color_lick_PSTH = dc.colors_for_conditions_DA['60s'],
                                                            color_DA_PSTH = dc.colors_for_conditions_DA['60s'],
                                                            plot_lick_raster = plot_lick,
                                                            plot_lick_PSTH = plot_lick,
                                                            plot_DA_heatmap = plot_DA,
                                                            plot_DA_PSTH = plot_DA,
                                                            axsize = dc.axsize_raster_PSTH_panel,
                                                            fig_path = fig_path_Figure2_ExtDataFig4,
                                                            save_fig = save_figs,
                                                            )
#%%
"""
FIGURE 2:
 - PANEL D: EXAMPLE ANIMAL LICK AND DA CUMSUM
"""
use_trial_normalized_y = True
ylim_left = [None, None]
ylim_right = [None, None]
lick_left = True
fig_ex_dlight_60s_cumsum, ax_ex_dlight_60s_cumsum = ff.plotExampleCumsumLearnedTrial(df_dlight_trials_CSplus_learners,
                                                                                           animal = dc.cumsum_examples_DA['600s'],
                                                                                           color_lick = dc.colors_for_conditions_DA['600s'],
                                                                                           color_da = dc.colors_for_conditions_DA['600s'],
                                                                                           plot_lick = 1,
                                                                                           plot_da = True,
                                                                                           use_trial_normalized_y = use_trial_normalized_y,
                                                                                           axsize = dc.axsize_cumsum_examples,
                                                                                           ylim_left = ylim_left,
                                                                                           ylim_right = [None, None],
                                                                                           lick_left = lick_left,
                                                                                           save_fig = save_figs,
                                                                                           fig_path = fig_path_Figure2_ExtDataFig4,
                                                                                            )

fig_ex_dlight_600s_cumsum, ax_ex_dlight_600s_cumsum = ff.plotExampleCumsumLearnedTrial(df_dlight_trials_CSplus_learners,
                                                                                           animal = dc.cumsum_examples_DA['60s'],
                                                                                           color_lick = dc.colors_for_conditions_DA['60s'],
                                                                                           color_da = dc.colors_for_conditions_DA['60s'],
                                                                                           plot_lick = 1,
                                                                                           plot_da = True,
                                                                                           use_trial_normalized_y = use_trial_normalized_y,
                                                                                           axsize = dc.axsize_cumsum_examples,
                                                                                           ylim_left = ylim_left,
                                                                                           ylim_right = [None, None],
                                                                                           lick_left = lick_left,
                                                                                           save_fig = save_figs,
                                                                                           fig_path = fig_path_Figure2_ExtDataFig4,
                                                                                            )
#%%
"""
FIGURE 2:
 - PANEL E: TRIALS TO LEARN - DA, 60s 600s
 - PANEL F: TRIALS FROM DA TO BEHAVIOR LAG, 60s 600s
 - PANEL G: TIME TO LEARN DA

"""

conditions_to_plot = ['60s',
                      '600s',]
ylim = [None, 100]
learned_trial_data = ff.getCumSumLearnedTrialsAndPlot(df_dlight_trials_CSplus_learners,
                                                      conditions_to_plot=conditions_to_plot,
                                                      get_DA_learned_trial = True,
                                                      )
learned_trial_DA = learned_trial_data['learned_trial_DA']
figax_learned_trial = ff.plotBarsFromDict(learned_trial_DA,
                                           condition_colors = dc.colors_for_conditions,
                                           ylabel ='trials to DA learning',
                                           data_is_nested_dict = True,
                                           plot_individuals = True,
                                           logscale = True,
                                           ylim = ylim,
                                           save_stats = save_stats,
                                           axsize = dc.axsize_bars_2,
                                           save_fig = save_figs,
                                           fig_path = fig_path_Figure2_ExtDataFig4,
                                           )
learned_trial_lag_to_learn = learned_trial_data['lag_to_learn']
figax_learned_trial = ff.plotBarsFromDict(learned_trial_lag_to_learn,
                                           condition_colors = dc.colors_for_conditions,
                                           ylabel ='trials from DA to beh.',
                                           data_is_nested_dict = True,
                                           plot_individuals = True,
                                           logscale = True,
                                           ylim = ylim,
                                           save_stats = save_stats,
                                           axsize = dc.axsize_bars_2,
                                           save_fig = save_figs,
                                           fig_path = fig_path_Figure2_ExtDataFig4,
                                           )


time_to_learn_dict = lpf.calculate_time_to_learn_from_learned_trials(learned_trial_DA, df_dlight_trials_CSplus_learners)
figax_learned_trial = ff.plotBarsFromDict(time_to_learn_dict,
                                           condition_colors = dc.colors_for_conditions,
                                           ylabel ='total conditioning time before cue DA (s) ',
                                           data_is_nested_dict = True,
                                           plot_individuals = True,
                                           plot_stats = True,
                                           save_stats = True,
                                           plot_sem = True,
                                           logscale = False,
                                           axsize = dc.axsize_bars_2,
                                           ylim = [None, None],
                                           save_fig = True,
                                           fig_path = fig_path_Figure2_ExtDataFig4,
                                           )


#%%
"""
FIGURE 2:
 - PANEL H: TIMECOURSES OF CUMSUM LICK DA ALIGNED TO BHEAVIORAL LEARNED TRIAL

"""
learned_trial_DA_lick = learned_trial_data['learned_trial_lick']

plot_cumsum = True
plot_lick = True
plot_DA_cue = True
plot_DA_reward = False
align_to_DA = False
align_to_lick = True
peak_or_auc = 'auc'
norm_to_max_individual_rewards = 0
norm_to_one_individual = True
ylim_left = [None, 0.91]
conditions = ['60s']
aligned_lickDAtimecourses_60s = ff.plotDALickOverTimeAligned(df_dlight_trials_CSplus_learners,
                                                                conditions_to_plot = conditions,
                                                                learned_trials = learned_trial_DA_lick,
                                                                learned_trials_DA = learned_trial_DA,
                                                                colors_for_conditions = dc.colors_for_conditions,
                                                                plot_cumsum = plot_cumsum,
                                                                plot_lick = plot_lick,
                                                                plot_DA_cue = plot_DA_cue,
                                                                plot_DA_reward = plot_DA_reward,
                                                                peak_or_auc = peak_or_auc,
                                                                norm_to_max_individual_rewards = norm_to_max_individual_rewards,
                                                                axsize = dc.axsize_timecourse,
                                                                colors_for_conditions_DA = dc.colors_for_conditions_DA,
                                                                align_to_lick = align_to_lick,
                                                                align_to_DA = align_to_DA,
                                                                norm_to_one_individual = norm_to_one_individual,
                                                                ylim_left = ylim_left,
                                                                save_figs = save_figs,
                                                                fig_path = fig_path_Figure2_ExtDataFig4,
                                                                )
conditions = ['600s']
aligned_lickDAtimecourses_600s = ff.plotDALickOverTimeAligned(df_dlight_trials_CSplus_learners,
                                                                conditions_to_plot = conditions,
                                                                learned_trials = learned_trial_DA_lick,
                                                                learned_trials_DA = learned_trial_DA,
                                                                colors_for_conditions = dc.colors_for_conditions,
                                                                plot_cumsum = plot_cumsum,
                                                                plot_lick = plot_lick,
                                                                plot_DA_cue = plot_DA_cue,
                                                                plot_DA_reward = plot_DA_reward,
                                                                peak_or_auc = peak_or_auc,
                                                                norm_to_max_individual_rewards = norm_to_max_individual_rewards,
                                                                axsize = dc.axsize_timecourse,
                                                                colors_for_conditions_DA = dc.colors_for_conditions_DA,
                                                                align_to_lick = align_to_lick,
                                                                align_to_DA = align_to_DA,
                                                                norm_to_one_individual = norm_to_one_individual,
                                                                ylim_left = aligned_lickDAtimecourses_60s.get_ylim(),
                                                                save_figs = save_figs,
                                                                fig_path = fig_path_Figure2_ExtDataFig4,
                                                                )
#%%
"""
FIGURE 2:
 - PANEL I: CUMSUM CUE DA 60 AND 600s
"""
plot_lick = False
plot_DA_cue = True
plot_cumsum = True
scaled = True
linewidth_DA_cue = 1
scaled_align_to_1 = True
fig_cueDA_cumsum_scale, ax_cueDA_cumsum_scale = ff.plotDALickOverTime(df_dlight_trials_CSplus_learners,
                                                                      conditions_to_plot = ['600s',
                                                                                            '60s'],
                                                                      colors_for_conditions = dc.colors_for_conditions,
                                                                      plot_cumsum = plot_cumsum,
                                                                      plot_lick = plot_lick,
                                                                      plot_DA_cue = plot_DA_cue,
                                                                      linewidth_DA_cue = linewidth_DA_cue,
                                                                      linewidth_0_lick = 1,
                                                                      scaled = scaled,
                                                                      scaled_align_to_1 = scaled_align_to_1,
                                                                      axsize = dc.axsize_timecourse,
                                                                      save_figs = save_figs,
                                                                      fig_path  = fig_path_Figure2_ExtDataFig4,
                                                                      )

#%%
"""
EXTENDED DATA FIG 4
 - PANEL A: MORE EXAMPLE RASTER HEATMAP / PSTH
 - PANEL B: INDIVIDUAL CUMSUM LICK AND DA REMAINING ANIMALS
"""
#PANEL A
plot_lick = True
plot_DA = True
fig_ex_dlight_600s_supp, ax_ex_dlight_600s_supp = ff.plotPSTHbyDay(all_trial_data_df,
                                                            dc.cumsum_examples_DA_supp['600s'],
                                                            color_lick_PSTH = dc.colors_for_conditions_DA['600s'],
                                                            color_DA_PSTH = dc.colors_for_conditions_DA['600s'],
                                                            plot_lick_raster = plot_lick,
                                                            plot_lick_PSTH = plot_lick,
                                                            plot_DA_heatmap = plot_DA,
                                                            plot_DA_PSTH = plot_DA,
                                                            fig_path = fig_path_Figure2_ExtDataFig4,
                                                            save_fig = save_figs,
                                                            )
fig_ex_dlight_60s_supp, ax_ex_dlight_60s_supp = ff.plotPSTHbyDay(all_trial_data_df,
                                                            dc.cumsum_examples_DA_supp['60s'],
                                                            color_lick_PSTH = dc.colors_for_conditions_DA['60s'],
                                                            color_DA_PSTH = dc.colors_for_conditions_DA['60s'],
                                                            plot_lick_raster = plot_lick,
                                                            plot_lick_PSTH = plot_lick,
                                                            plot_DA_heatmap = plot_DA,
                                                            plot_DA_PSTH = plot_DA,
                                                            fig_path = fig_path_Figure2_ExtDataFig4,
                                                            save_fig = save_figs,
                                                            )
#PANEL B
conditions_to_plot = ['60s',
                      '600s',
                      ]
learned_trial_lick_DA = ff.getCumSumLearnedTrialsAndPlot(df_dlight_trials_CSplus_learners,
                                                      conditions_to_plot=conditions_to_plot,
                                                      colors_for_conditions = dc.colors_for_conditions,
                                                      colors_for_conditions_DA = dc.colors_for_conditions_DA,
                                                      plot_all_individuals = True,
                                                      get_DA_learned_trial = True,
                                                      use_trial_normalized_y = True,
                                                      plot_examples = False,
                                                      condition_examples = dc.cumsum_examples_DA,
                                                      save_fig = save_figs,
                                                      fig_path = fig_path_Figure2_ExtDataFig4,
                                                      plot_on_2_lines = False,
                                                      sharey_DA = False,
                                                      sharex = False,
                                                      axsize = dc.axsize_cumsum_all_individuals_DA_nosharey)

#%%
"""
EXTENDED DATA FIG 4: SCALED TIMECOURSES
 - PANEL D: CUE DA AUC AVG
 - PANEL E: CUE DA PEAK AVG
 - PANEL F: CUE DA PEAK CUMSUM
"""
#PANEL D
conditions_to_plot = ['60s',
                      '600s',
                      ]

scaled = True
scaled_align_to_1 = True
plot_lick = False
plot_DA_cue = True
norm_to_max_individual_rewards = 3
linewidth_0_lick = 1
xlim_DA = [0.9, None]
plot_cumsum = False
peak_or_auc = 'auc'
linewidth_DA_cue = 0.35
ylim_DA = [-0.1, 0.715]
fig_cueDA_cumsum_scale, ax_cueDA_cumsum_scale = ff.plotDALickOverTime(df_dlight_trials_CSplus_learners,
                                                                      conditions_to_plot = conditions_to_plot,
                                                                      colors_for_conditions = dc.colors_for_conditions,
                                                                      plot_cumsum = plot_cumsum,
                                                                      plot_lick = plot_lick,
                                                                      plot_DA_cue = plot_DA_cue,
                                                                      peak_or_auc = peak_or_auc,
                                                                      linewidth_DA_cue = linewidth_DA_cue,
                                                                      linewidth_0_lick = linewidth_0_lick,
                                                                      scaled = scaled,
                                                                      scaled_align_to_1 = scaled_align_to_1,
                                                                      xlim_DA = xlim_DA,
                                                                      ylim_DA = ylim_DA,
                                                                      axsize = dc.axsize_timecourse,
                                                                      save_figs = save_figs,
                                                                      fig_path  = fig_path_Figure2_ExtDataFig4,
                                                                      )
#PANEL E
plot_cumsum = False
linewidth_DA_cue = 0.35
peak_or_auc = 'peak'
fig_cueDA_cumsum_scale, ax_cueDA_cumsum_scale = ff.plotDALickOverTime(df_dlight_trials_CSplus_learners,
                                                                      conditions_to_plot = conditions_to_plot,
                                                                      colors_for_conditions = dc.colors_for_conditions,
                                                                      plot_cumsum = plot_cumsum,
                                                                      plot_lick = plot_lick,
                                                                      plot_DA_cue = plot_DA_cue,
                                                                      peak_or_auc = peak_or_auc,
                                                                      linewidth_DA_cue = linewidth_DA_cue,
                                                                      linewidth_0_lick = linewidth_0_lick,
                                                                      scaled = scaled,
                                                                      scaled_align_to_1 = scaled_align_to_1,
                                                                      xlim_DA = xlim_DA,
                                                                      ylim_DA =ylim_DA,
                                                                      axsize = dc.axsize_timecourse,
                                                                      save_figs = save_figs,
                                                                      fig_path  = fig_path_Figure2_ExtDataFig4,
                                                                      )
#PANEL F
plot_cumsum = True
linewidth_DA_cue = 1
peak_or_auc = 'peak'
fig_cueDA_cumsum_scale, ax_cueDA_cumsum_scale = ff.plotDALickOverTime(df_dlight_trials_CSplus_learners,
                                                                      conditions_to_plot = conditions_to_plot,
                                                                      colors_for_conditions = dc.colors_for_conditions,
                                                                      plot_cumsum = plot_cumsum,
                                                                      plot_lick = plot_lick,
                                                                      plot_DA_cue = plot_DA_cue,
                                                                      peak_or_auc = peak_or_auc,
                                                                      linewidth_DA_cue = linewidth_DA_cue,
                                                                      linewidth_0_lick = linewidth_0_lick,
                                                                      scaled = scaled,
                                                                      scaled_align_to_1 = scaled_align_to_1,
                                                                      xlim_DA = xlim_DA,
                                                                      axsize = dc.axsize_timecourse,
                                                                      save_figs = save_figs,
                                                                      fig_path  = fig_path_Figure2_ExtDataFig4,
                                                                      )

#%%
"""
EXTENDED DATA FIG 4: BEHAVIORAL LEARNING ALIGNED TIMECOURSES
 - PANEL C: CUE DA + LICK PEAK CUMSUM (NORM TO MAX)
 - PANEL G: CUE DA + REWARD DA AUC AVG
 - PANEL H: CUE DA + REWARD DA PEAK AVG

"""
#PANEL C
conditions_to_plot = ['60s',
                      '600s',]

learned_trial_data = ff.getCumSumLearnedTrialsAndPlot(df_dlight_trials_CSplus_learners,
                                                      conditions_to_plot=conditions_to_plot,
                                                      get_DA_learned_trial = True,
                                                      )
learned_trial_DA_lick = learned_trial_data['learned_trial_lick']

plot_cumsum = True
plot_lick = True
plot_DA_cue = True
plot_DA_reward = False
align_to_DA = False
align_to_lick = True
peak_or_auc = 'peak'
linewidth_lick = 1
linewidth_DA_cue = 1
norm_to_max_individual_rewards = 0
norm_to_one_individual = True
ylim_left = [None, 0.91]
conditions = ['60s']
aligned_lickDAtimecourses_60s = ff.plotDALickOverTimeAligned(df_dlight_trials_CSplus_learners,
                                                            conditions_to_plot = conditions,
                                                            learned_trials = learned_trial_DA_lick,
                                                            learned_trials_DA = learned_trial_DA,
                                                            colors_for_conditions = dc.colors_for_conditions,
                                                            plot_cumsum = plot_cumsum,
                                                            plot_lick = plot_lick,
                                                            plot_DA_cue = plot_DA_cue,
                                                            plot_DA_reward = plot_DA_reward,
                                                            peak_or_auc = peak_or_auc,
                                                            linewidth_lick =linewidth_lick,
                                                            linewidth_DA_cue = linewidth_DA_cue,
                                                            norm_to_max_individual_rewards = norm_to_max_individual_rewards,
                                                            axsize = dc.axsize_timecourse,
                                                            colors_for_conditions_DA = dc.colors_for_conditions_DA,
                                                            align_to_lick = align_to_lick,
                                                            norm_to_one_individual = norm_to_one_individual,
                                                            ylim_left = ylim_left,
                                                            save_figs = save_figs,
                                                            fig_path = fig_path_Figure2_ExtDataFig4,
                                                            )
conditions = ['600s']
aligned_lickDAtimecourses_600s = ff.plotDALickOverTimeAligned(df_dlight_trials_CSplus_learners,
                                                                conditions_to_plot = conditions,
                                                                learned_trials = learned_trial_DA_lick,
                                                                learned_trials_DA = learned_trial_DA,
                                                                colors_for_conditions = dc.colors_for_conditions,
                                                                plot_cumsum = plot_cumsum,
                                                                plot_lick = plot_lick,
                                                                plot_DA_cue = plot_DA_cue,
                                                                plot_DA_reward = plot_DA_reward,
                                                                peak_or_auc = peak_or_auc,
                                                                linewidth_lick =linewidth_lick,
                                                                linewidth_DA_cue = linewidth_DA_cue,
                                                                norm_to_max_individual_rewards = norm_to_max_individual_rewards,
                                                                axsize = dc.axsize_timecourse,
                                                                colors_for_conditions_DA = dc.colors_for_conditions_DA,
                                                                align_to_lick = align_to_lick,
                                                                norm_to_one_individual = norm_to_one_individual,
                                                                ylim_left = aligned_lickDAtimecourses_60s.get_ylim(),
                                                                save_figs = save_figs,
                                                                fig_path = fig_path_Figure2_ExtDataFig4,
                                                                )
#PANEL G
plot_cumsum = False
plot_lick = False
plot_DA_cue = True
plot_DA_reward = True



align_to_lick = True
peak_or_auc = 'auc'
linewidth_DA_cue = 0.35
linewidth_DA_reward = 0.35

norm_to_max_individual_rewards = 3
norm_to_one_individual = False
ylim_left = [-0.2, 1.1]
conditions = ['60s']
aligned_lickDAtimecourses_60s = ff.plotDALickOverTimeAligned(df_dlight_trials_CSplus_learners,
                                                                conditions_to_plot = conditions,
                                                                learned_trials = learned_trial_DA_lick,
                                                                learned_trials_DA = learned_trial_DA,
                                                                colors_for_conditions = dc.colors_for_conditions,
                                                                plot_cumsum = plot_cumsum,
                                                                plot_lick = plot_lick,
                                                                plot_DA_cue = plot_DA_cue,
                                                                plot_DA_reward = plot_DA_reward,
                                                                peak_or_auc = peak_or_auc,
                                                                linewidth_DA_cue = linewidth_DA_cue,
                                                                linewidth_DA_reward = linewidth_DA_reward,
                                                                norm_to_max_individual_rewards = norm_to_max_individual_rewards,
                                                                axsize = dc.axsize_timecourse,
                                                                colors_for_conditions_DA = dc.colors_for_conditions_DA,
                                                                align_to_lick = align_to_lick,
                                                                norm_to_one_individual = norm_to_one_individual,
                                                                ylim_left = ylim_left,
                                                                save_figs = save_figs,
                                                                fig_path = fig_path_Figure2_ExtDataFig4,
                                                                )
conditions = ['600s']
aligned_lickDAtimecourses_600s = ff.plotDALickOverTimeAligned(df_dlight_trials_CSplus_learners,
                                                                conditions_to_plot = conditions,
                                                                learned_trials = learned_trial_DA_lick,
                                                                learned_trials_DA = learned_trial_DA,
                                                                colors_for_conditions = dc.colors_for_conditions,
                                                                plot_cumsum = plot_cumsum,
                                                                plot_lick = plot_lick,
                                                                plot_DA_cue = plot_DA_cue,
                                                                plot_DA_reward = plot_DA_reward,
                                                                peak_or_auc = peak_or_auc,
                                                                linewidth_DA_cue = linewidth_DA_cue,
                                                                linewidth_DA_reward = linewidth_DA_reward,
                                                                norm_to_max_individual_rewards = norm_to_max_individual_rewards,
                                                                axsize = dc.axsize_timecourse,
                                                                colors_for_conditions_DA = dc.colors_for_conditions_DA,
                                                                align_to_lick = align_to_lick,
                                                                norm_to_one_individual = norm_to_one_individual,
                                                                ylim_left = aligned_lickDAtimecourses_60s.get_ylim(),
                                                                save_figs = save_figs,
                                                                fig_path = fig_path_Figure2_ExtDataFig4,
                                                                )
#PANEL H
peak_or_auc = 'peak'
conditions = ['60s']
aligned_lickDAtimecourses_60s = ff.plotDALickOverTimeAligned(df_dlight_trials_CSplus_learners,
                                                                conditions_to_plot = conditions,
                                                                learned_trials = learned_trial_DA_lick,
                                                                learned_trials_DA = learned_trial_DA,
                                                                colors_for_conditions = dc.colors_for_conditions,
                                                                plot_cumsum = plot_cumsum,
                                                                plot_lick = plot_lick,
                                                                plot_DA_cue = plot_DA_cue,
                                                                plot_DA_reward = plot_DA_reward,
                                                                peak_or_auc = peak_or_auc,
                                                                linewidth_DA_cue = linewidth_DA_cue,
                                                                linewidth_DA_reward = linewidth_DA_reward,
                                                                norm_to_max_individual_rewards = norm_to_max_individual_rewards,
                                                                axsize = dc.axsize_timecourse,
                                                                colors_for_conditions_DA = dc.colors_for_conditions_DA,
                                                                align_to_lick = align_to_lick,
                                                                norm_to_one_individual = norm_to_one_individual,
                                                                ylim_left = ylim_left,
                                                                save_figs = save_figs,
                                                                fig_path = fig_path_Figure2_ExtDataFig4,
                                                                )
conditions = ['600s']
aligned_lickDAtimecourses_600s = ff.plotDALickOverTimeAligned(df_dlight_trials_CSplus_learners,
                                                                conditions_to_plot = conditions,
                                                                learned_trials = learned_trial_DA_lick,
                                                                learned_trials_DA = learned_trial_DA,
                                                                colors_for_conditions = dc.colors_for_conditions,
                                                                plot_cumsum = plot_cumsum,
                                                                plot_lick = plot_lick,
                                                                plot_DA_cue = plot_DA_cue,
                                                                plot_DA_reward = plot_DA_reward,
                                                                peak_or_auc = peak_or_auc,
                                                                linewidth_DA_cue = linewidth_DA_cue,
                                                                linewidth_DA_reward = linewidth_DA_reward,
                                                                norm_to_max_individual_rewards = norm_to_max_individual_rewards,
                                                                axsize = dc.axsize_timecourse,
                                                                colors_for_conditions_DA = dc.colors_for_conditions_DA,
                                                                align_to_lick = align_to_lick,
                                                                norm_to_one_individual = norm_to_one_individual,
                                                                ylim_left = aligned_lickDAtimecourses_60s.get_ylim(),
                                                                save_figs = save_figs,
                                                                fig_path = fig_path_Figure2_ExtDataFig4,
                                                                )