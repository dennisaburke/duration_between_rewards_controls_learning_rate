# -*- coding: utf-8 -*-
"""
@author: Dennis A. Burke (dennis.burke@ucsf.edu ; permanent address: dennis.a.burke AT gmail)

from Burke et al. - Duration between rewards controls the rate of behavioral and dopaminergic learning

HELPER FUNCTIONS
"""

import os
from pathlib import Path
import itertools
import pingouin
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import math
from scipy.ndimage import gaussian_filter1d
import scipy.stats as stats
import warnings

import stats_functions as sf
import lick_photo_functions as lpf
import data_wrangling as dw


def plotPSTHbyDay(trials_df,
                  animal,
                  days_to_plot = (1,8),
                  plot_lick_raster = True,
                  plot_lick_PSTH = True,
                  plot_DA_heatmap = True,
                  plot_DA_PSTH = True,
                  align_to_cue = True,
                  plot_in_sec = True,
                  xlim_in_sec = [-2.5, 7.5],
                  ylim_lick = [-1, 12],
                  ylim_DA = [None, None],
                  dff_to_plot ='%',
                  axsize = (0.5, 0.35),
                  sharey = 'row',
                  stroke_raster = 0.35,
                  stroke_PSTH = 0.5,
                  color_DA_PSTH = 'dodgerblue',
                  color_lick_PSTH = 'green',
                  alpha_PSTH_error = 0.3,
                  color_cue_shade = '#939598',
                  alpha_cue_shade = 0.5,
                  fontsize_title = 7,
                  fontsize_label = 7,
                  fontsize_ticks = 6,
                  smooth_lick = 0.75,
                  stroke_cue_reward_vertical = 0.25,
                  linestyle_lick = 'solid',
                  linestyle_DA = 'dashed',
                  fig_path = '',
                  save_fig = False,
                  save_png = False,
                  linewidth_0_lick = 0,
                  linewidth_0_DA = 0,
                  alpha0line = 0.5,
                  norm_to_max_rewards = 0,
                  colorplotmax = None,
                  ):

    #check if behavior data in s or ms

    data_in_s = True if np.max(trials_df['cue_dur'])<49 else False

    same_unit = plot_in_sec == data_in_s
    #scale between mc and s depending on data format and desired output
    same_unit = plot_in_sec == data_in_s
    scale_factor = 1 if same_unit else (1/1000 if plot_in_sec else 1000)
    binsize_for_lick_PSTH = 0.1 if data_in_s else 100


    if isinstance(days_to_plot, int):
        days = [days_to_plot]
    else:

        days = np.arange(days_to_plot[0], days_to_plot[1]+1)

    total_time_to_plot = (xlim_in_sec[1] - xlim_in_sec[0]) *scale_factor
    baseline_time_to_plot = xlim_in_sec[0] * (-scale_factor)
    #determines parameters and axes for figure
    total_days = len(days)
    total_rows =  plot_lick_raster + plot_lick_PSTH + plot_DA_heatmap + plot_DA_PSTH
    fig_width = total_days * axsize[0] + 2
    fig_height = total_rows * axsize[1] + 2# pad each side with an inch


    lick_DA_psth_fig, lick_DA_psth_ax = plt.subplots(total_rows,
                                                     total_days,
                                                     figsize = (fig_width, fig_height),
                                                     sharex = True,
                                                     sharey = sharey,
                                                     squeeze = False)# layout = 'constrained')#, subplotpars={ 'left': one_inch_width, 'top' : one_inch_height, 'hspace': 0.1, 'wspace': 0.1}) #, constrained_layout = True)
    lick_raster_unassigned = True
    lick_PSTH_unassigned = True
    DA_heatmap_unassigned = True
    DA_PSTH_unassigned = True
    for row in np.arange(total_rows):

        if plot_lick_raster and lick_raster_unassigned:
            lick_raster_row = row
            lick_raster_unassigned = False
        elif plot_lick_PSTH and lick_PSTH_unassigned:
            lick_PSTH_row = row
            lick_PSTH_unassigned = False
        elif plot_DA_heatmap and DA_heatmap_unassigned:
            DA_heatmap_row = row
            DA_heatmap_unassigned = False
        elif plot_DA_PSTH and DA_PSTH_unassigned:
            DA_PSTH_row = row
            DA_PSTH_unassigned = False


    #determine max trials per day for raster plot ylim purposes
    raster_ylen =  len(trials_df[((trials_df['animal'] == animal)
                                  & (trials_df['day_num'] >= days[0])
                                 & (trials_df['day_num'] <= days[-1]))]['trial_num'].unique())


    for d_idx, day in enumerate(days):
        animal_day_df = trials_df[((trials_df['animal'] == animal) & (trials_df['day_num'] == day))]
        if not animal_day_df.empty:
            condition = animal_day_df['condition'].iloc[0]
            if align_to_cue:
                cue_on =  0
                cue_off = (animal_day_df['cue_off'] - animal_day_df['cue_on'] ).mean() * scale_factor
                reward_time = (animal_day_df['reward_time'] - animal_day_df['cue_on'] ).mean() * scale_factor
            else:
                cue_on = (animal_day_df['cue_on'] - animal_day_df['reward_time'] ).mean() * scale_factor
                cue_off = (animal_day_df['cue_off'] - animal_day_df['reward_time'] ).mean() * scale_factor
                reward_time = 0

            #plot lick raster and PSTH
            if plot_lick_raster or plot_lick_PSTH:
                lick_times_list = (animal_day_df['licks_all'] * scale_factor).tolist()
                if align_to_cue:
                    lick_times_list = [trial_licks + reward_time for trial_licks in lick_times_list]


                #plot cue and reward times with lines and shading
                if plot_lick_raster:

                    lick_DA_psth_ax[lick_raster_row, d_idx].axvline(x = reward_time,
                                                                    color ='gray',
                                                                    linestyle='dashed',
                                                                    linewidth =  stroke_cue_reward_vertical,
                                                                    )
                    lick_DA_psth_ax[lick_raster_row, d_idx].axvspan(cue_on,
                                                                    cue_off,
                                                                    alpha = alpha_cue_shade,
                                                                    facecolor = color_cue_shade,
                                                                    linewidth = None,
                                                                    )

                    lick_DA_psth_ax[0, d_idx].eventplot(lick_times_list,
                                                        linewidths = stroke_raster,
                                                        linelengths = 0.75,
                                                        colors= 'black',
                                                        lineoffsets = np.arange(len(lick_times_list))+1,
                                                        )
                    lick_DA_psth_ax[lick_raster_row, d_idx].set_ylim([raster_ylen + 0.5, 0.5])
                    lick_DA_psth_ax[lick_raster_row, 0].set_ylabel('trial', fontsize = fontsize_label)
                    #lick_DA_psth_ax[lick_raster_row, d_idx].yaxis.set_major_locator(matplotlib.ticker.LinearLocator(2))

                    lick_DA_psth_ax[lick_raster_row, d_idx].yaxis.set_major_locator(matplotlib.ticker.FixedLocator([1, len(lick_times_list)]))
                    lick_DA_psth_ax[lick_raster_row, d_idx].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))


                if plot_lick_PSTH:
                    lick_DA_psth_ax[lick_PSTH_row, d_idx].axvline(x = reward_time,
                                                                  color ='gray',
                                                                  linestyle='dashed',
                                                                  linewidth =  stroke_cue_reward_vertical,
                                                                  )
                    lick_DA_psth_ax[lick_PSTH_row, d_idx].axhline(y = 0,
                                                                  color ='gray',
                                                                  linestyle='dashed',
                                                                  linewidth =  linewidth_0_lick,
                                                                  alpha = alpha0line,
                                                                  )
                    lick_DA_psth_ax[lick_PSTH_row, d_idx].axvspan(cue_on,
                                                                  cue_off,
                                                                  alpha = alpha_cue_shade,
                                                                  facecolor = color_cue_shade,
                                                                  linewidth = None,
                                                                  )

                    lick_hist_dict = lpf.getLickPSTH(lick_times_list,
                                                     binsize = binsize_for_lick_PSTH,
                                                     total_time_window= total_time_to_plot,
                                                     baseline_period = baseline_time_to_plot,
                                                     in_seconds = plot_in_sec,
                                                     time_window_in_seconds = data_in_s)
                    #print(lick_hist_dict)

                    if smooth_lick:
                        lick_hist_mean = gaussian_filter1d(lick_hist_dict['mean_hist'],
                                                           sigma = smooth_lick,
                                                           )
                        lick_hist_sem= gaussian_filter1d(lick_hist_dict['sem_hist'],
                                                         sigma = smooth_lick,
                                                         )
                    lick_DA_psth_ax[lick_PSTH_row, d_idx].plot(lick_hist_dict['bins'][1:],
                                                               lick_hist_mean,
                                                               color = color_lick_PSTH,
                                                               linewidth = stroke_PSTH,
                                                               linestyle = linestyle_lick,
                                                               )



                    lick_DA_psth_ax[lick_PSTH_row, d_idx].fill_between(lick_hist_dict['bins'][1:],
                                                                       lick_hist_mean - lick_hist_sem,
                                                                       lick_hist_mean + lick_hist_sem,
                                                                       facecolor= color_lick_PSTH,
                                                                       alpha = alpha_PSTH_error,
                                                                       )

                    lick_DA_psth_ax[lick_PSTH_row, d_idx].set_ylim(ylim_lick)
                    lick_DA_psth_ax[lick_PSTH_row, 0].set_ylabel('lick (Hz)',  fontsize = fontsize_label)
                    lick_DA_psth_ax[lick_PSTH_row, d_idx].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
                    lick_DA_psth_ax[lick_PSTH_row, d_idx].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))


            if plot_DA_heatmap or plot_DA_PSTH:

                epoch_dff_times = np.mean(np.array(animal_day_df['epoch_time'].to_list()), axis = 0) * scale_factor

                if align_to_cue:
                    epoch_dff_times = epoch_dff_times + reward_time

                if dff_to_plot =='%':

                    epoch_dff = np.array(animal_day_df['epoch_dff'].to_list())
                    dlight_ylabel = 'dLight\n(% dF/F)'

                epoch_dff_mean = np.mean(epoch_dff, axis = 0)
                epoch_dff_sem = stats.sem(epoch_dff,
                                          ddof = 1,
                                          axis = 0,
                                          nan_policy = 'propagate',
                                          )

                # if norm_to_max_rewards:
                #     max_reward_values = trials_df[trials_df['animal'] == animal].sort_values(['epoch_dff_peak_consume_norm'],ascending=False).groupby(['animal']).head(10)
                #     print(max_reward_values)
                #     DA_normalization = max_reward_values.groupby(['animal'])['epoch_dff_peak_consume_norm'].agg(lambda g: g.iloc[0:norm_to_max_individual_rewards].mean())

                #     print(DA_normalization)
                #     epoch_dff_mean = epoch_dff_mean/ float(DA_normalization)
                #     epoch_dff_sem = epoch_dff_sem /float(DA_normalization)
                if plot_DA_heatmap:
                    try:
                        lick_DA_psth_ax[DA_heatmap_row, d_idx].axvline(x=cue_on,
                                                                       color='gray',
                                                                       linestyle='dashed',
                                                                       linewidth = stroke_cue_reward_vertical,
                                                                       )
                        lick_DA_psth_ax[DA_heatmap_row, d_idx].axvline(x=cue_off,
                                                                       color='gray',
                                                                       linestyle='dashed',
                                                                       linewidth = stroke_cue_reward_vertical,
                                                                       )
                        lick_DA_psth_ax[DA_heatmap_row, d_idx].axvline(x = reward_time,
                                                                       color ='gray',
                                                                       linestyle='dashed',
                                                                       linewidth =  stroke_cue_reward_vertical,
                                                                       )
                        if d_idx == 0:
                            if colorplotmax is None:
                                colorplotmax = np.max(epoch_dff_mean)#*1.5
                            lick_DA_psth_ax[DA_heatmap_row, 0].set_ylabel('trial',
                                                                          fontsize = fontsize_label,
                                                                          )
                        dff_heatmap = lick_DA_psth_ax[DA_heatmap_row, d_idx].imshow(epoch_dff,
                                                                                    cmap=plt.cm.viridis,
                                                                                    interpolation='none',
                                                                                    aspect="auto",
                                                                                    extent=[epoch_dff_times[0],
                                                                                            epoch_dff_times[-1],
                                                                                            len(epoch_dff),
                                                                                            0,
                                                                                            ],
                                                                                    origin = 'upper',
                                                                                    vmax=colorplotmax,
                                                                                    vmin=-1,
                                                                                    )
                        lick_DA_psth_ax[DA_heatmap_row, d_idx].yaxis.set_major_locator(
                            matplotlib.ticker.MultipleLocator(len(lick_times_list)))
                        lick_DA_psth_ax[DA_heatmap_row, d_idx].yaxis.set_minor_locator(
                            matplotlib.ticker.AutoMinorLocator(2))
                    except Exception as e:
                        print(f'error with DA data {animal} day {day} \n {e}')
                if plot_DA_PSTH:
                    try:
                        lick_DA_psth_ax[DA_PSTH_row, d_idx].axvline(x = reward_time,
                                                                    color ='gray',
                                                                    linestyle='dashed',
                                                                    linewidth =  stroke_cue_reward_vertical,
                                                                    )
                        lick_DA_psth_ax[DA_PSTH_row, d_idx].axhline(y = 0,
                                                                    color ='gray',
                                                                    linestyle='dashed',
                                                                    linewidth =  linewidth_0_DA,
                                                                    alpha = alpha0line,
                                                                    )
                        lick_DA_psth_ax[DA_PSTH_row, d_idx].axvspan(cue_on,
                                                                    cue_off,
                                                                    alpha = alpha_cue_shade,
                                                                    facecolor = color_cue_shade,
                                                                    linewidth = None,
                                                                    )
                        lick_DA_psth_ax[DA_PSTH_row, d_idx].plot(epoch_dff_times,
                                                                 epoch_dff_mean,
                                                                 color = color_DA_PSTH,
                                                                 linewidth = stroke_PSTH,
                                                                 linestyle = linestyle_DA,
                                                                 )
                        lick_DA_psth_ax[DA_PSTH_row, d_idx].fill_between(epoch_dff_times,
                                                                         epoch_dff_mean - epoch_dff_sem,
                                                                         epoch_dff_mean + epoch_dff_sem,
                                                                         facecolor= color_DA_PSTH,
                                                                         alpha = alpha_PSTH_error,
                                                                         )
                        lick_DA_psth_ax[DA_PSTH_row, 0].set_ylabel(f'{dlight_ylabel}',
                                                                   fontsize = fontsize_label,
                                                                   )
                        if d_idx == 0:
                            max_dff_day =  0
                            min_dff_day = 0
                        max_dff = np.max(epoch_dff_mean)
                        min_dff = np.min(epoch_dff_mean)
                        if max_dff > max_dff_day:
                            max_dff_day = max_dff
                        if min_dff < min_dff_day:
                            min_dff_day = min_dff
                        if day == days[-1]:
                            rounded_max_avg_dff = (math.floor(max_dff_day / 5) * 5) #rounding to nearest 5 near max to set
                            if rounded_max_avg_dff == 0:
                                rounded_max_avg_dff = 1
                            lick_DA_psth_ax[DA_PSTH_row, 0].set_ylim([min_dff_day*1.2, max_dff_day*1.2])
                            lick_DA_psth_ax[DA_PSTH_row, 0].set_ylim(ylim_DA)
                            lick_DA_psth_ax[DA_PSTH_row, d_idx].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(abs(rounded_max_avg_dff)))
                            #lick_DA_psth_ax[DA_PSTH_row, d_idx].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(2))
                            lick_DA_psth_ax[DA_PSTH_row, d_idx].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
                    except Exception as e:
                        print(f'error with DA data {animal} day {day} \n {e}')

            if d_idx ==0:
                lick_DA_psth_ax[0, d_idx].set_title(f'day: {int(day)}', fontsize = fontsize_title)
            else:
                lick_DA_psth_ax[0, d_idx].set_title(f'{int(day)}', fontsize = fontsize_title)
            lick_DA_psth_ax[-1, d_idx].set_xlim(xlim_in_sec)
            lick_DA_psth_ax[-1, d_idx].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
            lick_DA_psth_ax[-1, d_idx].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
    if plot_DA_heatmap:
        try:
            cbar_ax = lick_DA_psth_ax[DA_heatmap_row, -1].inset_axes([1.02, 0, .05, 1])
            cbar = plt.colorbar(dff_heatmap, cax = cbar_ax, pad= 0, fraction=0.04)# label = '% dF/F')
            cbar.set_label('% dF/F', rotation = 270, va = 'bottom')
        except:
            pass
    lick_DA_psth_fig.suptitle(f'{animal} - {condition}',
                              fontsize = fontsize_title,
                              )
    if align_to_cue:
        lick_DA_psth_fig.supxlabel('time from cue onset (s)',
                                   fontsize = fontsize_label,
                                   )
    else:
        lick_DA_psth_fig.supxlabel('time from reward (s)',
                                   fontsize = fontsize_label,
                                   )
    #for ax in lick_DA_psth_ax.reshape(-1):
    standardize_plot_graphics(lick_DA_psth_ax)
    set_ax_size_inches(axsize[0], axsize[1],  lick_DA_psth_ax)
    if save_fig:
        lick_DA_psth_fig.savefig(os.path.join(fig_path, f'example raster PSTH {animal}.pdf'),
                                 transparent = True,
                                 bbox_inches = 'tight',
                                 bbox_extra_artists = [lick_DA_psth_fig._suptitle,
                                                       lick_DA_psth_fig._supxlabel]
                                 )
        if save_png:
            lick_DA_psth_fig.savefig(os.path.join(fig_path, f'example raster PSTH {animal}.png'),
                                     transparent = True,
                                     bbox_inches = 'tight',
                                     bbox_extra_artists = [lick_DA_psth_fig._suptitle,
                                                           lick_DA_psth_fig._supxlabel]
                                     )
    return lick_DA_psth_fig, lick_DA_psth_ax



def getCumSumLearnedTrialsAndPlot(trial_df,
                                  conditions_to_plot = 'all',
                                  colors_for_conditions = defaultdict(lambda: 'black'),
                                  colors_for_conditions_DA = defaultdict(lambda: 'black'),
                                  percent_max_dist = 0.75,
                                  plot_all_individuals = False,
                                  linewidth_lick = 1,
                                  linewidth_learned_trial = 0.35,
                                  color_learned_trial = 'black',
                                  get_DA_learned_trial = False,
                                  peak_or_auc = 'auc',
                                  get_DA_reward_decrease = False,
                                  get_omission_learned_trial = False,
                                  norm_omission_like_cue = False,
                                  omission_wind = '2',
                                  DA_trial_multiple = 1.5,
                                  linewidth_DA = 1,
                                  linewidth_DA_trial = 0.35,
                                  color_DA_trial = 'k',
                                  plot_vertical = False,
                                  use_trial_normalized_y = True,
                                  sharex = False,
                                  sharey = True,
                                  xlim = [None, None],
                                  sharey_DA = False,
                                  linestyle_DA = (0, (3,1)),
                                  linestyle_lick = 'solid',
                                  linestyle_DA_omission = 'dashdot',
                                  linewidth_diagonal_lick = 0,
                                  linewidth_diagonal_DA = 0,
                                  plot_examples = True,
                                  condition_examples = {},
                                  learning_cutoff = -999,
                                  nonlearners_list =[],
                                  plot_DA_reward = False,
                                  axsize = (1.1, 0.82, 0.1, 0.1),
                                  fontsize_label = 7,
                                  fontsize_ticks = 6,
                                  plot_on_2_lines = False,
                                  save_fig = False,
                                  save_png = False,
                                  fig_path = '',
                                  linestyle_learned_trial = 'solid',
                                  linestyle_learned_trial_DA = 'dashed',
                                  renamed_mice = True,
                                  ylim_lick = [None, None],
                                  ylim_DA = [None, None],
                                  norm_to_max_individual_rewards = 3,
                                  lick_left = True,
                                  cue_type = 'CS_plus',
                                  trial_or_reward_or_omission = 'trials',
                                  plot_omission = False,
                                  plot_lick = True,
                                  ):

    conditions = lpf.get_conditions_as_list(conditions_to_plot, trial_df)
    original_df = trial_df.copy()
    trial_df = trial_df[trial_df['cue_type'] == cue_type].copy()
    trial_df = trial_df[trial_df['condition'].isin(conditions)].copy()
    if (trial_or_reward_or_omission == 'reward'
        or trial_or_reward_or_omission == 'omission'
        ):
        trial_df = lpf.cumLickTrialCount(trial_df,
                                         grouping_var = ['animal',
                                                         'trial_type'
                                                         ]
                                         )
        trial_df = trial_df[trial_df['trial_type'] == trial_or_reward_or_omission].copy()

    elif trial_or_reward_or_omission == 'trials':
        trial_df = lpf.cumLickTrialCount(trial_df,
                                         grouping_var = ['animal',
                                                         'cue_type',
                                                         ]
                                         )
    else:
        raise Exception('unclear whether to plot against: check "trial_or_reward_or_omission" input')
    animals_by_condition = trial_df.groupby(['condition'])['animal'].unique().to_dict()
    #for plotting
    num_animals_by_condition = {x: len(animals_by_condition[x])
                                for x
                                in animals_by_condition.keys()}
    max_animal_single_condition = num_animals_by_condition[max(num_animals_by_condition, key = num_animals_by_condition.get)]

    DA_time_wind = '_500ms' #


    lick_cumsum_df = trial_df.pivot(index = ['cue_trial_num'],
                                    columns = ['condition', 'animal'],
                                    values = 'cumsum_antic_norm',
                                    )
    if get_DA_learned_trial:
        DA_cue_cumsum_df = trial_df.pivot(index = ['cue_trial_num'],
                                      columns = ['condition', 'animal'],
                                      values = f'cumsum_antic_dff_{peak_or_auc}_norm{DA_time_wind}',
                                      )
    if get_DA_reward_decrease:
        DA_reward_cumsum_df = trial_df.pivot(index = ['cue_trial_num'],
                                             columns = ['condition', 'animal'],
                                             values = f'cumsum_consume_dff_{peak_or_auc}_norm{DA_time_wind}',
                                             )
    if get_omission_learned_trial:
        DA_omission_cumsum_df = trial_df.pivot(index = ['cue_trial_num'],
                                             columns = ['condition', 'animal'],
                                             values = f'cumsum_consume_dff_{peak_or_auc}_norm_{omission_wind}s',
                                             )

    title = 'cumsum individuals licks'
    if get_DA_learned_trial or get_DA_reward_decrease:
        if norm_to_max_individual_rewards: # how many rewards responses to average  epoch_dff_auc_consume_norm_lickaligned_500ms
            # max_reward_values = original_df.sort_values([f'epoch_dff_{peak_or_auc}_consume_norm_lickaligned{DA_time_wind}'],ascending=False).groupby(['condition', 'animal']).head(10)
            # DA_normalization = max_reward_values.groupby(['condition', 'animal'])[f'epoch_dff_{peak_or_auc}_consume_norm_lickaligned{DA_time_wind}'].agg(lambda g: g.iloc[0:norm_to_max_individual_rewards].mean())
            max_reward_values = (original_df
                                 .sort_values([f'epoch_dff_{peak_or_auc}_consume_norm_lickaligned{DA_time_wind}'],
                                              ascending=False
                                              )
                                 .groupby(['condition',
                                             'animal'])
                                 .head(10)
                                 )
            DA_normalization = (max_reward_values
                                .groupby(['condition','animal'])
                                [f'epoch_dff_{peak_or_auc}_consume_norm_lickaligned{DA_time_wind}']
                                .agg(lambda g: g.iloc[0:norm_to_max_individual_rewards].mean())
                                )

            if get_omission_learned_trial:
                if norm_omission_like_cue:
                    DA_normalization_omission = DA_normalization
                else:

                    max_reward_values_omission = (original_df
                                                  .sort_values([f'epoch_dff_{peak_or_auc}_consume_norm_lickaligned_{omission_wind}s'],
                                                                ascending=False
                                                                )
                                                  .groupby(['condition',
                                                            'animal'])
                                                  .head(10)
                                                  )
                    DA_normalization_omission = (max_reward_values_omission
                                                 .groupby(['condition',
                                                           'animal'])
                                                 [f'epoch_dff_{peak_or_auc}_consume_norm_lickaligned_{omission_wind}s']
                                                 .agg(lambda g: g.iloc[0:norm_to_max_individual_rewards].mean())
                                                 )

            if get_DA_learned_trial:
                title = title+ ' DA norm to max {norm_to_max_individual_rewards}'
        else:
            DA_normalization = 1
    if get_DA_learned_trial:
        DA_cue_cumsum_df = DA_cue_cumsum_df / DA_normalization
    if get_DA_reward_decrease:
        DA_reward_cumsum_df = DA_reward_cumsum_df / DA_normalization
    if get_omission_learned_trial:
        DA_omission_cumsum_df = DA_omission_cumsum_df / DA_normalization_omission


    trial_normed_lick_cumsum_df =lick_cumsum_df / lick_cumsum_df.count()
    if get_DA_learned_trial:
        trial_normed_DA_cue_cumsum_df = DA_cue_cumsum_df/DA_cue_cumsum_df.count()
    if get_DA_reward_decrease:
        trial_normed_DA_reward_cumsum_df = DA_reward_cumsum_df/DA_reward_cumsum_df.count()
    if get_omission_learned_trial:
        trial_normed_DA_omission_cumsum_df = DA_omission_cumsum_df/ DA_omission_cumsum_df.count()

    if use_trial_normalized_y:
        lick_cumsum_df = trial_normed_lick_cumsum_df
        if get_DA_learned_trial:
            DA_cue_cumsum_df = trial_normed_DA_cue_cumsum_df
        if get_DA_reward_decrease:
            DA_reward_cumsum_df = trial_normed_DA_reward_cumsum_df
        if get_omission_learned_trial:
            DA_omission_cumsum_df = trial_normed_DA_omission_cumsum_df
    if plot_examples == False:
        max_animal_single_condition = max_animal_single_condition - 1
        title = title + ' without examples'
    if plot_vertical:
        total_row = max_animal_single_condition
        total_col = len(conditions)
        title = title + ' vertical'
    else:
        if plot_on_2_lines:
            total_row = len(conditions) *2
            total_col = 9
            title = title + ' on 2 lines'
        else:
            total_row = len(conditions)
            total_col = max_animal_single_condition

    if get_DA_learned_trial:
        title =title + ' with DA'
        DA_ax_list = []

    if plot_all_individuals:
        fig_cumsum_individuals, ax_cumsum_individuals = plt.subplots(total_row,
                                                                     total_col,
                                                                     figsize=(total_col +2 ,
                                                                              total_row + 2),
                                                                     sharex= sharex,
                                                                     sharey = sharey,
                                                                     constrained_layout= False,
                                                                     squeeze = False,
                                                                     )
    learned_trials_dict = dict.fromkeys(conditions, {})
    abruptness_of_change_max_dict = dict.fromkeys(conditions, {})
    abruptness_of_change_learned_trial_dict = dict.fromkeys(conditions, {})
    abruptness_of_change_max_normed_dict = dict.fromkeys(conditions, {})
    abruptness_of_change_learned_trial_normed_dict = dict.fromkeys(conditions, {})

    if get_DA_learned_trial:
        DA_learned_trials_dict = dict.fromkeys(conditions, {})
        DA_abruptness_of_change_max_dict = dict.fromkeys(conditions, {})
        DA_abruptness_of_change_learned_trial_dict = dict.fromkeys(conditions, {})
        DA_abruptness_of_change_max_normed_dict = dict.fromkeys(conditions, {})
        DA_abruptness_of_change_learned_trial_normed_dict = dict.fromkeys(conditions, {})
        lag_to_learn_dict = dict.fromkeys(conditions, {})
    if get_DA_reward_decrease:

        DA_reward_decrease_trials_dict = dict.fromkeys(conditions, {})
        DA_reward_abruptness_of_change_max_dict = dict.fromkeys(conditions, {})
        DA_reward_abruptness_of_change_learned_trial_dict = dict.fromkeys(conditions, {})
        DA_reward_abruptness_of_change_max_normed_dict = dict.fromkeys(conditions, {})
        DA_reward_abruptness_of_change_learned_trial_normed_dict = dict.fromkeys(conditions, {})
        lag_reward_decrease_to_DA_cue = dict.fromkeys(conditions, {})
        lag_reward_decrease_to_learning = dict.fromkeys(conditions, {})


    if get_omission_learned_trial:
        DA_omission_learned_trials_dict = dict.fromkeys(conditions, {})
        DA_omission_abruptness_of_change_max_dict = dict.fromkeys(conditions, {})
        DA_omission_abruptness_of_change_learned_trial_dict = dict.fromkeys(conditions, {})
        DA_omission_abruptness_of_change_max_normed_dict = dict.fromkeys(conditions, {})
        DA_omission_abruptness_of_change_learned_trial_normed_dict = dict.fromkeys(conditions, {})
        lag_learn_to_omission_dict = dict.fromkeys(conditions, {})
        lag_cue_to_omission_dict = dict.fromkeys(conditions, {})


    learned_trial_params_lick_dict = dict.fromkeys(conditions, {})
    learned_trial_params_DA_cue_dict = dict.fromkeys(conditions, {})

    for con_num, condition in enumerate(conditions):
        example_counter = 0
        example_animal = ''
        if plot_examples == False:
            example_animal = condition_examples[condition]
            #fig, ax =




        learned_trials_dict[condition] = dict.fromkeys(animals_by_condition[condition], [])
        abruptness_of_change_max_dict[condition] = dict.fromkeys(animals_by_condition[condition], [])
        abruptness_of_change_learned_trial_dict[condition] = dict.fromkeys(animals_by_condition[condition], [])
        abruptness_of_change_max_normed_dict[condition] = dict.fromkeys(animals_by_condition[condition], [])
        abruptness_of_change_learned_trial_normed_dict[condition] = dict.fromkeys(animals_by_condition[condition], [])
        learned_trial_params_lick_dict[condition] = dict.fromkeys(animals_by_condition[condition], [])
        if get_DA_learned_trial:
            DA_learned_trials_dict[condition] = dict.fromkeys(animals_by_condition[condition], [])
            DA_abruptness_of_change_max_dict[condition] = dict.fromkeys(animals_by_condition[condition], [])
            DA_abruptness_of_change_learned_trial_dict[condition] = dict.fromkeys(animals_by_condition[condition], [])
            DA_abruptness_of_change_max_normed_dict[condition] = dict.fromkeys(animals_by_condition[condition], [])
            DA_abruptness_of_change_learned_trial_normed_dict[condition] = dict.fromkeys(animals_by_condition[condition], [])
            lag_to_learn_dict[condition] = dict.fromkeys(animals_by_condition[condition], [])
            learned_trial_params_DA_cue_dict[condition] = dict.fromkeys(animals_by_condition[condition], [])
        if get_DA_reward_decrease:
            DA_reward_decrease_trials_dict[condition] = dict.fromkeys(animals_by_condition[condition], [])
            DA_reward_abruptness_of_change_max_dict[condition]  = dict.fromkeys(animals_by_condition[condition], [])
            DA_reward_abruptness_of_change_learned_trial_dict[condition]  = dict.fromkeys(animals_by_condition[condition], [])
            DA_reward_abruptness_of_change_max_normed_dict[condition]  = dict.fromkeys(animals_by_condition[condition], [])
            DA_reward_abruptness_of_change_learned_trial_normed_dict[condition] =   dict.fromkeys(animals_by_condition[condition], [])
            lag_reward_decrease_to_DA_cue[condition] =  dict.fromkeys(animals_by_condition[condition], [])
            lag_reward_decrease_to_learning[condition] = dict.fromkeys(animals_by_condition[condition], [])

        if get_omission_learned_trial:
            DA_omission_learned_trials_dict[condition] = dict.fromkeys(animals_by_condition[condition], [])
            DA_omission_abruptness_of_change_max_dict[condition] = dict.fromkeys(animals_by_condition[condition], [])
            DA_omission_abruptness_of_change_learned_trial_dict[condition] = dict.fromkeys(animals_by_condition[condition], [])
            DA_omission_abruptness_of_change_max_normed_dict[condition] = dict.fromkeys(animals_by_condition[condition], [])
            DA_omission_abruptness_of_change_learned_trial_normed_dict[condition] = dict.fromkeys(animals_by_condition[condition], [])
            lag_learn_to_omission_dict[condition] = dict.fromkeys(animals_by_condition[condition], [])
            lag_cue_to_omission_dict[condition] = dict.fromkeys(animals_by_condition[condition], [])
        sorted_animals = natsorted(animals_by_condition[condition])
        np.seterr(divide='ignore', invalid='ignore')

        for animal_num, animal in enumerate(sorted_animals):



            lick_cumsum_y = lick_cumsum_df[condition, animal].to_numpy() #cumsum licks
            trials_cumsum_x = lick_cumsum_df.index.values #trial numbers
            trials_cumsum_x = trials_cumsum_x[~np.isnan(lick_cumsum_y)] #filter out NaNs in licks and trials
            lick_cumsum_y = lick_cumsum_y[~np.isnan(lick_cumsum_y)]
            #print(lick_cumsum_y)
            learned_trial_params_lick = lpf.getCumsumChangePoint(trials_cumsum_x,
                                                                 lick_cumsum_y,
                                                                 percent_max_dist = percent_max_dist,
                                                                 data_direction = 'increase',
                                                                 animal_name = animal)
            #print(learned_trial_params_lick['learned_trial'])
            if ((max(lick_cumsum_y) > learning_cutoff) and (animal not in nonlearners_list)):
                learned_trials_dict[condition][animal] =  learned_trial_params_lick['learned_trial']

                abruptness_of_change_max_dict[condition][animal] =  learned_trial_params_lick['max_dist']
                abruptness_of_change_max_normed_dict[condition][animal] = learned_trial_params_lick['max_dist_norm']

                abruptness_of_change_learned_trial_dict[condition][animal] =  learned_trial_params_lick['dist_at_learned_trial']
                abruptness_of_change_learned_trial_normed_dict[condition][animal] = learned_trial_params_lick['dist_at_learned_trial_norm']

                learned_trial_params_lick_dict[condition][animal] = learned_trial_params_lick
                learned_trial_for_DA_cutoff = learned_trial_params_lick['learned_trial']
            else:
                learned_trial_for_DA_cutoff = np.max(trials_cumsum_x)
            if get_DA_learned_trial:
                DA_cumsum_y = DA_cue_cumsum_df[condition, animal].to_numpy()

                DA_cumsum_y = DA_cumsum_y[~np.isnan(DA_cumsum_y)]

                if len(DA_cumsum_y) > 0:
                    #now do same for DA



                    da_full_y = DA_cumsum_y

                    DA_cumsum_y = DA_cumsum_y[trials_cumsum_x
                                              <= (learned_trial_for_DA_cutoff
                                                  * DA_trial_multiple)]
                    DA_cumsum_y = DA_cumsum_y[~np.isnan(DA_cumsum_y)]
                    DA_trials_cumsum_x = trials_cumsum_x[trials_cumsum_x
                                                         <= (learned_trial_for_DA_cutoff
                                                             * DA_trial_multiple)]
                    DA_trials_cumsum_x = DA_trials_cumsum_x[~np.isnan(DA_trials_cumsum_x)]

                    learned_trial_params_DA_cue = lpf.getCumsumChangePoint(DA_trials_cumsum_x,
                                                                           DA_cumsum_y,
                                                                           percent_max_dist =
                                                                           percent_max_dist,
                                                                           data_direction = 'increase',
                                                                           animal_name = animal)


                    learned_trial_params_DA_cue_dict[condition][animal] = learned_trial_params_DA_cue
                    DA_learned_trials_dict[condition][animal] =  learned_trial_params_DA_cue['learned_trial']
                    DA_abruptness_of_change_max_dict[condition][animal] = learned_trial_params_DA_cue['max_dist']
                    DA_abruptness_of_change_learned_trial_dict[condition][animal] = learned_trial_params_DA_cue['dist_at_learned_trial']
                    DA_abruptness_of_change_max_normed_dict[condition][animal] = learned_trial_params_DA_cue['max_dist_norm']
                    DA_abruptness_of_change_learned_trial_normed_dict[condition][animal] = learned_trial_params_DA_cue['dist_at_learned_trial_norm']
                    lag_to_learn_dict[condition][animal] = learned_trial_params_lick['learned_trial'] - learned_trial_params_DA_cue['learned_trial']
                    if get_DA_reward_decrease:
                        DA_reward_cumsum_y = DA_reward_cumsum_df[condition, animal].to_numpy()

                        DA_reward_cumsum_y = DA_reward_cumsum_y[~np.isnan(DA_reward_cumsum_y)]

                        DA_reward_full_y = DA_reward_cumsum_y

                       # DA_reward_cumsum_y = DA_reward_cumsum_y[trials_cumsum_x <= (learned_trial_for_DA_cutoff*DA_trial_multiple)]
                        DA_reward_cumsum_y = DA_reward_cumsum_y[~np.isnan(DA_reward_cumsum_y)]
                        DA_reward_trials_cumsum_x = trials_cumsum_x #[trials_cumsum_x<=(learned_trial_for_DA_cutoff*DA_trial_multiple)]
                        DA_reward_trials_cumsum_x = DA_reward_trials_cumsum_x[~np.isnan(DA_reward_trials_cumsum_x)]

                        learned_trial_params_DA_reward = lpf.getCumsumChangePoint(DA_reward_trials_cumsum_x,
                                                                                  DA_reward_cumsum_y,
                                                                                  percent_max_dist = percent_max_dist,
                                                                                  data_direction = 'decrease',
                                                                                  animal_name = animal)

                        DA_reward_decrease_trials_dict[condition][animal] =  learned_trial_params_DA_reward['learned_trial']
                        DA_reward_abruptness_of_change_max_dict[condition][animal] = learned_trial_params_DA_reward['max_dist']
                        DA_reward_abruptness_of_change_learned_trial_dict[condition][animal] = learned_trial_params_DA_reward['dist_at_learned_trial']
                        DA_reward_abruptness_of_change_max_normed_dict[condition][animal] = learned_trial_params_DA_reward['max_dist_norm']
                        DA_reward_abruptness_of_change_learned_trial_normed_dict[condition][animal] = learned_trial_params_DA_reward['dist_at_learned_trial_norm']
                        lag_reward_decrease_to_DA_cue[condition][animal] = learned_trial_params_DA_cue['learned_trial'] - learned_trial_params_DA_reward['learned_trial']
                        lag_reward_decrease_to_learning[condition][animal] = learned_trial_params_lick['learned_trial'] - learned_trial_params_DA_reward['learned_trial']

                    if get_omission_learned_trial:
                        DA_omission_cumsum_y = DA_omission_cumsum_df[condition, animal].to_numpy()

                        DA_omission_cumsum_y = DA_omission_cumsum_y[~np.isnan(DA_omission_cumsum_y)]

                        DA_omission_full_y = DA_omission_cumsum_y

                       # DA_omission_cumsum_y = DA_omission_cumsum_y[trials_cumsum_x <= (learned_trial_for_DA_cutoff*DA_trial_multiple)]
                        DA_omission_cumsum_y = DA_omission_cumsum_y[~np.isnan(DA_omission_cumsum_y)]
                        DA_omission_trials_cumsum_x = trials_cumsum_x #[trials_cumsum_x<=(learned_trial_for_DA_cutoff*DA_trial_multiple)]
                        DA_omission_trials_cumsum_x = DA_omission_trials_cumsum_x[~np.isnan(DA_omission_trials_cumsum_x)]

                        learned_trial_params_DA_omission = lpf.getCumsumChangePoint(DA_omission_trials_cumsum_x,
                                                                                  DA_omission_cumsum_y,
                                                                                  percent_max_dist = percent_max_dist,
                                                                                  data_direction = 'decrease',
                                                                                  animal_name = animal)

                        DA_omission_learned_trials_dict[condition][animal] =  learned_trial_params_DA_omission['learned_trial']
                        DA_omission_abruptness_of_change_max_dict[condition][animal] = learned_trial_params_DA_omission['max_dist']
                        DA_omission_abruptness_of_change_learned_trial_dict[condition][animal] = learned_trial_params_DA_omission['dist_at_learned_trial']
                        DA_omission_abruptness_of_change_max_normed_dict[condition][animal] = learned_trial_params_DA_omission['max_dist_norm']
                        DA_omission_abruptness_of_change_learned_trial_normed_dict[condition][animal] = learned_trial_params_DA_omission['dist_at_learned_trial_norm']
                        lag_learn_to_omission_dict[condition][animal] = learned_trial_params_DA_omission['learned_trial'] - learned_trial_params_lick['learned_trial']
                        lag_cue_to_omission_dict[condition][animal] = learned_trial_params_DA_omission['learned_trial'] - learned_trial_params_DA_cue['learned_trial']




            if example_animal == animal:
                example_counter = 1
            if plot_all_individuals:
                if plot_vertical:
                    left_individual_ax = ax_cumsum_individuals[animal_num - example_counter, con_num]
                    fig_cumsum_individuals.suptitle(condition, position ='left')
                else:
                    if plot_on_2_lines:
                        if animal_num - example_counter < 9:
                            individual_ax = ax_cumsum_individuals[con_num *2, animal_num- example_counter]
                        else:
                            individual_ax = ax_cumsum_individuals[con_num *2 +1, animal_num- example_counter-9]
                    else:
                        individual_ax = ax_cumsum_individuals[con_num, animal_num- example_counter]
                    ax_cumsum_individuals[con_num,0].set_ylabel(condition,  rotation = 'horizontal', ha = 'right')
                if example_animal != animal:
                    individual_ax.axhline(0,
                                          linestyle = (0,(4,2)),
                                          color = 'gray',
                                          alpha =1,
                                          linewidth = 1,
                                          )
                    if plot_lick:
                        individual_ax.plot(trials_cumsum_x,
                                           lick_cumsum_y,
                                           color = colors_for_conditions[condition],
                                           linewidth = linewidth_lick,
                                           )
                    if renamed_mice:
                        individual_ax.set_title(animal, fontsize = 6)
                    else:
                        #individual_ax.set_title(animal.split('_')[-2]+'_'+animal.split('_')[-1])
                        individual_ax.set_title(animal.split('_')[-1])


                    if ((max(lick_cumsum_y) > learning_cutoff) and plot_lick and (animal not in nonlearners_list)):
                        individual_ax.plot(learned_trial_params_lick['diag_x'],
                                           learned_trial_params_lick['diag_y'],
                                           linewidth = linewidth_diagonal_lick,
                                           linestyle = 'dashed',
                                           )
                        individual_ax.axvline(learned_trial_params_lick['learned_trial'],
                                              linewidth = linewidth_learned_trial,
                                              color = color_learned_trial,
                                              linestyle = linestyle_learned_trial,
                                              )
                    if get_DA_learned_trial and (len(DA_cumsum_y) > 0):
                        if plot_lick:
                            DA_ax =individual_ax.twinx()
                        else:
                            DA_ax =individual_ax
                        DA_ax_list.append(DA_ax)
                        DA_ax.axvline(learned_trial_params_DA_cue['learned_trial'],
                                      linewidth = linewidth_DA_trial,
                                      color = color_DA_trial,
                                      linestyle = linestyle_learned_trial_DA,
                                      )
                        DA_ax.plot(trials_cumsum_x,
                                   da_full_y,
                                   color = colors_for_conditions_DA[condition],
                                   linewidth = linewidth_DA,
                                   linestyle = linestyle_DA,
                                   )
                        DA_ax.plot(learned_trial_params_DA_cue['diag_x'],
                                   learned_trial_params_DA_cue['diag_y'],
                                   linewidth = linewidth_diagonal_DA,
                                   linestyle = 'dashed',
                                   )

                        if plot_DA_reward:
                            DA_ax.axvline(learned_trial_params_DA_reward['learned_trial'],
                                          linewidth = linewidth_DA_trial,
                                          color = 'k',
                                          linestyle = 'dashdot',
                                          )
                            DA_ax.plot(trials_cumsum_x,
                                       DA_reward_full_y,
                                       color = colors_for_conditions_DA[condition],
                                       linewidth = linewidth_DA,
                                       linestyle = 'dashdot',
                                       )
                            DA_ax.plot(learned_trial_params_DA_reward['diag_x'],
                                       learned_trial_params_DA_reward['diag_y'],
                                       linewidth = linewidth_diagonal_DA,
                                       linestyle = 'dashdot',
                                       )

                        if plot_omission:
                            DA_ax.axvline(learned_trial_params_DA_omission['learned_trial'],
                                          linewidth = linewidth_DA_trial,
                                          color = color_DA_trial,
                                          linestyle = linestyle_DA_omission,
                                          )
                            DA_ax.plot(trials_cumsum_x,
                                       DA_omission_full_y,
                                       color = colors_for_conditions_DA[condition],
                                       linewidth = linewidth_DA,
                                       linestyle = linestyle_DA_omission,
                                       )
                            DA_ax.plot(learned_trial_params_DA_omission['diag_x'],
                                       learned_trial_params_DA_omission['diag_y'],
                                       linewidth = linewidth_diagonal_DA,
                                       linestyle = linestyle_DA_omission,
                                       )
                    if max(trials_cumsum_x) <= 8:
                        individual_ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(4)) #individual_ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([ 2, 4, 6,8]))
                        individual_ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))


                    elif max(trials_cumsum_x) <= 20:
                        individual_ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5)) #individual_ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([ 2, 4, 6,8]))
                        individual_ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
                        #individual_ax.set_xticklabels([ '2',  '4', '6', '8'])
                    elif max(trials_cumsum_x) <= 50:
                        individual_ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(20))
                        individual_ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
                    elif max(trials_cumsum_x) <= 80:
                        individual_ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(40))
                        individual_ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
                    elif max(trials_cumsum_x) <= 350:
                        individual_ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(100))
                        individual_ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
                    elif max(trials_cumsum_x) <= 600:
                        individual_ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(200))
                        individual_ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
                    elif max(trials_cumsum_x) <= 800:
                        individual_ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(400))
                        individual_ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
                    elif max(trials_cumsum_x) <= 2000:
                        individual_ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1000))
                        individual_ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
                    else:
                        individual_ax.xaxis.set_major_locator(matplotlib.ticker.AutoLocator())
                        individual_ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
    np.seterr(divide='warn', invalid='warn')
    if plot_all_individuals:
        fig_cumsum_individuals.supxlabel('trial', fontsize= fontsize_label)
        fig_cumsum_individuals.supylabel('cumsum(anticipatory licks)/num trials',
                                         fontsize = fontsize_label,
                                         ha = 'center',
                                         )
        if get_DA_learned_trial and sharey_DA:
            ylim_max_DA = max(map(lambda x: x.get_ylim()[1], DA_ax_list))
            ylim_min_DA = min(map(lambda x: x.get_ylim()[0], DA_ax_list))
            for ax in DA_ax_list:
                ax.sharey(DA_ax_list[0])
                if not ax.get_subplotspec().is_last_col():
                    ax.tick_params(labelright=False)
            DA_ax_list[0].set_ylim([ylim_min_DA, ylim_max_DA])
            DA_ax_list[0].set_ylim(ylim_DA)
        if plot_lick:
            ax_cumsum_individuals[0,0].set_ylim(ylim_lick)
        else:
            ax_cumsum_individuals[0,0].set_ylim(ylim_DA)
        if sharex:
            ax_cumsum_individuals[0,0].set_xlim = xlim
        ax_cumsum_individuals[0,0].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
        standardize_plot_graphics(ax_cumsum_individuals)
        set_ax_size_inches(axsize[0],
                               axsize[1],
                               ax_cumsum_individuals,
                               axsize[2],
                               axsize[3],)

        if save_fig:
            new_title = lpf.cleanStringForFilename(title)
            if save_png:
                fig_cumsum_individuals.savefig(os.path.join(fig_path, f'{new_title}' + '_'.join(conditions)+'.png'),
                                               dpi = 600,
                                               transparent = True,
                                               bbox_inches = 'tight'
                                               )
            fig_cumsum_individuals.savefig(os.path.join(fig_path, f'{new_title}' + '_'.join(conditions)+'.pdf'),
                                           transparent = True,
                                           bbox_inches = 'tight'
                                           )
    summary_dict = {'learned_trial_lick': learned_trials_dict,
                    'abruptness_max_lick': abruptness_of_change_max_dict,
                    'abruptness_max_lick_norm': abruptness_of_change_max_normed_dict,
                    'abruptness_learned_trial_lick': abruptness_of_change_learned_trial_dict,
                    'abruptness_learned_trial_lick_norm': abruptness_of_change_learned_trial_normed_dict,
                    'trials_cumsum_x': trials_cumsum_x
                    }
    if plot_all_individuals:
        summary_dict['fig'] = fig_cumsum_individuals
        summary_dict['ax'] = ax_cumsum_individuals


    if get_DA_learned_trial:
        summary_dict_DA = {'learned_trial_DA': DA_learned_trials_dict,
                            'abruptness_max_DA': DA_abruptness_of_change_max_dict,
                            'abruptness_max_DA_norm': DA_abruptness_of_change_max_normed_dict,
                            'abruptness_learned_trial_DA': DA_abruptness_of_change_learned_trial_dict,
                            'abruptness_learned_trial_DA_norm': DA_abruptness_of_change_learned_trial_normed_dict,
                            'lag_to_learn':lag_to_learn_dict,
                            }
        summary_dict = {**summary_dict, **summary_dict_DA}
    if get_DA_reward_decrease:
        summary_dict_DA_reward ={'learned_trial_DA_reward': DA_reward_decrease_trials_dict,
                                'abruptness_max_DA_reward': DA_reward_abruptness_of_change_max_dict,
                                'abruptness_max_DA_reward_norm': DA_reward_abruptness_of_change_max_normed_dict,
                                'abruptness_learned_trial_DA_reward': DA_reward_abruptness_of_change_learned_trial_dict,
                                'abruptness_learned_trial_DA_reward_norm': DA_reward_abruptness_of_change_learned_trial_normed_dict,
                                'lag_DA_reward_to_cue':lag_reward_decrease_to_DA_cue,
                                'lag_DA_reward_to_learn':lag_reward_decrease_to_learning
                                }
        summary_dict = {**summary_dict, **summary_dict_DA_reward}
    if get_omission_learned_trial:
        summary_dict_DA_omission = {'learned_trial_DA_omission': DA_omission_learned_trials_dict,
                                'abruptness_max_DA_omission': DA_omission_abruptness_of_change_max_dict,
                                'abruptness_max_DA_omission_norm': DA_omission_abruptness_of_change_max_normed_dict,
                                'abruptness_learned_trial_DA_omission': DA_omission_abruptness_of_change_learned_trial_dict,
                                'abruptness_learned_trial_DA_omission_norm': DA_omission_abruptness_of_change_learned_trial_normed_dict,
                                'lag_DA_omission_to_cue':lag_cue_to_omission_dict,
                                'lag_DA_omission_to_learn':lag_learn_to_omission_dict
                                }


        summary_dict = {**summary_dict, **summary_dict_DA_omission}

    return summary_dict


def plotExampleCumsumLearnedTrial(
                                trial_df,
                                animal,
                                # Data selection
                                plot_da=False,
                                plot_lick=True,
                                plot_cumsum=True,
                                # DA-specific parameters
                                da_trial_multiple=1.5,
                                norm_to_max_individual_rewards=3,
                                lick_left=True,
                                # Normalization
                                norm_lick_left=False,
                                use_trial_normalized_y=True,
                                plot_norm_and_raw=False,
                                # Colors
                                color_lick='#7A4A9D',
                                color_da='dodgerblue',
                                color_learned_trial='black',
                                color_da_trial='black',
                                # Line styles
                                linestyle_lick='solid',
                                linestyle_da=(0, (3, 1)),
                                linestyle_learned_trial='solid',
                                linestyle_learned_trial_da='dashed',
                                linestyle0_lick=(0, (4, 2)),
                                # Line widths
                                linewidth_lick=1.5,
                                linewidth_da=1,
                                linewidth_learned_trial=0.35,
                                linewidth_da_trial=0.25,
                                linewidth_diagonal_lick=0,
                                linewidth_diagonal_da=0,
                                linewidth_0_lick=1,
                                linewidth_0_da=0,
                                # Other parameters
                                percent_max_dist=0.75,
                                alpha0line=1,
                                axsize=(1, 1),
                                ylim_left=[None, None],
                                ylim_right=[None, None],
                                xlim=[None, None],
                                # Font sizes
                                fontsize_title=7,
                                fontsize_label=7,
                                fontsize_ticks=6,
                                # Save options
                                save_fig=False,
                                save_png=False,
                                fig_path='',
                            ):
    """
    Plot cumulative sum of learned trial data for licks and/or DA.

    Parameters
    ----------
    trial_df : DataFrame
        Trial data
    animal : str
        Animal identifier
    plot_da : bool
        Whether to plot DA data
    plot_lick : bool
        Whether to plot lick data
    plot_cumsum : bool
        Whether to plot cumsum (vs average)
    ... (other parameters)

    Returns
    -------
    ax or tuple of axes
        Returns single axis or tuple of (left_ax, right_ax) if dual plot
    """

    # Prepare lick data
    if plot_lick:
        lick_cumsum_df = trial_df.pivot(
            index='cue_trial_num',
            columns='animal',
            values='cumsum_antic_norm'
        )
        lick_avg_df = trial_df.pivot(
            index='cue_trial_num',
            columns='animal',
            values='antic_norm_rate_change'
        )
        # Apply trial normalization if needed
        trial_normed_lick_cumsum_df = lick_cumsum_df / lick_cumsum_df.count()

    # Prepare DA data if needed
    if plot_da:
        da_cue_cumsum_df = trial_df.pivot(
            index='cue_trial_num',
            columns='animal',
            values='cumsum_antic_dff_auc_norm_500ms'
        )
        da_cue_cumsum_df = dw.normalize_DA_cue_reward(da_cue_cumsum_df,
                                                      trial_df,
                                                      norm_to_max_individual_rewards = norm_to_max_individual_rewards,
                                                      flatten_norm_df = True,
                                                      )
        trial_normed_da_cue_cumsum_df = da_cue_cumsum_df / da_cue_cumsum_df.count()



    # Select data to use based on normalization setting
    if use_trial_normalized_y or (norm_lick_left and not plot_da):
        lick_cumsum_df = trial_normed_lick_cumsum_df
        scale_multiplier = lick_cumsum_df[animal].count()
        if plot_da:
            da_cue_cumsum_df = trial_normed_da_cue_cumsum_df
    else:
        scale_multiplier = 1 / lick_cumsum_df[animal].count()

    # Extract lick data arrays
    lick_cumsum_y = lick_cumsum_df[animal].to_numpy()
    trials_cumsum_x = lick_cumsum_df.index.values
    trials_cumsum_x = trials_cumsum_x[~np.isnan(lick_cumsum_y)]
    lick_cumsum_y = lick_cumsum_y[~np.isnan(lick_cumsum_y)]

    # Calculate lick learned trial
    learned_trial_params_lick = lpf.getCumsumChangePoint(
        trials_cumsum_x,
        lick_cumsum_y,
        percent_max_dist=percent_max_dist,
        data_direction='increase',
        animal_name=animal
    )

    # Prepare DA data if needed
    if plot_da:
        da_cumsum_y = da_cue_cumsum_df[animal].to_numpy()
        da_cumsum_y = da_cumsum_y[~np.isnan(da_cumsum_y)]
        da_full_y = da_cumsum_y

        # Limit DA to multiple of lick learned trial
        da_cutoff = learned_trial_params_lick['learned_trial'] * da_trial_multiple
        da_cumsum_y = da_cumsum_y[trials_cumsum_x <= da_cutoff]
        da_cumsum_y = da_cumsum_y[~np.isnan(da_cumsum_y)]
        da_trials_cumsum_x = trials_cumsum_x[trials_cumsum_x <= da_cutoff]
        da_trials_cumsum_x = da_trials_cumsum_x[~np.isnan(da_trials_cumsum_x)]

        # Calculate DA learned trial
        learned_trial_params_da_cue = lpf.getCumsumChangePoint(
            da_trials_cumsum_x,
            da_cumsum_y,
            percent_max_dist=percent_max_dist,
            data_direction='increase',
            animal_name=animal
        )

    title = _build_title_example_cumsum(animal,
                                        plot_lick,
                                        plot_da,
                                        plot_cumsum,
                                        norm_to_max_individual_rewards,
                                        use_trial_normalized_y,
                                        lick_left,
                                        norm_lick_left,
                                        plot_norm_and_raw,
                                        )
    # Create figure
    fig_cumsum_example, ax_cumsum_example_left = plt.subplots(
        1, 1,
       #  figsize=(axsize[0] + 2, axsize[1] + 2),
       # constrained_layout=True
    )

    # Determine which data to plot and where
    if plot_da and plot_lick:
        ax_cumsum_example_right = ax_cumsum_example_left.twinx()
        ax_lick = ax_cumsum_example_left if lick_left else ax_cumsum_example_right
        ax_da = ax_cumsum_example_right if lick_left else ax_cumsum_example_left
    elif plot_da:
        ax_da = ax_cumsum_example_left
        ax_lick = None
    else:
        ax_lick = ax_cumsum_example_left
        ax_da = None
        if plot_norm_and_raw:
            ax_cumsum_example_right = ax_cumsum_example_left.twinx()

    # Plot lick data
    if plot_lick and ax_lick is not None:
        lick_to_plot = lick_cumsum_y if plot_cumsum else lick_avg_df[animal].to_numpy()[
            ~np.isnan(lick_avg_df[animal].to_numpy())
        ]

        ax_lick.axhline(
            0,
            linestyle=linestyle0_lick,
            color='gray',
            alpha=alpha0line,
            linewidth=linewidth_0_lick
        )
        ax_lick.plot(
            trials_cumsum_x,
            lick_to_plot,
            color=color_lick,
            linewidth=linewidth_lick,
            linestyle=linestyle_lick
        )
        if plot_cumsum:
            ax_lick.plot(
                learned_trial_params_lick['diag_x'],
                learned_trial_params_lick['diag_y'],
                linewidth=linewidth_diagonal_lick,
                linestyle='dashed',
                color='gray'
            )

        ax_lick.axvline(
            learned_trial_params_lick['learned_trial'],
            linewidth=linewidth_learned_trial,
            color=color_learned_trial,
            linestyle=linestyle_learned_trial
        )


    # Plot DA data
    if plot_da and ax_da is not None:
        ax_da.axhline(
            0,
            linestyle=linestyle0_lick,
            color='gray',
            alpha=alpha0line,
            linewidth=linewidth_0_da
        )
        ax_da.plot(
            trials_cumsum_x,
            da_full_y,
            color=color_da,
            linewidth=linewidth_da,
            linestyle=linestyle_da
        )
        ax_da.plot(
            learned_trial_params_da_cue['diag_x'],
            learned_trial_params_da_cue['diag_y'],
            linewidth=linewidth_diagonal_da,
            linestyle='dashed'
        )
        ax_da.axvline(
            learned_trial_params_da_cue['learned_trial'],
            linewidth=linewidth_da_trial,
            color=color_da_trial,
            linestyle=linestyle_learned_trial_da
        )

    # Set y-axis labels
    _set_y_labels(
        ax_lick, ax_da, plot_lick, plot_da, plot_cumsum,
        use_trial_normalized_y, norm_lick_left, lick_left,
        fontsize_label
    )

    # Handle y-axis limits
    _set_y_limits(
        ax_cumsum_example_left, ax_lick, ax_da, plot_lick, plot_da,
        plot_norm_and_raw, lick_left, ylim_left, ylim_right,
        lick_cumsum_y, scale_multiplier
    )

    # Set x-axis ticks
    _set_x_ticks(ax_cumsum_example_left, trials_cumsum_x)

    # Set x-axis label
    ax_cumsum_example_left.set_xlabel('trial', fontsize=fontsize_label)

    # Standardize plot graphics
    fig_cumsum_example.set_tight_layout(True)
    standardize_plot_graphics(ax_cumsum_example_left)
    set_ax_size_inches(axsize[0], axsize[1], ax_cumsum_example_left)

    # Save figure if requested
    if save_fig:
        new_title = lpf.cleanStringForFilename(title)
        if save_png:
            fig_cumsum_example.savefig(
                os.path.join(fig_path, f'{new_title}.png'),
                dpi=600,
                transparent=True,
                bbox_inches = 'tight',
            )
        fig_cumsum_example.savefig(
            os.path.join(fig_path, f'{new_title}.pdf'),
            transparent=True,
            bbox_inches = 'tight',
        )

    # Return appropriate axes
    if plot_da and plot_lick:
        return ax_cumsum_example_left, ax_cumsum_example_right
    else:
        return ax_cumsum_example_left

def _build_title_example_cumsum(animal,
                                plot_lick,
                                plot_da,
                                plot_cumsum,
                                norm_to_max_individual_rewards,
                                use_trial_normalized_y,
                                lick_left,
                                norm_lick_left,
                                plot_norm_and_raw,
                                ):
    title_parts = []
    if plot_da:
        title_parts.append('example cumsum DA')
        if norm_to_max_individual_rewards:
            title_parts.append(f'norm to max {norm_to_max_individual_rewards}')
        if use_trial_normalized_y:
            title_parts.append('trial normalized')
        if plot_lick:
            title_parts.append('with lick')
            title_parts.append('on left' if lick_left else 'on right')
    else:
        title_parts.append('example cumsum lick' if plot_cumsum else 'example avg lick')
        if norm_lick_left and plot_norm_and_raw:
            title_parts.append('norm and raw-norm left')
        elif not norm_lick_left and plot_norm_and_raw:
            title_parts.append('norm and raw-norm right')

    title = f"{' '.join(title_parts)} {animal}"
    return title

def _set_y_labels(ax_lick, ax_da, plot_lick, plot_da, plot_cumsum,
                  use_trial_normalized_y, norm_lick_left, lick_left,
                  fontsize_label):
    """Helper function to set y-axis labels."""
    if plot_lick and ax_lick is not None:
        if plot_cumsum:
            if use_trial_normalized_y or norm_lick_left:
                label = 'cumsum(licks to cue)/\nnum trials'
            else:
                label = 'cumsum(licks to cue)'
        else:
            label = 'lick rate to cue (Hz)'

        rotation = 270 if (plot_da and not lick_left) else 90
        va = 'bottom' #if rotation == 270 else 'center'
        ax_lick.set_ylabel(label, fontsize=fontsize_label, rotation=rotation, va=va)

    if plot_da and ax_da is not None:
        if use_trial_normalized_y:
            label = 'norm. cumsum(DA to cue)/\nnum trials'
        else:
            label = 'norm. cumsum(DA to cue)'

        rotation = 270 if (plot_lick and lick_left) else 90
        va = 'bottom' #if rotation == 270 else 'center'
        ax_da.set_ylabel(label, fontsize=fontsize_label, rotation=rotation, va=va)

    # Handle special case for norm_and_raw
    if not plot_da and plot_lick and hasattr(ax_lick, 'figure'):
        fig_axes = ax_lick.figure.get_axes()
        if len(fig_axes) > 1:
            ax_right = [ax for ax in fig_axes if ax != ax_lick][0]
            if norm_lick_left:
                ax_right.set_ylabel(
                    'cumsum(licks to cue)',
                    fontsize=fontsize_label,
                    rotation=270,
                    va='bottom'
                )
            else:
                ax_right.set_ylabel(
                    'norm to num trials',
                    fontsize=fontsize_label,
                    rotation=270,
                    va='bottom'
                )


def _set_y_limits(ax_left, ax_lick, ax_da, plot_lick, plot_da,
                  plot_norm_and_raw, lick_left, ylim_left, ylim_right,
                  lick_cumsum_y, scale_multiplier):
    """Helper function to set y-axis limits."""
    if plot_da and plot_lick:
        if lick_left:
            ax_lick.set_ylim(ylim_left)
            new_right_ylim = twinx_align0(
                ax_lick.get_ylim(),
                ax_da.get_ylim(),
                template_ax_is_left=True,
                expand_neg=True
            )
            ax_da.set_ylim(new_right_ylim)
        else:
            ax_da.set_ylim(ylim_left)
            new_right_ylim = twinx_align0(
                ax_da.get_ylim(),
                ax_lick.get_ylim(),
                template_ax_is_left=True,
                expand_neg=True
            )
            ax_lick.set_ylim(new_right_ylim)
    elif plot_norm_and_raw and plot_lick:
        fig_axes = ax_left.figure.get_axes()
        if len(fig_axes) > 1:
            ax_right = [ax for ax in fig_axes if ax != ax_left][0]
            ax_right.plot(
                ax_left.get_lines()[1].get_xdata(),
                lick_cumsum_y * scale_multiplier,
                linewidth=0
            )
            right_ylim = list(ax_right.get_ylim())
            left_ylim = list(ax_left.get_ylim())

            if any(ylim_right):
                new_left_ylim = change_twinx_together(
                    current_y_left=left_ylim,
                    current_y_right=right_ylim,
                    new_y=ylim_right,
                    new_y_is_left=False
                )
                ax_left.set_ylim(new_left_ylim)
                ax_right.set_ylim(ylim_right)
            elif any(ylim_left):
                new_right_ylim = change_twinx_together(
                    current_y_left=left_ylim,
                    current_y_right=right_ylim,
                    new_y=ylim_left,
                    new_y_is_left=True
                )
                ax_right.set_ylim(new_right_ylim)
                ax_left.set_ylim(ylim_left)
    else:
        ax_left.set_ylim(ylim_left)


def _set_x_ticks(ax, trials_cumsum_x):
    """Helper function to set x-axis ticks based on data range."""
    max_trial = max(trials_cumsum_x)

    tick_configs = [
        (20, 2),
        (50, 10),
        (100, 20),
        (200, 50),
        (500, 100),
        (1000, 200),
        (2000, 500),
    ]


    for threshold, major_tick in tick_configs:
        if max_trial <= threshold:
            ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(major_tick))
            ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
            return

    # Default for values > 2000
    ax.xaxis.set_major_locator(matplotlib.ticker.AutoLocator())
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))




def plotDALickOverTime(trial_df,
                       conditions_to_plot = 'all',
                       colors_for_conditions = defaultdict(lambda: 'black'),
                       plot_cumsum = False,
                       plot_lick = False,
                       plot_DA_cue = False,
                       plot_DA_reward = False,
                       plot_DA_omissions = False,
                       omissions_and_reward_plot_cue_and_lick = 'combine',
                       trials_or_rewards = 'trials',
                       plot_omissions_other_ax = False,
                       lick_left = True,
                       plot_days = False,
                       peak_or_auc: str = 'auc',
                       use_DA_cue_full = False,
                       norm_to_max_individual_rewards =3,
                       norm_to_max_individual_antic = 0,
                       norm_omission_like_cue = False,
                       omission_wind = '2',
                       trial_norm = False,
                       scaled = False,
                       scaled_align_to_1 = False,
                       plot_symbols = False,
                       markersize = 5,
                       markeredgewidth = 0.5,
                       plot_error = True,
                       plot_individuals = False,
                       shaded_error = True,
                       alpha_error = 0.3,
                       linestyle_lick = 'solid',
                       linestyle_DA_cue = (0,(3,1)),
                       linestyle_DA_reward = 'dashdot',
                       linewidth_lick = 0.35,
                       linewidth_DA_cue = 0.35,
                       linewidth_DA_reward = 1,
                       linewidth_individuals = 0.25,
                       elinewidth = 0.25,
                       alpha_individuals = 0.6,
                       linestyle0_lick = (0,(4,2)),
                       linewidth_0_lick = 0,
                       alpha0line = 1,
                       save_figs = False,
                       save_png = False,
                       plot_conditions_separately = False,
                       sharey = False,
                       fig_path ='',
                       title = '',
                       alpha_lick = 1,
                       axsize = (1, 1),
                       colors_for_conditions_DA = None,
                       norm_to_first_day = False,
                       xlim_lick = [None, None],
                       xlim_DA = [None, None],
                       ylim_omissions = [None, None],
                       ylim_lick = [None, None],
                       ylim_DA = [None, None],
                       ticker_multiple_DA = None,
                       ticker_multiple_lick = None,
                       x_MulitpleLocator = None,
                       DA_normalization_precalced = None,
                       return_data = False,
                       ):
    if colors_for_conditions_DA is None:
        colors_for_conditions_DA = colors_for_conditions
    if (not (plot_lick or plot_DA_cue or plot_DA_reward or plot_DA_omissions)):
        raise Exception('Unclear what format data is in, check plot_lick, plot_DA_cue, plot_DA_reward, or plot_DA_omissions flags')
    conditions = lpf.get_conditions_as_list(conditions_to_plot, trial_df)

    trial_df = trial_df[trial_df['condition'].isin(conditions)].copy()
    if plot_days:
        days_or_trials = 'day_num'
        xlabel = 'day'
    else:
        days_or_trials = 'cue_trial_num'
        xlabel = 'trial'

    if use_DA_cue_full:
        DA_time_wind = ''
    else:
        DA_time_wind = '_500ms'
    if plot_symbols:
        symbol = 'o'
    else:
        symbol = ''
    if plot_cumsum:
        DA_to_plot_antic = f'cumsum_antic_dff_{peak_or_auc}_norm{DA_time_wind}'
        DA_to_plot_reward = f'cumsum_consume_dff_{peak_or_auc}_norm{DA_time_wind}'
        DA_to_plot_omissions = f'cumsum_consume_dff_{peak_or_auc}_norm_{omission_wind}s'
        lick_to_plot = 'cumsum_antic_norm'
        ylabel_DA = f'cumsum(DA {peak_or_auc})'
        ylabel_lick = 'cumsum(licks to cue)'
    else:
        DA_to_plot_antic = f'epoch_dff_{peak_or_auc}_antic_norm{DA_time_wind}'
        DA_to_plot_reward = f'epoch_dff_{peak_or_auc}_consume_norm_lickaligned{DA_time_wind}'
        DA_to_plot_omissions = f'epoch_dff_{peak_or_auc}_consume_norm_{omission_wind}s'
        lick_to_plot = 'antic_norm_rate_change'
        ylabel_DA = f'DA ({peak_or_auc})'
        ylabel_lick = 'lick to cue (Hz)'
    xlim_max = 0
    #groupbys unneccesary for plotting trials, but makes code work for plotting trials or days by variable flip
    #for trials technically taking mean of each individual trial
    if plot_DA_omissions:
        trial_df = lpf.cumLickTrialCount(trial_df,
                                         grouping_var = ['animal',
                                                         'trial_type'
                                                         ]
                                         )
        omissions_df = trial_df[trial_df['trial_type'] == 'omission'].copy()
        df_DA_omissions = lpf.group_and_pivot(df = omissions_df,
                                                group_vars = ['condition', 'animal', days_or_trials],
                                                value_col = DA_to_plot_omissions,
                                                )
        if plot_DA_reward:
            reward_df = trial_df[trial_df['trial_type'] == 'reward'].copy()
            df_DA_reward = lpf.group_and_pivot(df = reward_df,
                                                    group_vars = ['condition', 'animal', days_or_trials],
                                                    value_col = DA_to_plot_reward,
                                                    )
            if omissions_and_reward_plot_cue_and_lick == 'combine':
                df_to_use = trial_df

            elif omissions_and_reward_plot_cue_and_lick == 'omission':
                df_to_use = omissions_df

            elif omissions_and_reward_plot_cue_and_lick == 'reward':
                df_to_use = reward_df
            else:
                raise Exception('Unclear which cue and lick to use for omission and reward trials')
        else:
            df_to_use = omissions_df

        df_lick = lpf.group_and_pivot(df = df_to_use,
                                        group_vars = ['condition', 'animal', days_or_trials],
                                        value_col = lick_to_plot,
                                        )
        df_DA_cue = lpf.group_and_pivot(df = df_to_use,
                                        group_vars = ['condition', 'animal', days_or_trials],
                                        value_col = DA_to_plot_antic,
                                        )

            # #TODO: check this, different df than others
            # df_DA_reward = lpf.group_and_pivot(df = trial_df,
            #                                 group_vars = ['condition', 'animal', days_or_trials],
            #                                 value_col = DA_to_plot_reward,
            #                                 )
    else:
        if trials_or_rewards == 'rewards':
            trial_df = lpf.cumLickTrialCount(trial_df,
                                             grouping_var = ['animal',
                                                             'trial_type'
                                                             ]
                                             )
            trials_or_rewards_df = trial_df[trial_df['trial_type'] == 'reward'].copy()
        elif trials_or_rewards == 'trials':
            trials_or_rewards_df = lpf.cumLickTrialCount(trial_df,
                                                         grouping_var = ['animal',
                                                                         'cue_type',
                                                                         ]
                                                         )

        df_lick = lpf.group_and_pivot(df = trials_or_rewards_df,
                                        group_vars = ['condition', 'animal', days_or_trials],
                                        value_col = lick_to_plot,
                                        )
        if plot_DA_cue:

            df_DA_cue = lpf.group_and_pivot(df = trials_or_rewards_df,
                                            group_vars = ['condition', 'animal', days_or_trials],
                                            value_col = DA_to_plot_antic,
                                            )
        if plot_DA_reward:
            df_DA_reward = lpf.group_and_pivot(df = trials_or_rewards_df,
                                            group_vars = ['condition', 'animal', days_or_trials],
                                            value_col = DA_to_plot_reward,
                                            )
    DA_normalization = 1
    if plot_DA_cue or plot_DA_reward or plot_DA_omissions:
        if DA_normalization_precalced is None:
            if (norm_to_max_individual_rewards
                and norm_to_max_individual_antic
                and norm_to_first_day
                ):
                raise Exception(f"two normalizations entered norm_to_max_individual_rewards:{norm_to_max_individual_rewards} and norm_to_max_individual_antic{norm_to_max_individual_antic}")
            elif norm_to_max_individual_rewards:
                max_reward_values = (trial_df
                                     .sort_values([f'epoch_dff_{peak_or_auc}_consume_norm_lickaligned{DA_time_wind}'],
                                                  ascending=False
                                                  )
                                     .groupby(['condition',
                                                 'animal'])
                                     .head(10)
                                     )
                DA_normalization = (max_reward_values
                                    .groupby(['condition','animal'])
                                    [f'epoch_dff_{peak_or_auc}_consume_norm_lickaligned{DA_time_wind}']
                                    .agg(lambda g: g.iloc[0:norm_to_max_individual_rewards].mean())
                                    )

                if plot_DA_omissions:
                    if norm_omission_like_cue:
                        DA_normalization_omission = DA_normalization
                        print('norm ommissions like cue')
                    else:
                        print('norm ommissions by same length')
                        max_reward_values_omission = (trial_df
                                                      .sort_values([f'epoch_dff_{peak_or_auc}_consume_norm_lickaligned_{omission_wind}s'],
                                                                    ascending=False
                                                                    )
                                                      .groupby(['condition',
                                                                'animal'])
                                                      .head(10)
                                                      )
                        DA_normalization_omission = (max_reward_values_omission
                                                     .groupby(['condition',
                                                               'animal'])
                                                     [f'epoch_dff_{peak_or_auc}_consume_norm_lickaligned_{omission_wind}s']
                                                     .agg(lambda g: g.iloc[0:norm_to_max_individual_rewards].mean())
                                                     )
                        #print(DA_normalization_omission)
                ylabel_DA =ylabel_DA + f'\n norm to max {norm_to_max_individual_rewards} reward DA'
                #print(DA_normalization)
            elif norm_to_max_individual_antic:
                max_antic_values = trial_df.sort_values([f'epoch_dff_{peak_or_auc}_antic_norm{DA_time_wind}'],ascending=False).groupby(['condition','animal']).head(10)
                DA_normalization = max_antic_values.groupby(['condition','animal'])[f'epoch_dff_{peak_or_auc}_antic_norm{DA_time_wind}'].agg(lambda g: g.iloc[0:norm_to_max_individual_antic].mean())
                ylabel_DA =ylabel_DA + f'\n norm to max {norm_to_max_individual_antic} cue DA'
            elif norm_to_first_day:
                mean_first_day_reward = (trial_df[trial_df['day_num']==1]
                                         .groupby(['condition',
                                                   'animal'])
                                         [f'epoch_dff_{peak_or_auc}_consume_norm_lickaligned{DA_time_wind}']
                                         .agg(lambda g: g.iloc[0:norm_to_first_day].mean())
                                         )
                DA_normalization = mean_first_day_reward
                ylabel_DA =ylabel_DA + f'\n norm to first {norm_to_first_day} trial reward DA'
            else:
                DA_normalization = 1
        else:
            DA_normalization = DA_normalization_precalced
    # print(DA_normalization)
    df_lick_normed = df_lick
    if plot_DA_cue:
        df_DA_cue_normed = df_DA_cue / DA_normalization
    if plot_DA_reward:
        df_DA_reward_normed = df_DA_reward / DA_normalization
    if plot_DA_omissions:
        df_DA_omissions_normed = df_DA_omissions/DA_normalization_omission

    if plot_cumsum and (scaled or trial_norm): #divide y axis by total trial numbers for scaled cumsum plot
        df_lick_normed = df_lick_normed / df_lick_normed.count()
        if plot_DA_cue:
            df_DA_cue_normed = df_DA_cue_normed / df_DA_cue_normed.count()
        if plot_DA_reward:
            df_DA_reward_normed = df_DA_reward_normed / df_DA_reward_normed.count()
        if plot_DA_omissions:
            df_DA_omissions_normed = df_DA_omissions_normed/df_DA_omissions_normed.count()

    if plot_conditions_separately:
        fig_lick_DA, ax_DA_full = plt.subplots(1, len(conditions), figsize = (2*len(conditions)+2,2), sharey =sharey, squeeze = False)
    else:
        fig_lick_DA, ax_left = plt.subplots(1,1, figsize = (3,2), sharey = sharey, layout= 'constrained')
        if plot_lick and not (plot_DA_cue or plot_DA_reward):
            ax_DA = ax_left
            ax_lick = ax_DA

        elif plot_lick:
            if lick_left:
                ax_lick = ax_left
                ax_DA = ax_lick.twinx()
            else:
                ax_DA = ax_left
                ax_lick = ax_DA.twinx()
        else:
            ax_DA = ax_left
            ax_lick = ax_DA

    if plot_lick:
        title = title + ylabel_lick
    if plot_DA_cue or plot_DA_reward:
        title = title + ylabel_DA
    ax_DA.axhline(0, linestyle = linestyle0_lick, color = 'gray', alpha = alpha0line, linewidth = linewidth_0_lick)
    if return_data:
        da_cue_timecourse = dict.fromkeys(conditions, [])
    for con_idx, condition in enumerate(conditions):
        # if plot_conditions_separately:
        #     ax_DA  = ax_DA_full[0, con_idx]

        # if (plot_DA_cue or plot_DA_reward):
        #     ax_lick = ax_DA.twinx()
        lick_condition = df_lick_normed[condition].copy()
        if plot_DA_cue:
            DA_cue_condition = df_DA_cue_normed[condition].copy()
        if plot_DA_reward:
            DA_reward_condition = df_DA_reward_normed[condition].copy()
        if plot_DA_omissions:
            DA_omission_condition = df_DA_omissions_normed[condition].copy()

        lick_condition_means = lick_condition.mean(axis = 1)
        lick_condition_sems = lick_condition.sem(axis = 1, ddof = 1)
        if plot_DA_cue:
            DA_cue_condition_means = DA_cue_condition.mean(axis = 1)
            DA_cue_condition_sems = DA_cue_condition.sem(axis = 1, ddof = 1)
            if return_data:
                da_cue_timecourse[condition] = DA_cue_condition
        if plot_DA_reward:
            DA_reward_condition_means = DA_reward_condition.mean(axis = 1)
            DA_reward_condition_sems = DA_reward_condition.sem(axis =1, ddof = 1)
        if plot_DA_omissions:
            DA_omission_condition_means = DA_omission_condition.mean(axis = 1)

            DA_omission_condition_sems = DA_omission_condition.sem(axis = 1, ddof = 1)

        if not plot_error:
            lick_condition_sems = None
            if plot_DA_cue:
                DA_cue_condition_sems = None
            if plot_DA_reward:
                DA_reward_condition_sems = None

        if scaled and not plot_days:
            if scaled_align_to_1:
                xlabels_scaled = ['', '', '', '', '',]
                if condition == '30s':
                    xaxis_trials = ((np.arange(len(lick_condition_means))+1)/120)+(1-1/120)
                if condition == '60s':
                    xaxis_trials = ((np.arange(len(lick_condition_means))+1)/60)+(1-1/60)
                if condition == '300s':
                    xaxis_trials = ((np.arange(len(lick_condition_means))+1)/12)+(1-1/12)
                if condition == '600s':
                    xaxis_trials = ((np.arange(len(lick_condition_means))+1)/6)+(1-1/6)
                if condition == '3600s':
                    xaxis_trials = (np.arange(len(lick_condition_means))+1)
            else:
                if condition == '60s':
                    xaxis_trials = (np.arange(len(lick_condition_means))+1)/10
                if condition == '30s':
                    xaxis_trials = (np.arange(len(lick_condition_means))+1)/20
                if condition == '300s':
                    xaxis_trials = (np.arange(len(lick_condition_means))+1)/2
                if ((condition == '600s')
                    or (condition == '60s-CSminus')
                    or (condition == '60s-few')
                    or (condition ==  '60s-few-ctxt')
                    or (condition ==  '600s-bgdmilk')
                    ):

                    xaxis_trials = np.arange(len(lick_condition_means))+1
                if condition == '3600s':
                    xaxis_trials = (np.arange(len(lick_condition_means))+1)*5
                # if condition == 'TypicalITI_50percentOmissions':
                #     xaxis_trials = np.arange(len(lick_condition_means))+1
                #title = ylabel_DA + 'scaled trials'
        else:
            xaxis_trials = np.arange(len(lick_condition_means))+1
        max_trials = max(xaxis_trials)
        if max_trials > xlim_max:
            xlim_max = max_trials
        if plot_lick:
            ax_lick = plot_with_error(ax_lick,
                                        xaxis_trials,
                                        lick_condition_means,
                                        y_sem = lick_condition_sems,
                                        label = f'lick {condition}',
                                        color = colors_for_conditions[condition],
                                        linewidth = linewidth_lick,
                                        elinewidth = elinewidth,
                                        linestyle = linestyle_lick,
                                        marker = symbol,
                                        markersize = markersize,
                                        markeredgewidth = markeredgewidth,
                                        alpha = alpha_lick,
                                        alpha_error = alpha_error,
                                        plot_error = plot_error,
                                        shaded_error = shaded_error,
                                        plot_individuals = plot_individuals,
                                        df_individuals = df_lick_normed[condition].to_numpy(),
                                        linewidth_individuals = linewidth_individuals,
                                        alpha_individuals = alpha_individuals,
                                        )

        if plot_DA_cue:
            ax_DA = plot_with_error(ax_DA,
                                    xaxis_trials,
                                    DA_cue_condition_means,
                                    y_sem = DA_cue_condition_sems,
                                    label = f'Antic DA - {condition}\n n = {len(df_DA_cue_normed[condition].axes[1])}',
                                    color = colors_for_conditions_DA[condition],
                                    linewidth = linewidth_DA_cue,
                                    elinewidth = elinewidth,
                                    linestyle = linestyle_DA_cue,
                                    marker = symbol,
                                    markersize = markersize,
                                    markeredgewidth = markeredgewidth,
                                    alpha = alpha_lick,
                                    alpha_error = alpha_error,
                                    plot_error = plot_error,
                                    shaded_error = shaded_error,
                                    plot_individuals = plot_individuals,
                                    df_individuals = df_DA_cue_normed[condition].to_numpy(),
                                    linewidth_individuals = linewidth_individuals,
                                    alpha_individuals = alpha_individuals,
                                    )

        if plot_DA_reward:
            ax_DA = plot_with_error(ax_DA,
                                    xaxis_trials,
                                    DA_reward_condition_means,
                                    y_sem = DA_reward_condition_sems,
                                    label = f'Reward DA - {condition}\n n = {len(df_DA_reward_normed[condition].axes[1])}',
                                    color = colors_for_conditions_DA[condition],
                                    linewidth = linewidth_DA_reward,
                                    elinewidth = elinewidth,
                                    linestyle = linestyle_DA_reward,
                                    marker = symbol,
                                    markersize = markersize,
                                    markeredgewidth = markeredgewidth,
                                    alpha = alpha_lick,
                                    alpha_error = alpha_error,
                                    plot_error = plot_error,
                                    shaded_error = shaded_error,
                                    plot_individuals = plot_individuals,
                                    df_individuals = df_DA_reward_normed[condition].to_numpy(),
                                    linewidth_individuals = linewidth_individuals,
                                    alpha_individuals = alpha_individuals,
                                    )

        if plot_DA_omissions:
            xaxis_trials_omission = np.arange(len(DA_omission_condition_means))+1
            if plot_omissions_other_ax:
                ax_omissions = ax_DA.twinx()
                ax_omissions.set_ylabel(ylabel_DA + '\n omissions',
                                        rotation = 270,
                                        va = 'bottom'
                                        )
            else:
                ax_omissions = ax_DA

            ax_omissions = plot_with_error(ax_omissions,
                                            xaxis_trials_omission,
                                            DA_omission_condition_means,
                                            y_sem = DA_omission_condition_sems,
                                            label = f'omission DA - {condition}\n n = {len(df_DA_omissions_normed[condition].axes[1])}',
                                            color = colors_for_conditions_DA[condition],
                                            linewidth = linewidth_DA_reward,
                                            elinewidth = elinewidth,
                                            linestyle = linestyle_DA_reward,
                                            marker = symbol,
                                            markersize = markersize,
                                            markeredgewidth = markeredgewidth,
                                            alpha = alpha_lick,
                                            alpha_error = alpha_error,
                                            plot_error = plot_error,
                                            shaded_error = shaded_error,
                                            plot_individuals = plot_individuals,
                                            df_individuals = df_DA_omissions_normed[condition].to_numpy(),
                                            linewidth_individuals = linewidth_individuals,
                                            alpha_individuals = alpha_individuals,
                                            )

    if plot_days:
        ax_DA.set_xlim([0,8.7])
        title = title +' across days'
        ax_DA.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([2, 4, 6, 8]))
        ax_DA.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))

    else:
        if scaled:
            xlabel = '(scaled) ' + xlabel
            if scaled_align_to_1:
                ax_DA.set_xlim([None,7.7])
                ax_DA.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([1, 1+10/6, 1+20/6, 1+30/6, 1+40/6]))
                ax_DA.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
                ax_DA.set_xticklabels(['1\n1\n1\n1\n1',
                                       '2.67\n11\n21\n101\n201',
                                       '4.33\n21\n41\n201\n401',
                                       '6\n31\n61\n301\n601',
                                       '7.67\n41\n81\n401\n801']
                                      )
            else:
                ax_DA.set_xlim([0,40.7])
                ax_DA.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([10, 20, 30, 40]))
                ax_DA.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
                ax_DA.set_xticklabels(['10\n100', '20\n200', '30\n300', '40\n400'])
            title = title + ' across scaled trials'

        else:
            if x_MulitpleLocator is None:
                if max(xaxis_trials) <= 8:
                    ax_DA.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
                    ax_DA.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
                    #ax_cumsum_example_left.set_xticklabels([ '2',  '4', '6', '8'])
                elif max(xaxis_trials) <= 20:
                    ax_DA.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(8))
                    ax_DA.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
                elif max(xaxis_trials) <= 50:
                    ax_DA.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
                    ax_DA.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
                elif max(xaxis_trials) <= 100:
                    ax_DA.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(20))
                    ax_DA.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
                elif max(xaxis_trials) <= 200:
                    ax_DA.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(50))
                    ax_DA.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
                elif max(xaxis_trials) <= 500:
                    ax_DA.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(100))
                    ax_DA.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
                elif max(xaxis_trials) <= 1000:
                    ax_DA.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(200))
                    ax_DA.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
                elif max(xaxis_trials) <= 2000:
                    ax_DA.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(500))
                    ax_DA.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
                else:
                    ax_DA.xaxis.set_major_locator(matplotlib.ticker.AutoLocator())
                    ax_DA.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
            else:
                ax_DA.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(x_MulitpleLocator))
                ax_DA.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))


            ax_DA.set_xlim([None,xlim_max+0.7])

            title = title + f' across {len(df_lick)} trials'

    ax_left.set_xlabel(xlabel)
    if plot_DA_cue or plot_DA_reward:
        ax_DA.set_xlim(xlim_DA)
        ax_DA.set_ylim(ylim_DA)
        if plot_lick and lick_left:
            ax_DA.set_ylabel(ylabel_DA, ha = 'center', va = 'bottom', rotation = 270)
        else:
            ax_DA.set_ylabel(ylabel_DA)
        if ticker_multiple_DA is not None:
            ax_DA.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(ticker_multiple_DA))
        ax_DA.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))


    if plot_lick:
        if not (plot_DA_cue or plot_DA_reward):
            ax_lick.set_ylabel(ylabel_lick, ha = 'center', va = 'bottom')
            if plot_cumsum:
                ax_lick.set_ylim([-0.5, None])
            else:
                ax_lick.set_ylim([-0.75, None])
            ax_lick.set_ylim(ylim_lick)
            ax_lick.set_xlim(xlim_lick)
        else:
            if lick_left:
                ax_lick.set_ylabel(ylabel_lick, ha = 'center', va = 'bottom')
                new_right_ylim = twinx_align0(ax_lick.get_ylim(),
                                                  ax_DA.get_ylim(),
                                                  template_ax_is_left = True,
                                                  expand_neg = True
                                                  )
                ax_DA.set_ylim(new_right_ylim)
            else:
                ax_lick.set_ylabel(ylabel_lick,
                                   ha = 'center',
                                   va = 'bottom',
                                   rotation = 270
                                   )
                new_right_ylim = twinx_align0(ax_DA.get_ylim(),
                                                  ax_lick.get_ylim(),
                                                  template_ax_is_left = True,
                                                  expand_neg = True
                                                  )
                ax_lick.set_ylim(new_right_ylim)
        ax_lick.set_ylim(ylim_lick)
        ax_lick.set_xlim(xlim_lick)
        if ticker_multiple_lick is not None:
            ax_lick.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(ticker_multiple_lick))
        ax_lick.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
    if plot_individuals:
        title = title + ' with individuals'
    if plot_DA_omissions and plot_omissions_other_ax:
        #ax_DA.set_ylim(ylim_DA)
        ax_DA.set_xlim(xlim_DA)
        ax_omissions.set_ylim(ylim_omissions)
        new_omissions_ylim = twinx_align0(ax_DA.get_ylim(),
                                              ax_omissions.get_ylim(),
                                              template_ax_is_left = True,
                                              expand_neg = True
                                              )
        ax_omissions.set_ylim(new_omissions_ylim)
    standardize_plot_graphics(ax_DA)
    title = title + f' conditions -{conditions_to_plot}'
    new_title = lpf.cleanStringForFilename(title)
    set_ax_size_inches(axsize[0], axsize[1], ax_DA )
    if save_figs:
        if save_png:
            fig_lick_DA.savefig(os.path.join(fig_path, f'{new_title}.png'),
                                bbox_inches = 'tight',
                                dpi = 600,
                                transparent = True
                                )
        fig_lick_DA.savefig(os.path.join(fig_path, f'{new_title}.pdf'),
                            bbox_inches = 'tight',
                            transparent = True
                            )
    if plot_DA_cue and plot_DA_omissions:
        return DA_cue_condition_means, DA_omission_condition_means, ax_DA
    elif return_data:
        return da_cue_timecourse
    return fig_lick_DA, ax_DA


def plotDALickOverTimeAligned(trial_df,
                              conditions_to_plot,
                              learned_trials ={},
                              learned_trials_DA ={},
                              colors_for_conditions = defaultdict(lambda: 'black'),
                              colors_for_conditions_DA = None,
                              colors_for_conditions_DA_reward = None,
                              plot_cumsum = True,
                              plot_lick = True,
                              plot_DA_cue = True,
                              plot_DA_reward = False,
                              plot_days = False,
                              peak_or_auc: str = 'auc',
                              use_DA_cue_full = False,
                              norm_to_max_individual_rewards =3,
                              norm_to_max_individual_antic = 0,
                              norm_to_day = 0,
                              scaled = False,
                              plot_symbols = False,
                              plot_error = True,
                              plot_individuals = False,
                              shaded_error = True,
                              alpha_error = 0.3,
                              linestyle_lick = 'solid',
                              linestyle_DA_cue = (0,(3,1)),
                              linestyle_DA_reward = 'dashdot',
                              linewidth_lick = 1,
                              linewidth_DA_cue = 1,
                              linewidth_DA_reward = 1,
                              linewidth_individuals = 0.25,
                              elinewidth = 0.25,
                              alpha_individuals = 0.6,
                              save_figs = False,
                              save_png = False,
                              markersize = 5,
                              plot_conditions_separately = False,
                              sharey = True,
                              fig_path ='',
                              alpha_lick = 1,
                              alpha_DA_cue = 1,
                              alpha_DA_reward = 1,
                              axsize = (1, 1),
                              align_to_lick = True,
                              align_to_DA = False,
                              norm_to_one_average = False,
                              norm_to_one_individual = False,
                              norm_to_first_day = False,
                              ylim_left = [None, None],
                              plot_learned_trial = False):



    learned_trials_dict = {x: learned_trials[x].values() for x in list(learned_trials.keys())}
    DA_learned_trials_dict = {x: learned_trials_DA[x].values() for x in list (learned_trials_DA.keys())}
    if colors_for_conditions_DA is None:
        colors_for_conditions_DA = colors_for_conditions
    if colors_for_conditions_DA_reward is None:
        colors_for_conditions_DA_reward = colors_for_conditions_DA

    # learned_trials_dict_day =
    # DA_learned_trials_dict =

    animals_by_condition = trial_df.groupby(['condition'])['animal'].unique().to_dict()


    title = 'aligned '
    conditions = lpf.get_conditions_as_list(conditions_to_plot, trial_df)
    if plot_days:
        days_or_trials = 'day_num'
        xlabel = 'days from '
    else:
        days_or_trials = 'cue_trial_num'
        xlabel = 'trials from '

    if use_DA_cue_full:
        DA_time_wind = ''
    else:
        DA_time_wind = '_500ms'
    if plot_symbols:
        symbol = 'o'
    else:
        symbol = ''



    if plot_cumsum:
        DA_to_plot_antic = f'cumsum_antic_dff_{peak_or_auc}_norm{DA_time_wind}'
        DA_to_plot_reward = f'cumsum_consume_dff_{peak_or_auc}_norm{DA_time_wind}'
        lick_to_plot = 'cumsum_antic_norm'
        ylabel_DA = f'cumsum(DA {peak_or_auc})'
        ylabel_lick = 'cumsum(licks to cue)'
    else:
        DA_to_plot_antic = f'epoch_dff_{peak_or_auc}_antic_norm{DA_time_wind}'
        DA_to_plot_reward = f'epoch_dff_{peak_or_auc}_consume_norm_lickaligned{DA_time_wind}'
        lick_to_plot = 'antic_norm_rate_change'
        ylabel_DA = f'DA ({peak_or_auc})'
        ylabel_lick = 'lick to cue (Hz)'

    #groupbys unneccesary for plotting trials, but makes code work for plotting trials or days by variable flip
    #for trials technically taking mean of each individual trial
    lick_group = trial_df.groupby(['condition', 'animal', days_or_trials], as_index = False)[lick_to_plot].mean()
    df_lick = lick_group.pivot(index = [days_or_trials], columns = ['condition', 'animal'],
                               values = lick_to_plot)

    DA_cue_group = trial_df.groupby(['condition', 'animal', days_or_trials], as_index = False)[DA_to_plot_antic].mean()
    df_DA_cue = DA_cue_group.pivot(index = [days_or_trials], columns = ['condition', 'animal'],
                                values = DA_to_plot_antic)

    DA_reward_group = trial_df.groupby(['condition', 'animal', days_or_trials], as_index = False)[DA_to_plot_reward].mean()
    df_DA_reward = DA_reward_group.pivot(index = [days_or_trials], columns = ['condition', 'animal'],
                                 values = DA_to_plot_reward)

    if norm_to_max_individual_rewards and norm_to_max_individual_antic and norm_to_first_day:
        raise Exception(f"two normalizations entered norm_to_max_individual_rewards:{norm_to_max_individual_rewards} and norm_to_max_individual_antic{norm_to_max_individual_antic}")
    elif norm_to_max_individual_rewards:
        max_reward_values = trial_df.sort_values([f'epoch_dff_{peak_or_auc}_consume_norm_lickaligned{DA_time_wind}'],ascending=False).groupby(['condition','animal']).head(10)
        DA_normalization = max_reward_values.groupby(['condition','animal'])[f'epoch_dff_{peak_or_auc}_consume_norm_lickaligned{DA_time_wind}'].agg(lambda g: g.iloc[0:norm_to_max_individual_rewards].mean())
        ylabel_DA =ylabel_DA + f'\n norm to max {norm_to_max_individual_rewards} reward DA'
    elif norm_to_max_individual_antic:
        max_antic_values = trial_df.sort_values([f'epoch_dff_{peak_or_auc}_antic_norm{DA_time_wind}'],ascending=False).groupby(['condition','animal']).head(10)
        DA_normalization = max_antic_values.groupby(['condition','animal'])[f'epoch_dff_{peak_or_auc}_antic_norm{DA_time_wind}'].agg(lambda g: g.iloc[0:norm_to_max_individual_antic].mean())
        ylabel_DA =ylabel_DA + f'\n norm to max {norm_to_max_individual_antic} cue DA'

    elif norm_to_day:
        mean_day_reward = trial_df[trial_df['day_num']==norm_to_day].groupby(['condition','animal'])[f'epoch_dff_{peak_or_auc}_consume_norm_lickaligned{DA_time_wind}'].agg(lambda g: g.iloc[-50].mean())
        DA_normalization = mean_day_reward
        ylabel_DA =ylabel_DA + f'\n norm to day {norm_to_first_day} average reward DA'
    else:
        DA_normalization = 1
    df_lick_normed = df_lick
    df_DA_cue_normed = df_DA_cue / DA_normalization
    df_DA_reward_normed = df_DA_reward / DA_normalization


    if scaled and plot_cumsum: #divide y axis by total trial numbers for scaled cumsum plot
        df_lick_normed = df_lick_normed / df_lick_normed.count()
        df_DA_cue_normed = df_DA_cue_normed / df_DA_cue_normed.count()
        df_DA_reward_normed = df_DA_reward_normed / df_DA_reward_normed.count()


    if norm_to_one_individual: #divide by last non-NaN value in each column
        df_lick_normed = df_lick_normed.divide(df_lick_normed.ffill(axis=0).iloc[-1,:], axis =1)
        df_DA_cue_normed = df_DA_cue_normed.divide(df_DA_cue_normed.ffill(axis=0).iloc[-1, :], axis =1)
        df_DA_reward_normed = df_DA_reward_normed.divide(df_DA_reward_normed.ffill(axis=0).iloc[-1,:], axis= 1)



    fig_lick_DA, ax_DA = plt.subplots(1,1, figsize = (1,1), layout= 'constrained')


    if plot_lick and not norm_to_one_individual:
        aligned_lick_ax = ax_DA.twinx()
    elif plot_lick and norm_to_one_individual:
        aligned_lick_ax = ax_DA

    if plot_lick:
        title = title + ylabel_lick
    if plot_DA_cue or plot_DA_reward:
        title = title + ylabel_DA

    cue_DA_aligned =  dict.fromkeys(conditions, [])
    reward_DA_aligned =  dict.fromkeys(conditions, [])
    lick_aligned = dict.fromkeys(conditions, [])
    axis_trials_aligned = dict.fromkeys(conditions, [])




    for condition in conditions:
        if align_to_lick:


            condition_learned_trial_max = max(learned_trials_dict[condition])
            condition_learned_trial_min = min(learned_trials_dict[condition])

        elif align_to_DA:
            condition_learned_trial_max = max(DA_learned_trials_dict[condition])
            condition_learned_trial_min = min(DA_learned_trials_dict[condition])
        condition_trial_learned_dif = condition_learned_trial_max - condition_learned_trial_min
        number_of_columns = max(df_lick_normed[condition].count()) +(condition_trial_learned_dif)

        cue_DA_aligned[condition] = np.empty((len(learned_trials_dict[condition]), number_of_columns))
        reward_DA_aligned[condition] = np.empty((len(learned_trials_dict[condition]), number_of_columns))
        lick_aligned[condition] = np.empty((len(learned_trials_dict[condition]), number_of_columns))
        axis_trials_aligned[condition] = (np.arange(number_of_columns)+1 ) - condition_learned_trial_max
        for a_idx, animal in enumerate(animals_by_condition[condition]):
            if align_to_lick:
                array_offset = condition_learned_trial_max - learned_trials[condition][animal] #pad at beginning
            if align_to_DA:
                array_offset = condition_learned_trial_max - learned_trials_DA[condition][animal] #pad at beginning
            pad_end = condition_trial_learned_dif - array_offset
            # print(animal)
            # print(array_offset)
            # print(pad_end)
            padded_cue_evoked_DA = np.pad(df_DA_cue_normed[condition][animal][df_DA_cue_normed[condition][animal].notnull()].to_numpy(), (array_offset, pad_end) , 'constant', constant_values = (np.nan,))
            padded_reward_evoked_DA = np.pad(df_DA_reward_normed[condition][animal][df_DA_reward_normed[condition][animal].notnull()].to_numpy(), (array_offset, pad_end) , 'constant', constant_values = (np.nan,))
            padded_antic_lick = np.pad(df_lick_normed[condition][animal][df_lick_normed[condition][animal].notnull()].to_numpy(), (array_offset, pad_end) , 'constant', constant_values = (np.nan,))

            cue_DA_aligned[condition][a_idx, :] = padded_cue_evoked_DA
            reward_DA_aligned[condition][a_idx, :] = padded_reward_evoked_DA
            lick_aligned[condition][a_idx, :] = padded_antic_lick



        if plot_days:
            trials_to_chop_end = 0
            trials_to_chop_begin = 0
        elif condition == '60s':

            trials_to_chop_end = -condition_trial_learned_dif-19
            if plot_cumsum:
                trials_to_chop_begin = 30
            else:
                trials_to_chop_begin = 0

        elif condition == '600s':
            trials_to_chop_end = -condition_trial_learned_dif-4
            if plot_cumsum:
                trials_to_chop_begin = 0
            else:
                trials_to_chop_begin = 0
        elif condition == '60s-10%':
            trials_to_chop_end = -condition_trial_learned_dif-500
            if plot_cumsum:
                trials_to_chop_begin = 230
            else:
                trials_to_chop_begin = 0
        cue_DA_aligned[condition] = cue_DA_aligned[condition][:, trials_to_chop_begin:trials_to_chop_end]
        lick_aligned[condition] = lick_aligned[condition][:, trials_to_chop_begin:trials_to_chop_end]

        axis_trials_aligned[condition] = axis_trials_aligned[condition][ trials_to_chop_begin:trials_to_chop_end]
        reward_DA_aligned[condition] = reward_DA_aligned[condition][:, trials_to_chop_begin:trials_to_chop_end]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            means_DA_cue = np.nanmean(cue_DA_aligned[condition], axis = 0)
            sems_DA_cue = stats.sem(cue_DA_aligned[condition], axis = 0, ddof =1, nan_policy = 'omit')
            means_lick = np.nanmean(lick_aligned[condition], axis = 0)
            sems_lick = stats.sem(lick_aligned[condition], axis = 0, ddof =1, nan_policy ='omit')
            means_DA_reward = np.nanmean(reward_DA_aligned[condition], axis =0)
            sems_DA_reward = stats.sem(reward_DA_aligned[condition], axis = 0, ddof =1, nan_policy ='omit')

        if ((condition == '60s') & (scaled)):

            x_aligned = axis_trials_aligned[condition]/10

        else:
            x_aligned = axis_trials_aligned[condition]



        if norm_to_one_average:
            means_DA_cue = means_DA_cue/np.max(means_DA_cue)
            sems_DA_cue = sems_DA_cue/np.max(means_DA_cue)
            means_lick =means_lick/np.max(means_lick)
            sems_lick =sems_lick/np.max(means_lick)

        ax_DA.axhline(0, color = '#808080', linewidth = 1, linestyle = (0,(4,2)))

        if plot_lick:
            if plot_learned_trial:
                learned_trial_params_lick = lpf.getCumsumChangePoint(x_aligned, means_lick, percent_max_dist = 0.75)
                aligned_lick_ax.axvline(0, color = 'k', linewidth = 0.5, linestyle = 'solid')
            aligned_lick_ax.plot(x_aligned, means_lick, linewidth = linewidth_lick, color=  colors_for_conditions[condition], linestyle = linestyle_lick, alpha = alpha_lick)

            if plot_error:
                aligned_lick_ax.fill_between(x_aligned, means_lick - sems_lick, means_lick +  sems_lick, facecolor=  colors_for_conditions[condition], alpha = alpha_error, linewidth = 0)
        if plot_DA_cue:
            if plot_learned_trial:
                learned_trial_params_DA_cue = lpf.getCumsumChangePoint(x_aligned, means_DA_cue, percent_max_dist = 0.75)
                aligned_lick_ax.axvline(learned_trial_params_DA_cue['learned_trial'], color = 'k', linewidth = 0.5, linestyle = 'dashed')
            ax_DA.plot(x_aligned, means_DA_cue, color = colors_for_conditions_DA[condition], linewidth = linewidth_DA_cue, linestyle = linestyle_DA_cue, alpha = alpha_DA_cue)
        if plot_DA_reward:
            ax_DA.plot(x_aligned, means_DA_reward, color = colors_for_conditions_DA_reward[condition], linewidth = linewidth_DA_reward, linestyle = linestyle_DA_reward, alpha = alpha_DA_reward)
        if plot_error:
            if plot_DA_cue:
                ax_DA.fill_between(x_aligned, means_DA_cue - sems_DA_cue, means_DA_cue +  sems_DA_cue,  facecolor=  colors_for_conditions_DA[condition], alpha = alpha_error, linewidth = 0)
            if plot_DA_reward:
                ax_DA.fill_between(x_aligned, means_DA_reward - sems_DA_reward, means_DA_reward +  sems_DA_reward,  facecolor=  colors_for_conditions_DA_reward[condition], alpha = alpha_error, linewidth = 0)


        if plot_individuals:
            if plot_DA_cue:
                ax_DA.plot(x_aligned, cue_DA_aligned[condition].T, color = colors_for_conditions_DA[condition], linestyle = (0,(1,0.5)), linewidth = linewidth_individuals, alpha = alpha_individuals)
            if plot_DA_reward:
                ax_DA.plot(x_aligned, reward_DA_aligned[condition].T, color = colors_for_conditions_DA_reward[condition], linestyle = (0,(2,2)), linewidth = linewidth_individuals, alpha = alpha_individuals)
            if plot_lick:
                aligned_lick_ax.plot(x_aligned, lick_aligned[condition].T, color = colors_for_conditions[condition], linestyle = 'solid', linewidth = linewidth_individuals, alpha = alpha_individuals )



    if plot_cumsum and norm_to_one_individual and plot_lick:
        ax_DA.set_ylabel('cumsum( DA or lick)\n/(norm to max)')
    elif not norm_to_one_individual:
        ax_DA.set_ylabel(ylabel_DA)
        if plot_lick:
            aligned_lick_ax.set_ylabel(ylabel_lick, ha = 'center', va = 'bottom', rotation = 270)
    if align_to_DA:
        xlabel = xlabel + 'DA learning'

    else:
        xlabel = xlabel + 'learning'
    if scaled:
        xlabel = '(scaled) '+ xlabel
    ax_DA.set_xlabel(xlabel)
    ax_DA.set_ylim(ylim_left)
    if plot_lick and not norm_to_one_individual:
        right_ylim = list(aligned_lick_ax.get_ylim())
        left_ylim = list(ax_DA.get_ylim())


        new_lick_ylim = twinx_align0(left_ylim, right_ylim, template_ax_is_left = True, expand_neg = True)

        #ax_cumsum_example_left.set_ylim(new_left_ylim)
        aligned_lick_ax.set_ylim(new_lick_ylim)

    standardize_plot_graphics(ax_DA)
    title = title + f'{condition}'
    new_title = lpf.cleanStringForFilename(title)
    set_ax_size_inches(axsize[0], axsize[1], ax_DA )
    if save_figs:
        if save_png:
            fig_lick_DA.savefig(os.path.join(fig_path, f'{new_title}.png'),
                                dpi = 600,
                                transparent = True,
                                bbox_inches = 'tight')
        fig_lick_DA.savefig(os.path.join(fig_path, f'{new_title}.pdf'),
                            transparent = True,
                            bbox_inches = 'tight')
    return ax_DA


def compare_asymptote_bars(trial_df,
                           conditions_to_plot,
                           range_to_plot,
                           condition_colors,
                           plot_lick_or_DA = 'lick',
                           plot_individuals = True,
                           plot_error = True,
                           plot_stats = True,
                           save_stats = False,
                           ylim =[None,None],
                           bar_alpha = 0.3,
                           markersize = 2.1,
                           plot_sem = True,
                           bar_width = 0.8,
                           ylabel ='',
                           jitter_individuals = True,
                           jitter_width = 20,
                           logscale = False,
                           save_fig = False,
                           save_png = False,
                           fig_path = '',
                           title='',
                           data_is_nested_dict = False,
                           data_is_regular_dict = False,
                           data_is_df = True,
                           maxNLocator = True,
                           norm_to_max_individual_rewards = 3,
                           axsize = (0.64, 1)):

    condition_means = dw.get_mean_values_from_trial_range(trial_df,
                                         conditions_to_subset = conditions_to_plot,
                                         range_to_subset = range_to_plot,
                                         data_to_return = plot_lick_or_DA,
                                         norm_to_max_individual_rewards = norm_to_max_individual_rewards,
                                         )
    conditions = lpf.get_conditions_as_list(conditions_to_plot, trial_df)

    fig_dict = plotBarsFromDict(condition_means,
                         condition_colors = condition_colors,
                         order_to_plot = conditions,
                         plot_individuals = plot_individuals,
                         plot_error = plot_error,
                         plot_stats = plot_stats,
                         save_stats = save_stats,
                         ylim =ylim,
                         bar_alpha = bar_alpha,
                         markersize = markersize,
                         plot_sem = plot_sem,
                         bar_width = bar_width,
                         ylabel =ylabel,
                         jitter_individuals = jitter_individuals,
                         jitter_width = jitter_width,
                         logscale = logscale,
                         save_fig = save_fig,
                         save_png = save_png,
                         title='',
                         data_is_nested_dict = False,
                         data_is_df = True,
                         fig_path = fig_path,
                         maxNLocator = maxNLocator,
                         axsize = axsize)
    return fig_dict['fig'], fig_dict['ax'], #fig_dict['jitter_x_dict'], fig_dict['data_conditions']

def compare_prob_antic_lick_bars(trial_df,
                           conditions_to_plot,
                           range_to_plot,
                           condition_colors,
                           antic_lick_threshold = 2,
                           plot_individuals = True,
                           plot_error = True,
                           plot_stats = True,
                           save_stats = False,
                           ylim =[None,None],
                           bar_alpha = 0.3,
                           markersize = 2.1,
                           plot_sem = True,
                           bar_width = 0.8,
                           ylabel ='',
                           jitter_individuals = True,
                           jitter_width = 20,
                           logscale = False,
                           save_fig = False,
                           save_png = False,
                           fig_path = '',
                           title='',
                           maxNLocator = True,
                           axsize = (0.64, 1),
                           ):
    conditions = lpf.get_conditions_as_list(conditions_to_plot, trial_df)
    subset_df = trial_df[trial_df['condition'].isin(conditions)].copy()
    subset_df['antic_lick_prob'] = [1
                                    if x >= antic_lick_threshold
                                    else 0
                                    for
                                    x in subset_df['nlicks_antic_norm']
                                    ]
    condition_means = dw.get_mean_values_from_trial_range(subset_df,
                                                        conditions_to_subset = conditions_to_plot,
                                                        range_to_subset = range_to_plot,
                                                        data_to_return = 'antic_lick_prob',
                                                        )
    ranges_list = [range_to_plot[x][1] - range_to_plot[x][0] +1
                   for x
                   in range_to_plot
                   ]
    ylabel_new = f'prob. of > {antic_lick_threshold-1} licks to cue\n last {ranges_list} trials' + f'\n{ylabel}'
    fig_dict = plotBarsFromDict(condition_means,
                         condition_colors = condition_colors,
                         order_to_plot = conditions,
                         plot_individuals = plot_individuals,
                         plot_error = plot_error,
                         plot_stats = plot_stats,
                         save_stats = save_stats,
                         ylim =ylim,
                         bar_alpha = bar_alpha,
                         markersize = markersize,
                         plot_sem = plot_sem,
                         bar_width = bar_width,
                         ylabel =ylabel_new,
                         jitter_individuals = jitter_individuals,
                         jitter_width = jitter_width,
                         logscale = logscale,
                         save_fig = save_fig,
                         save_png = save_png,
                         title='',
                         data_is_nested_dict = False,
                         data_is_df = True,
                         fig_path = fig_path,
                         maxNLocator = maxNLocator,
                         axsize = axsize)
    return {'fig':fig_dict['fig'], 'ax':fig_dict['ax'], 'data':condition_means}

def plot_IRI_vs_learned_trial_scatter(learned_trial_dict,
                                      colors_for_conditions = defaultdict(lambda: 'black') ,
                                      nested_dict = True,
                                      conditions_to_plot = 'all',
                                      conditions_for_fitline = 'all',
                                      plot_fig = True,
                                      plot_line = True,
                                      marker = 'o',
                                      alpha = 1,
                                      markeredgewidth = 0,
                                      open_marker = False,
                                      linewidth_fitline = 1,
                                      linestyle_fitline = 'solid',
                                      error = 'std',
                                      title = '',
                                      xlim = [None, None],
                                      ylim = [None, 300],
                                      axsize = (1,1),
                                      ax_to_plot = None,
                                      save_fig = False,
                                      save_png = False,
                                      fig_path = ''):
    trial_posttrial_duration = 4.25
    condition_ITIs = {'30s': 30,
                      '30 s ITI': 30,
                      '60s': 60,
                      '60 s ITI': 60,
                      '300s': 300,
                      '300 s ITI': 300,
                      '600s': 600,
                      '600 s ITI': 600,
                      '3600s': 3600,
                      '3600 s ITI': 3600,
                      '600s-bgdmilk': 600,
                      '60s-50%': 124.25,
                      'ThirtySec_100trials': 30,

                                        'SixtySec_50trials': 60,

                                        'FiveMin_11trials': 300,

                                        'TenMin_6trials': 600,

                                        'OneHour_2trials': 3600,

                                        'TenMin_6trials_bgdMilk': 600,
                                        'TypicalITI_50percentOmissions': 124.25}
    condition_IRIs = {cond: iti + trial_posttrial_duration
                      for (cond, iti)
                      in condition_ITIs.items()
                      }

    conditions = lpf.get_conditions_as_list(conditions_to_plot, learned_trial_dict)
    if conditions_for_fitline == 'all':
        conditions_for_fitline = conditions
    if nested_dict:
        learned_trial_by_condition  = {x: [y
                                           for y
                                           in learned_trial_dict[x].values()
                                           if type(y) is not list
                                           ]
                                       for x
                                       in learned_trial_dict
                                       }
    else:
        learned_trial_by_condition = learned_trial_dict
    trials_to_learn_by_IRI_mean = {x: np.mean(learned_trial_by_condition[x])
                                   for x
                                   in conditions
                                   }
    trials_to_learn_by_IRI_std = {x: np.std(learned_trial_by_condition[x], ddof = 1)
                                  for x
                                  in conditions
                                  }
    trials_to_learn_by_IRI_sem = {x: stats.sem(learned_trial_by_condition[x], ddof = 1)
                                  for x
                                  in conditions
                                  }
    IRIs_log = [np.log10(condition_IRIs[x])
                for x
                in conditions
                if x in conditions_for_fitline
                ]
    trials_to_learn_log = [np.log10(trials_to_learn_by_IRI_mean[y])
                           for y
                           in conditions
                           if y in conditions_for_fitline
                           ]
    fit_line = stats.linregress(IRIs_log, trials_to_learn_log)

    if plot_fig:
        if error == 'std':
            yerr = trials_to_learn_by_IRI_std
        elif error == 'sem':
            yerr = trials_to_learn_by_IRI_sem
        elif error is None:
            yerr = {x: None for x in conditions}
        if ax_to_plot is None:
            fig, ax = plt.subplots()
        else:
            ax = ax_to_plot
            fig = ax_to_plot.get_figure()


        for condition in conditions:
            if open_marker:
                markerfacecolor = 'white'#'None' # (1, 1, 1, 1)
                markeredgecolor = colors_for_conditions[condition]
                markersize = 5
            else:
                markerfacecolor = colors_for_conditions[condition]
                markeredgecolor = 'None'
                if marker == 's' or marker == 'D':
                    markersize = 5
                else:
                    markersize = 6
            ax.errorbar(condition_IRIs[condition],
                        trials_to_learn_by_IRI_mean[condition],
                        yerr = yerr[condition],
                        ecolor = colors_for_conditions[condition],
                        markeredgewidth = markeredgewidth,
                        linewidth = 0,
                        elinewidth = 0.5,
                        marker = marker,
                        markersize = markersize,
                        markerfacecolor = markerfacecolor,
                        markeredgecolor=markeredgecolor,
                        alpha = alpha)



        ax.set_xscale("log", base=10)
        ax.set_yscale("log", base=10)
        ax.set_ylabel('trials to learn')
        ax.set_xlabel('inter reward interval (s)')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(title)

        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        if plot_line:
            axlim = np.arange(ax.get_xlim()[0], ax.get_xlim()[1], 1 )
            ax.plot(axlim,
                    np.power(10, fit_line.intercept)*np.power(axlim, fit_line.slope),
                    color = 'k',
                    linewidth = linewidth_fitline,
                    linestyle = linestyle_fitline,
                    alpha = alpha)
        standardize_plot_graphics(ax)
        set_ax_size_inches(axsize[0], axsize[1], ax)
        if save_fig:
            if save_png:
                fig.savefig(os.path.join(fig_path,
                                         'linearLearning_learningRate_linefitwo3600.png'),
                            dpi = 600,
                            bbox_inches = 'tight',
                            transparent = True,
                            )
            fig.savefig(os.path.join(fig_path,
                                     f'{title}-IRIvsLearnedTrialScatter_{*conditions,}.pdf'),
                        bbox_inches = 'tight',
                        transparent = True,
                        )

        return {'fig': fig, 'ax':ax, 'fit_line': fit_line}
    else:
        return {'fit_line': fit_line}
def plot_model_sweep_timecourses(ITI_means_dict,
                                 colors_for_conditions = defaultdict(lambda: 'black'),
                                 conditions_to_plot = 'all',
                                 linewidth = 1,
                                 alpha_error = 0.3,
                                 scaled = True,
                                 scaled_align_to_1 = True,
                                 ax_to_plot = None,
                                 fontsize_title = 7,
                                 fontsize_label = 7,
                                 fontsize_ticks = 6,
                                 linestyle = 'solid',
                                 linewidth_individuals = 0.25,
                                 alpha_individuals = 0.6,
                                 linestyle_0 = (0,(4,2)),
                                 linewidth_0 = 1,
                                 alpha0line = 1,
                                 xlim = [None, None],
                                 ylim = [None, None],
                                 title = '',
                                 ylabel = '',
                                 axsize = (1, 1),
                                 fig_path = '',
                                 save_fig = False,
                                 save_png = False,
                                 ):

    conditions = lpf.get_conditions_as_list(conditions_to_plot, ITI_means_dict)

    if ax_to_plot is None:
        fig, ax = plt.subplots()
    else:
        ax = ax_to_plot
        fig = ax_to_plot.get_figure()
    ax.axhline(0, color = 'gray',
               linewidth = linewidth_0,
               linestyle = linestyle_0,
               alpha = alpha0line)

    for key, data in ITI_means_dict.items():
        if key == '30 s ITI':
            if scaled:
                if scaled_align_to_1:
                    xaxis_trials = (data['trials'][data['trials']<=800]/120)+(1-1/120)
                else:
                    xaxis_trials = (data['trials'][data['trials']<=800]/120)
                iti_mean = data['mean'][data['trials']<=800]
                iti_sem = data['sem'][data['trials']<=800]
            else:
                xaxis_trials = data['trials']
                iti_mean = data['mean']
                iti_sem = data['sem']

        if key == '60 s ITI':
            if scaled:
                if scaled_align_to_1:
                    xaxis_trials = (data['trials'][data['trials']<=400]/60)+(1-1/60)
                else:
                    xaxis_trials = (data['trials'][data['trials']<=400]/60)
                iti_mean = data['mean'][data['trials']<=400]
                iti_sem = data['sem'][data['trials']<=400]
            else:
                xaxis_trials = data['trials']
                iti_mean = data['mean']
                iti_sem = data['sem']

        if key == '300 s ITI':
            if scaled:
                if scaled_align_to_1:
                    xaxis_trials = (data['trials'][data['trials']<=80]/12)+(1-1/12)
                else:
                    xaxis_trials = (data['trials'][data['trials']<=80]/12)
                iti_mean = data['mean'][data['trials']<=80]
                iti_sem = data['sem'][data['trials']<=80]
            else:
                xaxis_trials = data['trials']
                iti_mean = data['mean']
                iti_sem = data['sem']

        if key == '600 s ITI':
            if scaled:
                if scaled_align_to_1:
                    xaxis_trials = (data['trials'][data['trials']<=40]/6)+(1-1/6)
                else:
                    xaxis_trials = (data['trials'][data['trials']<=40]/6)
                iti_mean = data['mean'][data['trials']<=40]
                iti_sem = data['sem'][data['trials']<=40]
            else:
                xaxis_trials = data['trials']
                iti_mean = data['mean']
                iti_sem = data['sem']

        if key == '3600 s ITI':
            if scaled:
                xaxis_trials = data['trials'][data['trials']<=8]
                iti_mean = data['mean'][data['trials']<=8]
                iti_sem = data['sem'][data['trials']<=8]
            else:
                xaxis_trials = data['trials']
                iti_mean = data['mean']
                iti_sem = data['sem']
        ax.plot(xaxis_trials, iti_mean, color = colors_for_conditions[key], linewidth = linewidth)
        ax.fill_between(xaxis_trials,
                                  iti_mean - iti_sem,
                                  iti_mean + iti_sem,
                                  linewidth = 0,
                                  facecolor = colors_for_conditions[key],
                                  alpha = alpha_error,)

    if scaled:
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([1, 1+10/6, 1+20/6, 1+30/6, 1+40/6]))
        ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
        ax.set_xticklabels(['1\n1\n1\n1\n1',
                            '2.67\n11\n21\n101\n201',
                            '4.33\n21\n41\n201\n401',
                            '6\n31\n61\n301\n601',
                            '7.67\n41\n81\n401\n801',])
        ax.set_xlim(xlim)
    else:
        if ax.get_xlim()[1] > 2000:
            ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(600))
            ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
        else:
            ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(200))
            ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
    if ax.get_ylim()[1] <0.2:
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    if ax.get_ylim()[1] <0.4:
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))
    elif ax.get_ylim()[1] <=2:
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    else:
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(2))
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
    ax.set_xlabel('trial')
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    ax.set_title(title)
    standardize_plot_graphics(ax)
    set_ax_size_inches(axsize[0], axsize[1], ax)
    new_title = title + '_'+f'{ITI_means_dict.keys()}'
    new_title = lpf.cleanStringForFilename(new_title)
    if save_fig:
        if save_png:
            fig.savefig(os.path.join(fig_path, f'{new_title}.png'),
                                bbox_inches = 'tight',
                                dpi = 600,
                                transparent = True
                                )
        fig.savefig(os.path.join(fig_path, f'{new_title}.pdf'),
                            bbox_inches = 'tight',
                            transparent = True
                            )
    return fig, ax


def plot_omission_heatmap_raster(trial_df,
                                 animal,
                                 axsize = (0.5, 2),
                                 plot_in_sec = True,
                                 align_to_cue = True,
                                 show_yticklabels = True,
                                 stroke_cue_reward_vertical = 0.25,
                                 color_cue_shade = '#939598',
                                 alpha_cue_shade = 0.5,
                                 stroke_raster = 0.25,
                                 fontsize_title = 7,
                                 fontsize_label = 7,
                                 fontsize_ticks = 6,
                                 xlim_in_sec = [-2.5, 5],
                                 min_max_heatmap = [-13, 11.5],
                                 color_map = plt.cm.PiYG,
                                 save_fig = False,
                                 save_png = False,
                                 fig_path = '',):
    trials = ['omission', 'reward']
    #check if behavior data in s or ms

    data_in_s = True if np.max(trial_df['cue_dur'])<49 else False

    same_unit = plot_in_sec == data_in_s
    #scale between mc and s depending on data format and desired output
    same_unit = plot_in_sec == data_in_s
    scale_factor = 1 if same_unit else (1/1000 if plot_in_sec else 1000)

    lick_DA_psth_fig, lick_DA_psth_ax = plt.subplots(1,4,
                                                     sharey = False,
                                                     sharex = True,
                                                     )
    for ax, trial in enumerate(trials):
        # lick_DA_psth_ax[ax*2+1].get_shared_y_axes().join(lick_DA_psth_ax[ax*2],
        #                                                  lick_DA_psth_ax[ax*2+1],
        #                                                  )

        lick_DA_psth_ax[ax*2+1].sharey(lick_DA_psth_ax[ax*2])
        if not show_yticklabels:
            lick_DA_psth_ax[ax*2+1].set_yticklabels([])
        animal_day_df = trial_df[trial_df['animal'] == animal].copy()
        animal_day_df = animal_day_df[animal_day_df['trial_type'] == trial].copy()
        reward_time = (((animal_day_df['reward_time']
                       - animal_day_df['cue_on'] )
                        .mean())
                       * scale_factor)
        if align_to_cue:
            cue_on =  0
            cue_off = (animal_day_df['cue_off'] - animal_day_df['cue_on'] ).mean() * scale_factor
            reward_time = (animal_day_df['reward_time'] - animal_day_df['cue_on'] ).mean() * scale_factor
            lick_DA_psth_fig.supxlabel('time from cue onset (s)',
                                       fontsize = fontsize_label)
        else:
            cue_on = (animal_day_df['cue_on'] - animal_day_df['reward_time'] ).mean() * scale_factor
            cue_off = (animal_day_df['cue_off'] - animal_day_df['reward_time'] ).mean() * scale_factor
            reward_time = 0
            lick_DA_psth_fig.supxlabel('time from reward delivery (s)',
                                       fontsize = fontsize_label)
        lick_times_list = (animal_day_df['licks_all'] * scale_factor).tolist()
        raster_ylen = len(lick_times_list)
        if align_to_cue:
            lick_times_list = [trial_licks + reward_time for trial_licks in lick_times_list]
        epoch_dff_times = np.mean(np.array(animal_day_df['epoch_time'].to_list()), axis = 0) * scale_factor


        epoch_dff_times = epoch_dff_times + reward_time
        epoch_dff = np.array(animal_day_df['epoch_dff'].to_list())
        #epoch_dff = [x - np.mean(x[((epoch_dff_times >= xlim_in_sec[0]) & (epoch_dff_times <0))]) for x in epoch_dff ]
        epoch_dff = [x[((epoch_dff_times >= xlim_in_sec[0]) & (epoch_dff_times <=xlim_in_sec[1]))] for x in epoch_dff ]

        epoch_dff_times =epoch_dff_times[((epoch_dff_times >= xlim_in_sec[0]) & (epoch_dff_times <=xlim_in_sec[1]))]
        epoch_dff_mean = np.mean(epoch_dff, axis = 0)
        epoch_dff_sem = stats.sem(epoch_dff, axis = 0)


        lick_DA_psth_ax[ax*2].axvline(x = reward_time, color ='gray', linestyle='dashed', linewidth =  stroke_cue_reward_vertical )
        lick_DA_psth_ax[ax*2].axvspan(cue_on, cue_off, alpha = alpha_cue_shade, facecolor = color_cue_shade, linewidth = None)

        lick_DA_psth_ax[ax*2].eventplot(lick_times_list, linewidths = stroke_raster, linelengths = 0.75, colors= 'black', lineoffsets = np.arange(len(lick_times_list))+1)
        lick_DA_psth_ax[ax*2].set_ylim([raster_ylen + 0.5, 0.5])
        lick_DA_psth_ax[0].set_ylabel('trial', fontsize = fontsize_label)
        #lick_DA_psth_ax[ax*2].yaxis.set_major_locator(matplotlib.ticker.LinearLocator(2))

        lick_DA_psth_ax[ax*2].yaxis.set_major_locator(matplotlib.ticker.FixedLocator([1, len(lick_times_list)]))
        lick_DA_psth_ax[ax*2].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))

        lick_DA_psth_ax[ax*2+1].axvline(x=cue_on, color='gray', linestyle='dashed', linewidth = stroke_cue_reward_vertical )
        lick_DA_psth_ax[ax*2+1].axvline(x=cue_off, color='gray', linestyle='dashed',  linewidth = stroke_cue_reward_vertical  )
        lick_DA_psth_ax[ax*2+1].axvline(x = reward_time, color ='gray', linestyle='dashed', linewidth =  stroke_cue_reward_vertical )
        lick_DA_psth_ax[ax*2].set_xlim(xlim_in_sec)
        lick_DA_psth_ax[ax*2+1].set_xlim(xlim_in_sec)
        colorplotmax = np.max(epoch_dff_mean)


        dff_heatmap = lick_DA_psth_ax[ax*2+1].imshow(epoch_dff, cmap=color_map , interpolation='none', aspect="auto",
                                   extent=[epoch_dff_times[0], epoch_dff_times[-1], len(epoch_dff), 0],
                                   origin = 'upper', vmax = min_max_heatmap[1], vmin = min_max_heatmap[0])
        cbar_ax = lick_DA_psth_ax[-1].inset_axes([1.02, 0, .05, 1])
        cbar = plt.colorbar(dff_heatmap, cax = cbar_ax, pad= 0, fraction=0.04)# label = '% dF/F')
        cbar.set_label('% dF/F', rotation = 270, va = 'bottom')
        lick_DA_psth_ax[ax*2+1].set_title(trial)
        lick_DA_psth_ax[ax*2+1].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(len(epoch_dff)))
        lick_DA_psth_ax[ax*2+1].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))

    standardize_plot_graphics(lick_DA_psth_ax)
    set_ax_size_inches(axsize[0], axsize[1], lick_DA_psth_ax)
    if save_fig:
        if save_png:
            lick_DA_psth_fig.savefig(os.path.join(fig_path,
                                                 f'example_omission_heatmap_raster_{animal}.png'
                                                 ),
                                    dpi = 600,
                                    transparent = True,
                                    bbox_inches = 'tight'
                                    )
        lick_DA_psth_fig.savefig(os.path.join(fig_path,
                                             f'example_omission_heatmap_raster_{animal}.pdf',
                                             ),
                                transparent = True,
                                bbox_inches = 'tight'
                                )
    return lick_DA_psth_ax

def plotProbabilityOfAnticLick(trial_df,
                               conditions_to_plot = 'all',
                               antic_lick_threshold = 2,
                               subtract_neg = False,
                               scaled = False,
                               ax = None,
                               colors:dict = None,
                               linewidth_0_lick = 0,
                               save_fig = False,
                               save_png = False,
                               fig_path = '',
                               axsize = (1, 1),
                               ):
    fig_probaanticlick, ax_probaanticlick = plt.subplots(figsize = (4,3))
    ax_probaanticlick.axhline(0, linestyle = 'dashed', color = 'gray', linewidth = linewidth_0_lick)
    trial_all_data = trial_df.copy()
    condition_colors = colors
    if colors == None:
        condition_colors = dict.fromkeys(conditions_to_plot, 'blue')
    if subtract_neg:

        trial_all_data['antic_lick_prob'] = [1
                                             if x >= antic_lick_threshold
                                             else -1
                                             if x <= -antic_lick_threshold
                                             else 0
                                             for x
                                             in trial_all_data['nlicks_antic_norm']
                                             ]
    else:
        trial_all_data['antic_lick_prob'] = [1
                                             if x >= antic_lick_threshold
                                             else 0
                                             for x
                                             in trial_all_data['nlicks_antic_norm']
                                             ]
    max_len_xaxis = 0
    for condition in conditions_to_plot:

        condition_df = trial_all_data[trial_all_data['condition'] == condition]
        #ax_probaanticlick = condition_df.groupby(['cue_trial_num'])['antic_lick_prob'].mean().T.plot(yerr=condition_df.groupby(['cue_trial_num'])['antic_lick_prob'].sem(), rot=0, label = condition)# width = 0.7)

        y = condition_df.groupby(['cue_trial_num'])['antic_lick_prob'].mean()
        yerr = condition_df.groupby(['cue_trial_num'])['antic_lick_prob'].sem()
        if ((condition == '60s') & (scaled)):

            xaxis = (np.arange(len(condition_df.groupby(['cue_trial_num'])))/10)+1
        else:
            xaxis = (np.arange(len(condition_df.groupby(['cue_trial_num']))))+1
        if len(xaxis) > max_len_xaxis:
            max_len_xaxis = len(xaxis)
        ax_probaanticlick.plot(xaxis,
                               y,
                               color = condition_colors[condition],
                               label = condition,
                               linewidth = 0.35
                               )
        ax_probaanticlick.fill_between(xaxis,
                                       y-yerr,
                                       y+yerr,
                                       facecolor = condition_colors[condition],
                                       alpha = 0.3,
                                       linewidth = 0,
                                       )


    ylabel = f'prop. of mice > {antic_lick_threshold-1} licks to cue'

    ax_probaanticlick.set_ylabel(ylabel)
    ax_probaanticlick.set_xlabel('trial')
    if scaled:
        ax_probaanticlick.set_xlim([0,40.7])
        ax_probaanticlick.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([10, 20, 30, 40]))
        ax_probaanticlick.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
        ax_probaanticlick.set_xticklabels(['10\n100', '20\n200', '30\n300', '40\n400'])
        ylabel = ylabel + ' across scaled trials'

    else:
        ax_probaanticlick.set_xlim([0,max_len_xaxis+0.7])
        ax_probaanticlick.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([max_len_xaxis/4,
                                                                                  max_len_xaxis*(2/4),
                                                                                  max_len_xaxis*(3/4),
                                                                                  max_len_xaxis]))
        ax_probaanticlick.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
        ylabel = ylabel + f' across {max_len_xaxis} trials'
    ax_probaanticlick.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0, 0.5, 1]))
    ax_probaanticlick.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
    standardize_plot_graphics(ax_probaanticlick)
    #plt.legend(loc = 4)
    new_title = lpf.cleanStringForFilename(ylabel)
    set_ax_size_inches(axsize[0], axsize[1], ax_probaanticlick )
    if save_fig:
        if save_png:
            fig_probaanticlick.savefig(os.path.join(fig_path, f'{new_title}.png'),
                                       dpi = 600,
                                       bbox_inches = 'tight',
                                       transparent = True,
                                       )
        fig_probaanticlick.savefig(os.path.join(fig_path, f'{new_title}.pdf'),
                                   bbox_inches = 'tight',
                                   transparent = True,
                                   )
    return trial_all_data

def plotFearPSTHbyDay(trials_df,
                      animal,
                      days_to_plot,
                      plot_motion = True,
                      plot_motion_as_percent = True,
                      plot_lick_raster = True,
                      plot_lick_PSTH = True,
                      plot_DA_heatmap = True,
                      plot_DA_PSTH = True,
                      align_to_cue = True,
                      plot_in_sec = True,
                      xlim_in_sec = [-15, 25],
                      ylim_lick = [-1, 12],
                      ylim_DA = [None, None],
                      dff_to_plot ='%',
                      axsize = (6, 2),
                      sharey = 'row',
                      stroke_raster = 0.35,
                      stroke_PSTH = 0.5,
                      color_DA_PSTH = 'dodgerblue',
                      color_lick_PSTH = 'green',
                      alpha_PSTH_error = 0.3,
                      color_cue_shade = '#939598',
                      alpha_cue_shade = 0.5,
                      fontsize_title = 14,
                      fontsize_label = 14,
                      fontsize_ticks = 12,
                      smooth_lick = 0.75,
                      stroke_cue_reward_vertical = 0.25,
                      linestyle_lick = 'solid',
                      linestyle_DA = 'dashed',
                      fig_path = '',
                      save_fig = False,
                      save_png = False,
                      linewidth_0_lick = 0,
                      linewidth_0_DA = 0.35,
                      alpha0line = 0.5,
                      norm_to_max_rewards = 0,
                      ):

    if plot_in_sec:
        scale_factor = 1/1000
        #x_limit_in_sec * scale_factor
    else:
        scale_factor = 1

    if isinstance(days_to_plot, int):
        days = [days_to_plot]
    else:

        days = np.arange(days_to_plot[0], days_to_plot[1]+1)
    total_time_to_plot_ms = (xlim_in_sec[1] - xlim_in_sec[0]) *1000
    baseline_time_to_plot_ms = xlim_in_sec[0] * (-1000)
    #determines parameters and axes for figure
    total_days = len(days)
    total_rows =  (plot_motion+plot_DA_heatmap) *2
    fig_width = total_days * axsize[0] + 2
    fig_height = total_rows * axsize[1] + 2# pad each side with an inch


    motion_DA_psth_fig, motion_DA_psth_ax = plt.subplots(total_rows,
                                                     total_days,
                                                     figsize = (fig_width, fig_height),
                                                     sharex = True,
                                                     sharey = sharey,
                                                     squeeze = False)# layout = 'constrained')#, subplotpars={ 'left': one_inch_width, 'top' : one_inch_height, 'hspace': 0.1, 'wspace': 0.1}) #, constrained_layout = True)


    if plot_motion:
        motion_heatmap_row = 0
        motion_PSTH_row = 1
        DA_heatmap_row = 2
        DA_PSTH_row = 3
    else:
        DA_heatmap_row = 0
        DA_PSTH_row = 1



    #determine max trials per day for raster plot ylim purposes
    raster_ylen =  len(trials_df[((trials_df['animal'] == animal)
                                  & (trials_df['day_num'] >= days[0])
                                 & (trials_df['day_num'] <= days[-1]))]['trial_num'].unique())


    for d_idx, day in enumerate(days):
        animal_day_df = trials_df[((trials_df['animal'] == animal) & (trials_df['day_num'] == day))]
        if not animal_day_df.empty:
            condition = animal_day_df['condition'].iloc[0]
            if align_to_cue:
                cue_on =  0
                cue_off = (animal_day_df['cue_off'] - animal_day_df['cue_on'] ).mean() * scale_factor
                shock_on = (animal_day_df['shock_on'] - animal_day_df['cue_on'] ).mean() * scale_factor
                shock_off = (animal_day_df['shock_off'] - animal_day_df['cue_on'] ).mean() * scale_factor

            else:
                cue_on = (animal_day_df['cue_on'] - animal_day_df['shock_on'] ).mean() * scale_factor
                cue_off = (animal_day_df['cue_off'] - animal_day_df['shock_on'] ).mean() * scale_factor
                shock_on = 0
                shock_off = (animal_day_df['shock_off'] - animal_day_df['shock_on'] ).mean() * scale_factor



            epoch_dff_times = np.mean(np.array(animal_day_df['epoch_time'].to_list()), axis = 0) * scale_factor

            if not align_to_cue:
                epoch_dff_times = epoch_dff_times - shock_on

            if dff_to_plot =='%':
                epoch_dff = np.array(animal_day_df['epoch_dff'].to_list())
                dlight_ylabel = 'dLight\n(% dF/F)'
            elif dff_to_plot == 'norm_to_3_peak':
                epoch_dff = np.array(animal_day_df['epoch_dff_normed_to_1'].to_list())
                dlight_ylabel = 'dLight\n(norm to max peak)'
            epoch_motion_times = np.mean(np.array(animal_day_df['epoch_motion_time'].to_list()), axis = 0)
            epoch_motion = np.array(animal_day_df['epoch_motion'].to_list())
            epoch_dff_mean = np.mean(epoch_dff, axis = 0)
            epoch_dff_sem = stats.sem(epoch_dff,
                                      ddof = 1,
                                      axis = 0,
                                      nan_policy = 'propagate',
                                      )
            epoch_motion_mean = np.mean(epoch_motion, axis = 0)
            epoch_motion_sem = stats.sem(epoch_motion,
                                      ddof = 1,
                                      axis = 0,
                                      nan_policy = 'propagate',
                                      )
            if plot_motion:


                motion_DA_psth_ax[motion_heatmap_row, d_idx].axvline(x=cue_on,
                                                               color='green',
                                                               linestyle='dashed',
                                                               linewidth = stroke_cue_reward_vertical,
                                                               )
                motion_DA_psth_ax[motion_heatmap_row, d_idx].axvline(x=cue_off,
                                                                color='green',
                                                                linestyle='dashed',
                                                                linewidth = stroke_cue_reward_vertical,
                                                                )
                motion_DA_psth_ax[motion_heatmap_row, d_idx].axvline(x = shock_on,
                                                                color ='red',
                                                                linestyle='dashed',
                                                                linewidth =  stroke_cue_reward_vertical,
                                                                )
                motion_DA_psth_ax[motion_heatmap_row, d_idx].axvline(x = shock_off,
                                                                color ='red',
                                                                linestyle='dashed',
                                                                linewidth =  stroke_cue_reward_vertical,
                                                                )
                if d_idx == 0:
                    colorplotmax_motion = np.max(epoch_motion_mean)
                    motion_DA_psth_ax[motion_heatmap_row, 0].set_ylabel('trial (motion)',
                                                                  fontsize = fontsize_label,
                                                                  )
                dff_motion_heatmap = motion_DA_psth_ax[motion_heatmap_row, d_idx].imshow(epoch_motion,
                                                                            cmap=plt.cm.binary,
                                                                            interpolation='none',
                                                                            aspect="auto",
                                                                            extent=[epoch_motion_times[0],
                                                                                    epoch_motion_times[-1],
                                                                                    len(epoch_motion),
                                                                                    0,
                                                                                    ],
                                                                            origin = 'upper',
                                                                            vmax=colorplotmax_motion,
                                                                            vmin=-1,
                                                                            )
                motion_DA_psth_ax[motion_heatmap_row, d_idx].yaxis.set_major_locator(
                    matplotlib.ticker.MultipleLocator(len(epoch_motion)))
                motion_DA_psth_ax[motion_heatmap_row, d_idx].yaxis.set_minor_locator(
                    matplotlib.ticker.AutoMinorLocator(2))


                # motion_DA_psth_ax[motion_PSTH_row, d_idx].axvline(x = shock_on,
                #                                             color ='gray',
                #                                             linestyle='dashed',
                #                                             linewidth =  stroke_cue_reward_vertical,
                #                                             )
                motion_DA_psth_ax[motion_PSTH_row, d_idx].axhline(y = 0,
                                                            color ='gray',
                                                            linestyle= 'dashed',
                                                            linewidth =  linewidth_0_DA,
                                                            alpha = alpha0line,
                                                            )
                motion_DA_psth_ax[motion_PSTH_row, d_idx].axvspan(cue_on,
                                                            cue_off,
                                                            alpha = alpha_cue_shade,
                                                            facecolor = color_cue_shade,
                                                            linewidth = None,
                                                            )
                motion_DA_psth_ax[motion_PSTH_row, d_idx].axvspan(shock_on,
                                                            shock_off,
                                                            alpha = alpha_cue_shade,
                                                            facecolor = 'pink',
                                                            linewidth = None,
                                                            )
                motion_DA_psth_ax[motion_PSTH_row, d_idx].plot(epoch_motion_times,
                                                         epoch_motion_mean,
                                                         color = 'k',
                                                         linewidth = stroke_PSTH,
                                                         linestyle = linestyle_DA,
                                                         )
                motion_DA_psth_ax[motion_PSTH_row, d_idx].fill_between(epoch_motion_times,
                                                                 epoch_motion_mean - epoch_motion_sem,
                                                                 epoch_motion_mean + epoch_motion_sem,
                                                                 facecolor= 'k',
                                                                 alpha = alpha_PSTH_error,
                                                                 )
                motion_DA_psth_ax[motion_PSTH_row, 0].set_ylabel('motion\n(a.u.)',
                                                           fontsize = fontsize_label,
                                                           labelpad = -20)
                if d_idx == 0:
                    max_motion_day =  0
                    min_motion_day = 0
                max_motion = np.max(epoch_motion_mean)
                min_motion = np.min(epoch_motion_mean)
                if max_motion > max_motion_day:
                    max_motion_day = max_motion
                if min_motion < min_motion_day:
                    min_motion_day = min_motion
                if day == days[-1]:
                    rounded_max_avg_motion = (math.floor(max_motion_day / 5) * 5) #rounding to nearest 5 near max to set
                    #rounded_max_avg_dff = (round(max_dff_day / 5) * 5)
                    motion_DA_psth_ax[motion_PSTH_row, 0].set_ylim([min_motion_day*1.2, max_motion_day*1.2])
                    if rounded_max_avg_motion == 0:
                        if max_motion_day*1.2 < 2:
                            rounded_max_avg_motion = 1
                        else:
                            rounded_max_avg_motion = 2
                    motion_DA_psth_ax[motion_PSTH_row, d_idx].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(abs(rounded_max_avg_motion)))
                #     #motion_DA_psth_ax[motion_PSTH_row, d_idx].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(2))
                    motion_DA_psth_ax[motion_PSTH_row, d_idx].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))

            if plot_DA_heatmap:
                try:
                    motion_DA_psth_ax[DA_heatmap_row, d_idx].axvline(x=cue_on,
                                                                   color='black',
                                                                   linestyle='dashed',
                                                                   linewidth = stroke_cue_reward_vertical,
                                                                   )
                    motion_DA_psth_ax[DA_heatmap_row, d_idx].axvline(x=cue_off,
                                                                    color='black',
                                                                    linestyle='dashed',
                                                                    linewidth = stroke_cue_reward_vertical,
                                                                    )
                    motion_DA_psth_ax[DA_heatmap_row, d_idx].axvline(x = shock_on,
                                                                    color ='black',
                                                                    linestyle='dashed',
                                                                    linewidth =  stroke_cue_reward_vertical,
                                                                    )
                    motion_DA_psth_ax[DA_heatmap_row, d_idx].axvline(x = shock_off,
                                                                    color ='black',
                                                                    linestyle='dashed',
                                                                    linewidth =  stroke_cue_reward_vertical,
                                                                    )
                    if d_idx == 0:
                        colorplotmax = np.max(epoch_dff_mean)
                        motion_DA_psth_ax[DA_heatmap_row, 0].set_ylabel('trial (dLight)',
                                                                      fontsize = fontsize_label,
                                                                      )
                    dff_heatmap = motion_DA_psth_ax[DA_heatmap_row, d_idx].imshow(epoch_dff,
                                                                                cmap=plt.cm.viridis,
                                                                                interpolation='none',
                                                                                aspect="auto",
                                                                                extent=[epoch_dff_times[0],
                                                                                        epoch_dff_times[-1],
                                                                                        len(epoch_dff),
                                                                                        0,
                                                                                        ],
                                                                                origin = 'upper',
                                                                                vmax=colorplotmax,
                                                                                vmin=-1,
                                                                                )
                    motion_DA_psth_ax[DA_heatmap_row, d_idx].yaxis.set_major_locator(
                        matplotlib.ticker.MultipleLocator(len(epoch_dff)))
                    motion_DA_psth_ax[DA_heatmap_row, d_idx].yaxis.set_minor_locator(
                        matplotlib.ticker.AutoMinorLocator(2))
                except Exception as e:
                    print(f'error with heatmap DA data {animal} day {day} \n {e}')
            if plot_DA_PSTH:
                try:
                    # motion_DA_psth_ax[DA_PSTH_row, d_idx].axvline(x = shock_on,
                    #                                             color ='gray',
                    #                                             linestyle='dashed',
                    #                                             linewidth =  stroke_cue_reward_vertical,
                    #                                             )
                    motion_DA_psth_ax[DA_PSTH_row, d_idx].axhline(y = 0,
                                                                color ='gray',
                                                                linestyle= 'dashed',
                                                                linewidth =  linewidth_0_DA,
                                                                alpha = alpha0line,
                                                                )
                    motion_DA_psth_ax[DA_PSTH_row, d_idx].axvspan(cue_on,
                                                                cue_off,
                                                                alpha = alpha_cue_shade,
                                                                facecolor = color_cue_shade,
                                                                linewidth = None,
                                                                )
                    motion_DA_psth_ax[DA_PSTH_row, d_idx].axvspan(shock_on,
                                                                shock_off,
                                                                alpha = alpha_cue_shade,
                                                                facecolor = 'pink',
                                                                linewidth = None,
                                                                )
                    # for epoch in epoch_dff:
                    #     motion_DA_psth_ax[DA_PSTH_row, d_idx].plot(epoch_dff_times,
                    #                                               epoch,
                    #                                               color = color_DA_PSTH,
                    #                                               linewidth = stroke_PSTH,
                    #                                               linestyle = linestyle_DA,
                    #                                               )

                    motion_DA_psth_ax[DA_PSTH_row, d_idx].plot(epoch_dff_times,
                                                              epoch_dff_mean,
                                                              color = color_DA_PSTH,
                                                              linewidth = stroke_PSTH,
                                                              linestyle = linestyle_DA,
                                                              )
                    motion_DA_psth_ax[DA_PSTH_row, d_idx].fill_between(epoch_dff_times,
                                                                      epoch_dff_mean - epoch_dff_sem,
                                                                      epoch_dff_mean + epoch_dff_sem,
                                                                      facecolor= color_DA_PSTH,
                                                                      alpha = alpha_PSTH_error,
                                                                      )
                    motion_DA_psth_ax[DA_PSTH_row, 0].set_ylabel(f'{dlight_ylabel}',
                                                               fontsize = fontsize_label,
                                                               )
                    if d_idx == 0:
                        max_dff_day =  0
                        min_dff_day = 0
                    max_dff = np.max(epoch_dff_mean)
                    min_dff = np.min(epoch_dff_mean)
                    if max_dff > max_dff_day:
                        max_dff_day = max_dff
                    if min_dff < min_dff_day:
                        min_dff_day = min_dff
                    if day == days[-1]:
                        rounded_max_avg_dff = (math.floor(max_dff_day / 5) * 5) #rounding to nearest 5 near max to set
                        #rounded_max_avg_dff = (round(max_dff_day / 5) * 5)
                        motion_DA_psth_ax[DA_PSTH_row, 0].set_ylim([min_dff_day*1.2, max_dff_day*1.2])
                        if rounded_max_avg_dff == 0:
                            if max_dff_day*1.2 < 2:
                                rounded_max_avg_dff = 1
                            else:
                                rounded_max_avg_dff = 2
                        if dff_to_plot == 'norm_to_3_peak':
                            rounded_max_avg_dff = 0.5
                        print(rounded_max_avg_dff)
                        motion_DA_psth_ax[DA_PSTH_row, d_idx].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(abs(rounded_max_avg_dff)))
                    #     #motion_DA_psth_ax[DA_PSTH_row, d_idx].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(2))
                        motion_DA_psth_ax[DA_PSTH_row, d_idx].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
                except Exception as e:
                    print(f'error with PSTH DA data {animal} day {day} \n {e}')

            if d_idx ==0:
                motion_DA_psth_ax[0, d_idx].set_title(f'day: {int(day)}', fontsize = fontsize_title)
            else:
                motion_DA_psth_ax[0, d_idx].set_title(f'{int(day)}', fontsize = fontsize_title)
            motion_DA_psth_ax[-1, d_idx].set_xlim(xlim_in_sec)
            motion_DA_psth_ax[-1, d_idx].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
            motion_DA_psth_ax[-1, d_idx].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
    if plot_DA_heatmap:
        try:
            cbar_ax = motion_DA_psth_ax[DA_heatmap_row, -1].inset_axes([1.02, 0, .05, 1])
            cbar = plt.colorbar(dff_heatmap, cax = cbar_ax, pad= 0, fraction=0.04)# label = '% dF/F')
            cbar.set_label('% dF/F', rotation = 270, va = 'bottom')
        except:
            pass
    if plot_motion:
        try:
            cbar_ax_motion = motion_DA_psth_ax[motion_heatmap_row, -1].inset_axes([1.02, 0, .05, 1])
            cbar_motion = plt.colorbar(dff_motion_heatmap, cax = cbar_ax_motion, pad= 0, fraction=0.04)# label = '% dF/F')
            cbar_motion.set_label('motion (au)', rotation = 270, va = 'bottom')
        except:
            pass
    motion_DA_psth_fig.suptitle(f'{animal} - {condition}',
                              fontsize = fontsize_title,
                              )
    if align_to_cue:
        motion_DA_psth_fig.supxlabel('time from cue onset (s)',
                                   fontsize = fontsize_label,
                                   )
    else:
        motion_DA_psth_fig.supxlabel('time from reward (s)',
                                   fontsize = fontsize_label,
                                   )
    #for ax in motion_DA_psth_ax.reshape(-1):
    standardize_plot_graphics(motion_DA_psth_ax)
    set_ax_size_inches(axsize[0], axsize[1],  motion_DA_psth_ax)
    if save_fig:
        motion_DA_psth_fig.savefig(os.path.join(fig_path, f'example fear motion DA PSTH {animal}.pdf'),
                                 transparent = True,
                                 bbox_inches = 'tight',
                                 bbox_extra_artists = [motion_DA_psth_fig._suptitle,
                                                       motion_DA_psth_fig._supxlabel]
                                 )
        if save_png:
            motion_DA_psth_fig.savefig(os.path.join(fig_path, f'example fear motion DA PSTH {animal}.png'),
                                     transparent = True,
                                     bbox_inches = 'tight',
                                     bbox_extra_artists = [motion_DA_psth_fig._suptitle,
                                                           motion_DA_psth_fig._supxlabel]
                                     )
    return motion_DA_psth_fig, motion_DA_psth_ax

def plotPSTHbyBeforeAfter(trial_df,
                          animal,
                          learned_trial,
                          trial_before_after = (5,5),
                          plot_lick_raster = True,
                          plot_lick_PSTH = True,
                          plot_DA_heatmap = True,
                          plot_DA_PSTH = True,
                          align_to_cue = True,
                          plot_in_sec = True,
                          xlim_in_sec = [-2.5, 7.5],
                          ylim_lick = [-1, 12],
                          ylim_DA = [None, None],
                          dff_to_plot ='%',
                          axsize = (0.5, 0.35),
                          sharey = 'row',
                          stroke_raster = 0.35,
                          stroke_PSTH = 0.5,
                          color_DA_PSTH = 'dodgerblue',
                          color_lick_PSTH = 'green',
                          alpha_PSTH_error = 0.3,
                          color_cue_shade = '#939598',
                          alpha_cue_shade = 0.5,
                          fontsize_title = 7,
                          fontsize_label = 7,
                          fontsize_ticks = 6,
                          smooth_lick = 0.75,
                          stroke_cue_reward_vertical = 0.25,
                          linestyle_lick = 'solid',
                          linestyle_DA = 'dashed',
                          fig_path = '',
                          save_fig = False,
                          save_png = False,
                          linewidth_0_lick = 0,
                          linewidth_0_DA = 0,
                          alpha0line = 0.5,
                          norm_to_max_rewards = 0,
                          ):
    data_in_s = True if np.max(trial_df['cue_dur'])<49 else False

    same_unit = plot_in_sec == data_in_s
    #scale between mc and s depending on data format and desired output
    same_unit = plot_in_sec == data_in_s
    scale_factor = 1 if same_unit else (1/1000 if plot_in_sec else 1000)
    binsize_for_lick_PSTH = 0.1 if data_in_s else 100

    trials_df= trial_df.copy()
    if 'cue_trial_num' not in trial_df.columns:
        trials_df['cue_trial_num'] = trials_df.groupby('animal').cumcount()+1


    if isinstance(trial_before_after, int):
        trials_to_plot = [trial_before_after,trial_before_after]
    else:

        trials_to_plot = trial_before_after
    trials_before = [learned_trial -trials_to_plot[0] +1, learned_trial]
    trials_after = [learned_trial + 1, learned_trial + trials_to_plot[1]]
    trial_ranges = [trials_before, trials_after]
    total_time_to_plot = (xlim_in_sec[1] - xlim_in_sec[0]) *scale_factor
    baseline_time_to_plot = xlim_in_sec[0] * (-scale_factor)

    #determines parameters and axes for figure
    total_col = 2
    total_rows =  plot_lick_raster + plot_lick_PSTH + plot_DA_heatmap + plot_DA_PSTH
    fig_width = total_col * axsize[0] + 2
    fig_height = total_rows * axsize[1] + 2# pad each side with an inch


    lick_DA_psth_fig, lick_DA_psth_ax = plt.subplots(total_rows,
                                                     total_col,
                                                     figsize = (fig_width, fig_height),
                                                     sharex = True,
                                                     sharey = sharey,
                                                     squeeze = False)# layout = 'constrained')#, subplotpars={ 'left': one_inch_width, 'top' : one_inch_height, 'hspace': 0.1, 'wspace': 0.1}) #, constrained_layout = True)
    lick_raster_unassigned = True
    lick_PSTH_unassigned = True
    DA_heatmap_unassigned = True
    DA_PSTH_unassigned = True
    for row in np.arange(total_rows):

        if plot_lick_raster and lick_raster_unassigned:
            lick_raster_row = row
            lick_raster_unassigned = False
        elif plot_lick_PSTH and lick_PSTH_unassigned:
            lick_PSTH_row = row
            lick_PSTH_unassigned = False
        elif plot_DA_heatmap and DA_heatmap_unassigned:
            DA_heatmap_row = row
            DA_heatmap_unassigned = False
        elif plot_DA_PSTH and DA_PSTH_unassigned:
            DA_PSTH_row = row
            DA_PSTH_unassigned = False


    #determine max trials per day for raster plot ylim purposes
    # raster_ylen =  len(trials_df[((trials_df['animal'] == animal)
    #                               & (trials_df['day_num'] >= days[0])
    #                              & (trials_df['day_num'] <= days[-1]))]['trial_num'].unique())
    raster_ylen = max(trials_to_plot)

    for t_idx, trials in enumerate(trial_ranges):
        animal_trials_df = trials_df[((trials_df['animal'] == animal)
                                      & (trials_df['cue_trial_num']>= trials[0])
                                      &(trials_df['cue_trial_num']<= trials[1]))]
        if not animal_trials_df.empty:
            condition = animal_trials_df['condition'].iloc[0]
            if align_to_cue:
                cue_on =  0
                cue_off = (animal_trials_df['cue_off'] - animal_trials_df['cue_on'] ).mean() * scale_factor
                reward_time = (animal_trials_df['reward_time'] - animal_trials_df['cue_on'] ).mean() * scale_factor
            else:
                cue_on = (animal_trials_df['cue_on'] - animal_trials_df['reward_time'] ).mean() * scale_factor
                cue_off = (animal_trials_df['cue_off'] - animal_trials_df['reward_time'] ).mean() * scale_factor
                reward_time = 0

            #plot lick raster and PSTH
            if plot_lick_raster or plot_lick_PSTH:
                lick_times_list = (animal_trials_df['licks_all'] * scale_factor).tolist()
                if align_to_cue:
                    lick_times_list = [trial_licks + reward_time for trial_licks in lick_times_list]


                #plot cue and reward times with lines and shading
                if plot_lick_raster:

                    lick_DA_psth_ax[lick_raster_row, t_idx].axvline(x = reward_time,
                                                                    color ='gray',
                                                                    linestyle='dashed',
                                                                    linewidth =  stroke_cue_reward_vertical,
                                                                    )
                    lick_DA_psth_ax[lick_raster_row, t_idx].axvspan(cue_on,
                                                                    cue_off,
                                                                    alpha = alpha_cue_shade,
                                                                    facecolor = color_cue_shade,
                                                                    linewidth = None,
                                                                    )

                    lick_DA_psth_ax[0, t_idx].eventplot(lick_times_list,
                                                        linewidths = stroke_raster,
                                                        linelengths = 0.75,
                                                        colors= 'black',
                                                        lineoffsets = np.arange(len(lick_times_list))+1,
                                                        )
                    lick_DA_psth_ax[lick_raster_row, t_idx].set_ylim([raster_ylen + 0.5, 0.5])
                    lick_DA_psth_ax[lick_raster_row, 0].set_ylabel('trial', fontsize = fontsize_label)
                    #lick_DA_psth_ax[lick_raster_row, t_idx].yaxis.set_major_locator(matplotlib.ticker.LinearLocator(2))

                    lick_DA_psth_ax[lick_raster_row, t_idx].yaxis.set_major_locator(matplotlib.ticker.FixedLocator([1, len(lick_times_list)]))
                    lick_DA_psth_ax[lick_raster_row, t_idx].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))


                if plot_lick_PSTH:
                    lick_DA_psth_ax[lick_PSTH_row, t_idx].axvline(x = reward_time,
                                                                  color ='gray',
                                                                  linestyle='dashed',
                                                                  linewidth =  stroke_cue_reward_vertical,
                                                                  )
                    lick_DA_psth_ax[lick_PSTH_row, t_idx].axhline(y = 0,
                                                                  color ='gray',
                                                                  linestyle='dashed',
                                                                  linewidth =  linewidth_0_lick,
                                                                  alpha = alpha0line,
                                                                  )
                    lick_DA_psth_ax[lick_PSTH_row, t_idx].axvspan(cue_on,
                                                                  cue_off,
                                                                  alpha = alpha_cue_shade,
                                                                  facecolor = color_cue_shade,
                                                                  linewidth = None,
                                                                  )

                    lick_hist_dict = lpf.getLickPSTH(lick_times_list,
                                                     binsize = binsize_for_lick_PSTH,
                                                     total_time_window= total_time_to_plot,
                                                     baseline_period = baseline_time_to_plot,
                                                     in_seconds = plot_in_sec,
                                                     time_window_in_seconds = data_in_s)

                    if smooth_lick:
                        lick_hist_mean = gaussian_filter1d(lick_hist_dict['mean_hist'],
                                                           sigma = smooth_lick,
                                                           )
                        lick_hist_sem= gaussian_filter1d(lick_hist_dict['sem_hist'],
                                                         sigma = smooth_lick,
                                                         )
                    lick_DA_psth_ax[lick_PSTH_row, t_idx].plot(lick_hist_dict['bins'][1:],
                                                               lick_hist_mean,
                                                               color = color_lick_PSTH,
                                                               linewidth = stroke_PSTH,
                                                               linestyle = linestyle_lick,
                                                               )



                    lick_DA_psth_ax[lick_PSTH_row, t_idx].fill_between(lick_hist_dict['bins'][1:],
                                                                       lick_hist_mean - lick_hist_sem,
                                                                       lick_hist_mean + lick_hist_sem,
                                                                       facecolor= color_lick_PSTH,
                                                                       alpha = alpha_PSTH_error,
                                                                       )

                    lick_DA_psth_ax[lick_PSTH_row, t_idx].set_ylim(ylim_lick)
                    lick_DA_psth_ax[lick_PSTH_row, 0].set_ylabel('lick (Hz)',  fontsize = fontsize_label)
                    lick_DA_psth_ax[lick_PSTH_row, t_idx].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
                    lick_DA_psth_ax[lick_PSTH_row, t_idx].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))


            if plot_DA_heatmap or plot_DA_PSTH:

                epoch_dff_times = np.mean(np.array(animal_trials_df['epoch_time'].to_list()), axis = 0) * scale_factor

                if align_to_cue:
                    epoch_dff_times = epoch_dff_times + reward_time

                if dff_to_plot =='%':

                    epoch_dff = np.array(animal_trials_df['epoch_dff'].to_list())
                    dlight_ylabel = 'dLight\n(% dF/F)'

                epoch_dff_mean = np.mean(epoch_dff, axis = 0)
                epoch_dff_sem = stats.sem(epoch_dff,
                                          ddof = 1,
                                          axis = 0,
                                          nan_policy = 'propagate',
                                          )

                # if norm_to_max_rewards:
                #     max_reward_values = trials_df[trials_df['animal'] == animal].sort_values(['epoch_dff_peak_consume_norm'],ascending=False).groupby(['animal']).head(10)
                #     print(max_reward_values)
                #     DA_normalization = max_reward_values.groupby(['animal'])['epoch_dff_peak_consume_norm'].agg(lambda g: g.iloc[0:norm_to_max_individual_rewards].mean())

                #     print(DA_normalization)
                #     epoch_dff_mean = epoch_dff_mean/ float(DA_normalization)
                #     epoch_dff_sem = epoch_dff_sem /float(DA_normalization)
                if plot_DA_heatmap:
                    try:
                        lick_DA_psth_ax[DA_heatmap_row, t_idx].axvline(x=cue_on,
                                                                       color='gray',
                                                                       linestyle='dashed',
                                                                       linewidth = stroke_cue_reward_vertical,
                                                                       )
                        lick_DA_psth_ax[DA_heatmap_row, t_idx].axvline(x=cue_off,
                                                                       color='gray',
                                                                       linestyle='dashed',
                                                                       linewidth = stroke_cue_reward_vertical,
                                                                       )
                        lick_DA_psth_ax[DA_heatmap_row, t_idx].axvline(x = reward_time,
                                                                       color ='gray',
                                                                       linestyle='dashed',
                                                                       linewidth =  stroke_cue_reward_vertical,
                                                                       )
                        if t_idx == 0:
                            colorplotmax = np.max(epoch_dff_mean)
                            lick_DA_psth_ax[DA_heatmap_row, 0].set_ylabel('trial',
                                                                          fontsize = fontsize_label,
                                                                          )
                        dff_heatmap = lick_DA_psth_ax[DA_heatmap_row, t_idx].imshow(epoch_dff,
                                                                                    cmap=plt.cm.viridis,
                                                                                    interpolation='none',
                                                                                    aspect="auto",
                                                                                    extent=[epoch_dff_times[0],
                                                                                            epoch_dff_times[-1],
                                                                                            len(epoch_dff),
                                                                                            0,
                                                                                            ],
                                                                                    origin = 'upper',
                                                                                    vmax=colorplotmax,
                                                                                    vmin=-1,
                                                                                    )
                        lick_DA_psth_ax[DA_heatmap_row, t_idx].yaxis.set_major_locator(
                            matplotlib.ticker.MultipleLocator(len(lick_times_list)))
                        lick_DA_psth_ax[DA_heatmap_row, t_idx].yaxis.set_minor_locator(
                            matplotlib.ticker.AutoMinorLocator(2))
                    except Exception as e:
                        print(f'error with DA data {animal} \n {e}')
                if plot_DA_PSTH:
                    try:
                        lick_DA_psth_ax[DA_PSTH_row, t_idx].axvline(x = reward_time,
                                                                    color ='gray',
                                                                    linestyle='dashed',
                                                                    linewidth =  stroke_cue_reward_vertical,
                                                                    )
                        lick_DA_psth_ax[DA_PSTH_row, t_idx].axhline(y = 0,
                                                                    color ='gray',
                                                                    linestyle='dashed',
                                                                    linewidth =  linewidth_0_DA,
                                                                    alpha = alpha0line,
                                                                    )
                        lick_DA_psth_ax[DA_PSTH_row, t_idx].axvspan(cue_on,
                                                                    cue_off,
                                                                    alpha = alpha_cue_shade,
                                                                    facecolor = color_cue_shade,
                                                                    linewidth = None,
                                                                    )
                        lick_DA_psth_ax[DA_PSTH_row, t_idx].plot(epoch_dff_times,
                                                                 epoch_dff_mean,
                                                                 color = color_DA_PSTH,
                                                                 linewidth = stroke_PSTH,
                                                                 linestyle = linestyle_DA,
                                                                 )
                        lick_DA_psth_ax[DA_PSTH_row, t_idx].fill_between(epoch_dff_times,
                                                                         epoch_dff_mean - epoch_dff_sem,
                                                                         epoch_dff_mean + epoch_dff_sem,
                                                                         facecolor= color_DA_PSTH,
                                                                         alpha = alpha_PSTH_error,
                                                                         )
                        lick_DA_psth_ax[DA_PSTH_row, 0].set_ylabel(f'{dlight_ylabel}',
                                                                   fontsize = fontsize_label,
                                                                   )
                        if t_idx == 0:
                            max_dff_day =  0
                            min_dff_day = 0
                        max_dff = np.max(epoch_dff_mean)
                        min_dff = np.min(epoch_dff_mean)
                        if max_dff > max_dff_day:
                            max_dff_day = max_dff
                        if min_dff < min_dff_day:
                            min_dff_day = min_dff
                        if t_idx == 1:
                            rounded_max_avg_dff = (round(max_dff_day / 5) * 5) #rounding to nearest 5 near max to set
                            lick_DA_psth_ax[DA_PSTH_row, 0].set_ylim([min_dff_day*1.2, max_dff_day*1.2])
                            lick_DA_psth_ax[DA_PSTH_row, 0].set_ylim(ylim_DA)
                            if rounded_max_avg_dff == 0:
                                if max_dff_day*1.2 < 2:
                                    rounded_max_avg_dff = 1
                                else:
                                    rounded_max_avg_dff = 2
                            lick_DA_psth_ax[DA_PSTH_row, t_idx].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(abs(rounded_max_avg_dff)))
                            lick_DA_psth_ax[DA_PSTH_row, t_idx].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
                    except Exception as e:
                        print(f'error with DA data {animal} \n {e}')

            if t_idx ==0:
                lick_DA_psth_ax[0, t_idx].set_title('before learned trial:', fontsize = fontsize_title)
            else:
                lick_DA_psth_ax[0, t_idx].set_title('after learned trial', fontsize = fontsize_title)
            lick_DA_psth_ax[-1, t_idx].set_xlim(xlim_in_sec)
            lick_DA_psth_ax[-1, t_idx].xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
            lick_DA_psth_ax[-1, t_idx].xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
    if plot_DA_heatmap:
        try:
            cbar_ax = lick_DA_psth_ax[DA_heatmap_row, -1].inset_axes([1.02, 0, .05, 1])
            cbar = plt.colorbar(dff_heatmap, cax = cbar_ax, pad= 0, fraction=0.04)# label = '% dF/F')
            cbar.set_label('% dF/F', rotation = 270, va = 'bottom')
        except:
            pass
    lick_DA_psth_fig.suptitle(f'{animal} - {condition}',
                              fontsize = fontsize_title,
                              )
    if align_to_cue:
        lick_DA_psth_fig.supxlabel('time from cue onset (s)',
                                   fontsize = fontsize_label,
                                   )
    else:
        lick_DA_psth_fig.supxlabel('time from reward (s)',
                                   fontsize = fontsize_label,
                                   )
    #for ax in lick_DA_psth_ax.reshape(-1):
    standardize_plot_graphics(lick_DA_psth_ax)
    set_ax_size_inches(axsize[0], axsize[1],  lick_DA_psth_ax)
    if save_fig:
        lick_DA_psth_fig.savefig(os.path.join(fig_path, f'example raster PSTH {animal}.pdf'),
                                 transparent = True,
                                 bbox_inches = 'tight',
                                 bbox_extra_artists = [lick_DA_psth_fig._suptitle,
                                                       lick_DA_psth_fig._supxlabel]
                                 )
        if save_png:
            lick_DA_psth_fig.savefig(os.path.join(fig_path, f'example raster PSTH {animal}.png'),
                                     transparent = True,
                                     bbox_inches = 'tight',
                                     bbox_extra_artists = [lick_DA_psth_fig._suptitle,
                                                           lick_DA_psth_fig._supxlabel]
                                     )
    return lick_DA_psth_fig, lick_DA_psth_ax

def getCumSumLearnedTrialsAndPlotFearConditioning(trial_df,
                                                  conditions_to_plot = 'all',
                                                  colors_for_conditions = defaultdict(lambda: 'black'),
                                                  colors_for_conditions_DA = defaultdict(lambda: 'black'),
                                                  percent_max_dist = 0.75,
                                                  plot_all_individuals = True,
                                                  linewidth_behavior = 1,
                                                  linewidth_learned_trial = 0.35,
                                                  color_learned_trial = 'black',
                                                  get_DA_learned_trial = False,
                                                  DA_trial_multiple = 1.5,
                                                  linewidth_DA = 1,
                                                  linewidth_DA_trial = 0.25,
                                                  color_DA_trial = 'k',
                                                  plot_vertical = False,
                                                  use_trial_normalized_y = True,
                                                  sharex = False,
                                                  sharey = True,
                                                  xlim = [None, None],
                                                  sharey_DA = False,
                                                  linestyle_DA = (0, (3,1)),
                                                  linewidth_diagonal_behavior = 0,
                                                  linewidth_diagonal_DA = 0,
                                                  plot_examples = True,
                                                  condition_examples = {},
                                                  learning_cutoff = 0.5,
                                                  nonlearners_list =[],
                                                  axsize = (1.1, 0.82, 0.1, 0.1),
                                                  fontsize_label = 7,
                                                  plot_on_2_lines = False,
                                                  save_fig = False,
                                                  save_png = False,
                                                  fig_path = '',
                                                  linestyle_learned_trial = 'solid',
                                                  linestyle_learned_trial_DA = 'dashed',
                                                  renamed_mice = True,
                                                  ylim_behavior = [None, None],
                                                  ylim_DA = [None, None],
                                                  plot_behavior = True,
                                                  behavior_cumsum = 'cumsum_freezing_norm',
                                                  da_cumsum = 'cumsum_cue_dff_auc_norm',
                                                  behavior_data_direction = 'increase',
                                                  da_data_direction = 'decrease',
                                                  nonlearners_list_DA =[],
                                                  ):
    conditions = lpf.get_conditions_as_list(conditions_to_plot, trial_df)
    original_df = trial_df.copy()
    trial_df = trial_df[trial_df['cue_type'] == cue_type].copy()
    trial_df = trial_df[trial_df['condition'].isin(conditions)].copy()

    animals_by_condition = trial_df.groupby(['condition'])['animal'].unique().to_dict()
    #for plotting
    num_animals_by_condition = {x: len(animals_by_condition[x])
                                for x
                                in animals_by_condition.keys()}
    max_animal_single_condition = num_animals_by_condition[max(num_animals_by_condition, key = num_animals_by_condition.get)]

    DA_time_wind = '_500ms' #


    behavior_cumsum_df = trial_df.pivot(index = ['cue_trial_num'],
                                    columns = ['condition', 'animal'],
                                    values = behavior_cumsum,
                                    )
    if get_DA_learned_trial:
        DA_cue_cumsum_df = trial_df.pivot(index = ['cue_trial_num'],
                                      columns = ['condition', 'animal'],
                                      values = da_cumsum,
                                      )
    title = f'cumsum individuals {behavior_cumsum.split('_')[1]}'

    if get_DA_learned_trial:
        DA_cue_cumsum_df = DA_cue_cumsum_df / DA_normalization


    trial_normed_behavior_cumsum_df =behavior_cumsum_df / behavior_cumsum_df.count()
    if get_DA_learned_trial:
        trial_normed_DA_cue_cumsum_df = DA_cue_cumsum_df/DA_cue_cumsum_df.count()

    if use_trial_normalized_y:
        behavior_cumsum_df = trial_normed_behavior_cumsum_df
        if get_DA_learned_trial:
            DA_cue_cumsum_df = trial_normed_DA_cue_cumsum_df

    if plot_examples == False:
        max_animal_single_condition = max_animal_single_condition - 1
        title = title + ' without examples'
    if plot_vertical:
        total_row = max_animal_single_condition
        total_col = len(conditions)
        title = title + ' vertical'
    else:
        if plot_on_2_lines:
            total_row = len(conditions) *2
            total_col = 9
            title = title + ' on 2 lines'
        else:
            total_row = len(conditions)
            total_col = max_animal_single_condition

    if get_DA_learned_trial:
        title =title + ' with DA'
        DA_ax_list = []

    if plot_all_individuals:
        fig_cumsum_individuals, ax_cumsum_individuals = plt.subplots(total_row,
                                                                     total_col,
                                                                     figsize=(total_col +2 ,
                                                                              total_row + 2),
                                                                     sharex= sharex,
                                                                     sharey = sharey,
                                                                     constrained_layout= False,
                                                                     squeeze = False,
                                                                     )
    learned_trials_dict = dict.fromkeys(conditions, {})
    abruptness_of_change_max_dict = dict.fromkeys(conditions, {})
    abruptness_of_change_learned_trial_dict = dict.fromkeys(conditions, {})
    abruptness_of_change_max_normed_dict = dict.fromkeys(conditions, {})
    abruptness_of_change_learned_trial_normed_dict = dict.fromkeys(conditions, {})

    if get_DA_learned_trial:
        DA_learned_trials_dict = dict.fromkeys(conditions, {})
        DA_abruptness_of_change_max_dict = dict.fromkeys(conditions, {})
        DA_abruptness_of_change_learned_trial_dict = dict.fromkeys(conditions, {})
        DA_abruptness_of_change_max_normed_dict = dict.fromkeys(conditions, {})
        DA_abruptness_of_change_learned_trial_normed_dict = dict.fromkeys(conditions, {})
        lag_to_learn_dict = dict.fromkeys(conditions, {})

    learned_trial_params_behavior_dict = dict.fromkeys(conditions, {})
    learned_trial_params_DA_cue_dict = dict.fromkeys(conditions, {})

    for con_num, condition in enumerate(conditions):
        example_counter = 0
        example_animal = ''
        if plot_examples == False:
            example_animal = condition_examples[condition]
            #fig, ax =




        learned_trials_dict[condition] = dict.fromkeys(animals_by_condition[condition], [])
        abruptness_of_change_max_dict[condition] = dict.fromkeys(animals_by_condition[condition], [])
        abruptness_of_change_learned_trial_dict[condition] = dict.fromkeys(animals_by_condition[condition], [])
        abruptness_of_change_max_normed_dict[condition] = dict.fromkeys(animals_by_condition[condition], [])
        abruptness_of_change_learned_trial_normed_dict[condition] = dict.fromkeys(animals_by_condition[condition], [])
        learned_trial_params_behavior_dict[condition] = dict.fromkeys(animals_by_condition[condition], [])
        if get_DA_learned_trial:
            DA_learned_trials_dict[condition] = dict.fromkeys(animals_by_condition[condition], [])
            DA_abruptness_of_change_max_dict[condition] = dict.fromkeys(animals_by_condition[condition], [])
            DA_abruptness_of_change_learned_trial_dict[condition] = dict.fromkeys(animals_by_condition[condition], [])
            DA_abruptness_of_change_max_normed_dict[condition] = dict.fromkeys(animals_by_condition[condition], [])
            DA_abruptness_of_change_learned_trial_normed_dict[condition] = dict.fromkeys(animals_by_condition[condition], [])
            lag_to_learn_dict[condition] = dict.fromkeys(animals_by_condition[condition], [])
            learned_trial_params_DA_cue_dict[condition] = dict.fromkeys(animals_by_condition[condition], [])

        sorted_animals = natsorted(animals_by_condition[condition])
        for animal_num, animal in enumerate(sorted_animals):



            behavior_cumsum_y = behavior_cumsum_df[condition, animal].to_numpy() #cumsum behaviors
            trials_cumsum_x = behavior_cumsum_df.index.values #trial numbers
            trials_cumsum_x = trials_cumsum_x[~np.isnan(behavior_cumsum_y)] #filter out NaNs in behaviors and trials
            behavior_cumsum_y = behavior_cumsum_y[~np.isnan(behavior_cumsum_y)]

            learned_trial_params_behavior = lpf.getCumsumChangePoint(trials_cumsum_x,
                                                                 behavior_cumsum_y,
                                                                 percent_max_dist = percent_max_dist,
                                                                 data_direction = behavior_data_direction,
                                                                 animal_name = animal)
            #print(learned_trial_params_behavior['learned_trial'])
            if ((max(behavior_cumsum_y) > learning_cutoff) and (animal not in nonlearners_list)):
                learned_trials_dict[condition][animal] =  learned_trial_params_behavior['learned_trial']

                abruptness_of_change_max_dict[condition][animal] =  learned_trial_params_behavior['max_dist']
                abruptness_of_change_max_normed_dict[condition][animal] = learned_trial_params_behavior['max_dist_norm']

                abruptness_of_change_learned_trial_dict[condition][animal] =  learned_trial_params_behavior['dist_at_learned_trial']
                abruptness_of_change_learned_trial_normed_dict[condition][animal] = learned_trial_params_behavior['dist_at_learned_trial_norm']

                learned_trial_params_behavior_dict[condition][animal] = learned_trial_params_behavior
                learned_trial_for_DA_cutoff = learned_trial_params_behavior['learned_trial']
            else:
                learned_trial_for_DA_cutoff = np.max(trials_cumsum_x)
            if get_DA_learned_trial and (animal not in nonlearners_list_DA):
                DA_cumsum_y = DA_cue_cumsum_df[condition, animal].to_numpy()

                DA_cumsum_y = DA_cumsum_y[~np.isnan(DA_cumsum_y)]

                if len(DA_cumsum_y) > 0:
                    #now do same for DA



                    da_full_y = DA_cumsum_y

                    DA_cumsum_y = DA_cumsum_y[trials_cumsum_x
                                              <= (learned_trial_for_DA_cutoff
                                                  * DA_trial_multiple)]
                    DA_cumsum_y = DA_cumsum_y[~np.isnan(DA_cumsum_y)]
                    DA_trials_cumsum_x = trials_cumsum_x[trials_cumsum_x
                                                         <= (learned_trial_for_DA_cutoff
                                                             * DA_trial_multiple)]
                    DA_trials_cumsum_x = DA_trials_cumsum_x[~np.isnan(DA_trials_cumsum_x)]

                    learned_trial_params_DA_cue = lpf.getCumsumChangePoint(DA_trials_cumsum_x,
                                                                           DA_cumsum_y,
                                                                           percent_max_dist =
                                                                           percent_max_dist,
                                                                           data_direction = da_data_direction,
                                                                           animal_name = animal)


                    learned_trial_params_DA_cue_dict[condition][animal] = learned_trial_params_DA_cue
                    DA_learned_trials_dict[condition][animal] =  learned_trial_params_DA_cue['learned_trial']
                    DA_abruptness_of_change_max_dict[condition][animal] = learned_trial_params_DA_cue['max_dist']
                    DA_abruptness_of_change_learned_trial_dict[condition][animal] = learned_trial_params_DA_cue['dist_at_learned_trial']
                    DA_abruptness_of_change_max_normed_dict[condition][animal] = learned_trial_params_DA_cue['max_dist_norm']
                    DA_abruptness_of_change_learned_trial_normed_dict[condition][animal] = learned_trial_params_DA_cue['dist_at_learned_trial_norm']
                    lag_to_learn_dict[condition][animal] = learned_trial_params_behavior['learned_trial'] - learned_trial_params_DA_cue['learned_trial']



            if example_animal == animal:
                example_counter = 1
            if plot_all_individuals:
                if plot_vertical:
                    left_individual_ax = ax_cumsum_individuals[animal_num - example_counter, con_num]
                    fig_cumsum_individuals.suptitle(condition, position ='left')
                else:
                    if plot_on_2_lines:
                        if animal_num - example_counter < 9:
                            individual_ax = ax_cumsum_individuals[con_num *2, animal_num- example_counter]
                        else:
                            individual_ax = ax_cumsum_individuals[con_num *2 +1, animal_num- example_counter-9]
                    else:
                        individual_ax = ax_cumsum_individuals[con_num, animal_num- example_counter]
                    ax_cumsum_individuals[con_num,0].set_ylabel(condition,  rotation = 'horizontal', ha = 'right')
                if example_animal != animal:
                    individual_ax.axhline(0,
                                          linestyle = (0,(4,2)),
                                          color = 'gray',
                                          alpha =1,
                                          linewidth = 1,
                                          )
                    if plot_behavior:
                        individual_ax.plot(trials_cumsum_x,
                                           behavior_cumsum_y,
                                           color = colors_for_conditions[condition],
                                           linewidth = linewidth_behavior,
                                           )
                    if renamed_mice:
                        individual_ax.set_title(animal, fontsize = 6)
                    else:

                        individual_ax.set_title(animal.split('_')[-1])


                    if ((max(behavior_cumsum_y) > learning_cutoff) and plot_behavior and (animal not in nonlearners_list)):
                        if linewidth_diagonal_behavior == 0:
                            linestyle_trial = 'solid'
                        else:
                            linestyle_trial = 'dashed'
                        individual_ax.plot(learned_trial_params_behavior['diag_x'],
                                           learned_trial_params_behavior['diag_y'],
                                           linewidth = linewidth_diagonal_behavior,
                                           linestyle = linestyle_trial,
                                           )
                        individual_ax.axvline(learned_trial_params_behavior['learned_trial'],
                                              linewidth = linewidth_learned_trial,
                                              color = color_learned_trial,
                                              linestyle = linestyle_learned_trial,
                                              )
                    if animal not in nonlearners_list_DA:
                        if get_DA_learned_trial and (len(DA_cumsum_y) > 0):
                            if linewidth_diagonal_DA == 0:
                                linestyle_trial = 'solid'
                            else:
                                linestyle_trial = 'dashed'
                            if plot_behavior:
                                DA_ax =individual_ax.twinx()
                            else:
                                DA_ax =individual_ax
                            DA_ax_list.append(DA_ax)
                            DA_ax.axvline(learned_trial_params_DA_cue['learned_trial'],
                                          linewidth = linewidth_DA_trial,
                                          color = color_DA_trial,
                                          linestyle = linestyle_learned_trial_DA,
                                          )
                            DA_ax.plot(trials_cumsum_x,
                                       da_full_y,
                                       color = colors_for_conditions_DA[condition],
                                       linewidth = linewidth_DA,
                                       linestyle = linestyle_DA,
                                       )
                            DA_ax.plot(learned_trial_params_DA_cue['diag_x'],
                                       learned_trial_params_DA_cue['diag_y'],
                                       linewidth = linewidth_diagonal_DA,
                                       linestyle = linestyle_trial,
                                       )


                    if max(trials_cumsum_x) <= 8:
                        individual_ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(4)) #individual_ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([ 2, 4, 6,8]))
                        individual_ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))


                    elif max(trials_cumsum_x) <= 20:
                        individual_ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5)) #individual_ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([ 2, 4, 6,8]))
                        individual_ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
                        #individual_ax.set_xticklabels([ '2',  '4', '6', '8'])
                    elif max(trials_cumsum_x) <= 50:
                        individual_ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(20))
                        individual_ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
                    elif max(trials_cumsum_x) <= 80:
                        individual_ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(40))
                        individual_ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
                    elif max(trials_cumsum_x) <= 350:
                        individual_ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(100))
                        individual_ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
                    elif max(trials_cumsum_x) <= 600:
                        individual_ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(200))
                        individual_ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
                    elif max(trials_cumsum_x) <= 800:
                        individual_ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(400))
                        individual_ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
                    elif max(trials_cumsum_x) <= 2000:
                        individual_ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1000))
                        individual_ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
                    else:
                        individual_ax.xaxis.set_major_locator(matplotlib.ticker.AutoLocator())
                        individual_ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
    if plot_all_individuals:
        fig_cumsum_individuals.supxlabel('trial', fontsize= fontsize_label)
        fig_cumsum_individuals.supylabel('cumsum(anticipatory behaviors)/num trials',
                                         fontsize = fontsize_label,
                                         ha = 'center',
                                         )
        if get_DA_learned_trial and sharey_DA:
            ylim_max_DA = max(map(lambda x: x.get_ylim()[1], DA_ax_list))
            ylim_min_DA = min(map(lambda x: x.get_ylim()[0], DA_ax_list))
            for ax in DA_ax_list:
                ax.sharey(DA_ax_list[0])
                if not ax.get_subplotspec().is_last_col():
                    ax.tick_params(labelright=False)
            DA_ax_list[0].set_ylim([ylim_min_DA, ylim_max_DA])
            DA_ax_list[0].set_ylim(ylim_DA)
        if plot_behavior:
            ax_cumsum_individuals[0,0].set_ylim(ylim_behavior)
        else:
            ax_cumsum_individuals[0,0].set_ylim(ylim_DA)
        if sharex:
            ax_cumsum_individuals[0,0].set_xlim = xlim
        ax_cumsum_individuals[0,0].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
        standardize_plot_graphics(ax_cumsum_individuals)
        set_ax_size_inches(axsize[0],
                               axsize[1],
                               ax_cumsum_individuals,
                               axsize[2],
                               axsize[3],)

        if save_fig:
            new_title = lpf.cleanStringForFilename(title)
            if save_png:
                fig_cumsum_individuals.savefig(os.path.join(fig_path, f'{new_title}' + '_'.join(conditions)+'.png'),
                                               dpi = 600,
                                               transparent = True,
                                               bbox_inches = 'tight'
                                               )
            fig_cumsum_individuals.savefig(os.path.join(fig_path, f'{new_title}' + '_'.join(conditions)+'.pdf'),
                                           transparent = True,
                                           bbox_inches = 'tight'
                                           )
    summary_dict = {'learned_trial_behavior': learned_trials_dict,
                    'abruptness_max_behavior': abruptness_of_change_max_dict,
                    'abruptness_max_behavior_norm': abruptness_of_change_max_normed_dict,
                    'abruptness_learned_trial_behavior': abruptness_of_change_learned_trial_dict,
                    'abruptness_learned_trial_behavior_norm': abruptness_of_change_learned_trial_normed_dict,
                    'trials_cumsum_x': trials_cumsum_x
                    }
    if plot_all_individuals:
        summary_dict['fig'] = fig_cumsum_individuals
        summary_dict['ax'] = ax_cumsum_individuals


    if get_DA_learned_trial:
        summary_dict_DA = {'learned_trial_DA': DA_learned_trials_dict,
                            'abruptness_max_DA': DA_abruptness_of_change_max_dict,
                            'abruptness_max_DA_norm': DA_abruptness_of_change_max_normed_dict,
                            'abruptness_learned_trial_DA': DA_abruptness_of_change_learned_trial_dict,
                            'abruptness_learned_trial_DA_norm': DA_abruptness_of_change_learned_trial_normed_dict,
                            'lag_to_learn':lag_to_learn_dict,
                            }
        summary_dict = {**summary_dict, **summary_dict_DA}

    return summary_dict

def plot_example_reward_and_omission_PSTHs_by_trial_bins(trial_df,
                                     animal,
                                     windows_subset = [[0, 30], [60, 90], [260, 290]],
                                     ylim_lick = [-1, 12],
                                     stroke_PSTH = 0.5,
                                     alpha_PSTH_error = 0.3,
                                     color_cue_shade = '#939598',
                                     alpha_cue_shade = 0.5,
                                     fontsize_label = 7,
                                     smooth_lick = 0.75,
                                     linewidth_0_lick = 0,
                                     linewidth_0_DA = 0,
                                     alpha0line = 0.5,
                                     align_to_cue = True,
                                     plot_in_sec = True,
                                     linestyle_lick = 'solid',
                                     turn_ms_into_s = False,
                                     axsize = (0.8, 0.56),
                                     save_fig = False,
                                     save_png = False,
                                     fig_path =''):
    if turn_ms_into_s:
        scale_factor = 1/1000
        binsize = 100
    else:
        scale_factor = 1
        binsize = 0.1
    trials = ['reward', 'omission', ]
    lick_DA_psth_fig, lick_DA_psth_ax = plt.subplots(2,len(windows_subset), sharey = 'row', sharex = True)
    for ax, trial in enumerate(trials):
        if trial == 'reward':
            color_DA_PSTH = 'dodgerblue' #'#ad2472'
            color_lick_PSTH = 'dodgerblue' #ad2472'
            linestyle_DA = 'dashdot'
        else:
            color_DA_PSTH = 'darkblue' #'#5a9c43'
            color_lick_PSTH = 'darkblue' #'#5a9c43'
            linestyle_DA = 'dashed'

        animal_day_df = trial_df[trial_df['animal'] == animal].copy()
        animal_day_df = animal_day_df[animal_day_df['trial_type'] == trial].reset_index().copy()

        reward_time = (animal_day_df['reward_time'] - animal_day_df['cue_on'] ).mean() * scale_factor
        if align_to_cue:
            cue_on =  0
            cue_off = (animal_day_df['cue_off'] - animal_day_df['cue_on'] ).mean() * scale_factor
            reward_time = (animal_day_df['reward_time'] - animal_day_df['cue_on'] ).mean() * scale_factor
        else:
            cue_on = (animal_day_df['cue_on'] - animal_day_df['reward_time'] ).mean() * scale_factor
            cue_off = (animal_day_df['cue_off'] - animal_day_df['reward_time'] ).mean() * scale_factor
            reward_time = 0
        for w, window in enumerate(windows_subset):
            subset_animal_day_df = animal_day_df.iloc[window[0]:window[1]].copy()
            lick_times_list = (subset_animal_day_df['licks_all'] * scale_factor).tolist()
            if align_to_cue:
                lick_times_list = [trial_licks + reward_time for trial_licks in lick_times_list]
            lick_hist_dict = lpf.getLickPSTH(lick_times_list, binsize = binsize, total_time_window= 15,
                                       baseline_period = 7, in_seconds = plot_in_sec, time_window_in_seconds = True)
            if smooth_lick:
                lick_hist_mean = gaussian_filter1d(lick_hist_dict['mean_hist'], sigma = smooth_lick)
                lick_hist_sem = gaussian_filter1d(lick_hist_dict['sem_hist'], sigma = smooth_lick)
            lick_DA_psth_ax[0, w].plot(lick_hist_dict['bins'][1:], lick_hist_mean, color = color_lick_PSTH, linewidth = stroke_PSTH, linestyle = linestyle_lick)
            lick_DA_psth_ax[0, w].fill_between(lick_hist_dict['bins'][1:], lick_hist_mean - lick_hist_sem, lick_hist_mean + lick_hist_sem,
                                               facecolor= color_lick_PSTH, alpha = alpha_PSTH_error )
            lick_DA_psth_ax[0, w].set_ylim(ylim_lick)
            lick_DA_psth_ax[0, 0].set_ylabel('lick (Hz)',  fontsize = fontsize_label)
            lick_DA_psth_ax[0, w].axvline(x = reward_time, color ='black', linestyle='solid', linewidth =  0.25 )
            lick_DA_psth_ax[0, w].axhline(y = 0, color ='gray', linestyle='dashed', linewidth =  linewidth_0_DA, alpha = alpha0line  )
            lick_DA_psth_ax[0, w].axvspan(cue_on, cue_off, alpha = alpha_cue_shade, facecolor = color_cue_shade, linewidth = None)
            lick_DA_psth_ax[0, w].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
            lick_DA_psth_ax[0, w].yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
            lick_DA_psth_ax[0, w].set_title(f'trials\n{window}')
            epoch_dff_times = np.mean(np.array(subset_animal_day_df['epoch_time'].to_list()), axis = 0) * scale_factor

            epoch_dff_times = epoch_dff_times + reward_time
            epoch_dff = np.array(subset_animal_day_df['epoch_dff'].to_list())
            dlight_ylabel = 'dLight\ndF/F (%)'
            epoch_dff = [x - np.mean(x[((epoch_dff_times >= -2) & (epoch_dff_times <0))]) for x in epoch_dff ]
            epoch_dff = [x[((epoch_dff_times >= -2) & (epoch_dff_times <=5))] for x in epoch_dff ]

            epoch_dff_times =epoch_dff_times[((epoch_dff_times >= -2) & (epoch_dff_times <=5))]
            epoch_dff_mean = np.mean(epoch_dff, axis = 0)
            epoch_dff_sem = stats.sem(epoch_dff, axis = 0)
            lick_DA_psth_ax[1, w].axhline(0, linestyle='dashed', color = 'gray', linewidth = 0.5, )
            lick_DA_psth_ax[1, w].axvline(x = reward_time, color ='black', linestyle='solid', linewidth =  0.25 )
            lick_DA_psth_ax[1, w].axhline(y = 0, color ='gray', linestyle='dashed', linewidth = linewidth_0_DA, alpha = alpha0line  )
            lick_DA_psth_ax[1, w].axvspan(cue_on, cue_off, alpha = alpha_cue_shade, facecolor = color_cue_shade, linewidth = None)

            lick_DA_psth_ax[1, w].plot(epoch_dff_times, epoch_dff_mean, color = color_DA_PSTH, linewidth = stroke_PSTH, linestyle = linestyle_DA)
            lick_DA_psth_ax[1, w].fill_between(epoch_dff_times, epoch_dff_mean - epoch_dff_sem, epoch_dff_mean + epoch_dff_sem,
                                               facecolor= color_DA_PSTH, alpha = alpha_PSTH_error )

            lick_DA_psth_ax[1, 0].set_ylabel(f'{dlight_ylabel}', fontsize = fontsize_label)
            lick_DA_psth_ax[0, 0].set_xlim([-2, 5])
            lick_DA_psth_fig.supxlabel('time from cue onset (s)', fontsize = fontsize_label)

            for ax in lick_DA_psth_ax.reshape(-1):
               standardize_plot_graphics(ax)
            set_ax_size_inches(axsize[0], axsize[1], lick_DA_psth_ax, print_new_sizes=False)

            if save_fig:
                if save_png:
                    lick_DA_psth_fig.savefig(os.path.join(fig_path, f'example_omissions_PSTHs-{animal}.png'),
                                        bbox_inches = 'tight',
                                        dpi = 600,
                                        transparent = True
                                        )
                lick_DA_psth_fig.savefig(os.path.join(fig_path,  f'example_omissions_PSTHs-{animal}.pdf'),
                                    bbox_inches = 'tight',
                                    transparent = True
                                    )
def plot_cued_reward_and_background_reward_consumption_PSTH(trial_df,
                                                             color_lick_PSTH= {'bgd_reward':'black',
                                                                               'reward': '#605656'},
                                                             stroke_PSTH = 0.5,
                                                             alpha_PSTH_error = 0.3,
                                                             fontsize_label = 7,
                                                             smooth_lick = 0.5,
                                                             linewidth_0_lick = 0,
                                                             linewidth_0_DA = 0,
                                                             alpha0line = 0.5,
                                                             align_to_cue = True,
                                                             plot_in_sec = True,
                                                             plot_individuals = False,
                                                             linestyle_lick = 'solid',
                                                             turn_ms_into_s = False,
                                                             axsize = (1.69, 1),
                                                             save_fig = False,
                                                             save_png = False,
                                                             fig_path =''):
    if turn_ms_into_s:
        scale_factor = 1/1000
        binsize = 100
    else:
        scale_factor = 1
        binsize = 0.1

    df_milk = trial_df[((trial_df['condition'] == '600s-bgdmilk')
                          & (trial_df['day_num'] <=8)) ].copy()


    label = {'bgd_reward':'milk', 'reward': 'sucrose'}
    trial_types = df_milk['trial_type'].unique()
    fig_all, ax_all = plt.subplots()
    if plot_individuals:
        fig_individuals, ax_individuals = plt.subplots(1,len(df_milk['animal'].unique()), sharey= True, sharex = True, squeeze = False)
    for trial_type in trial_types:
        animals_by_condition = df_milk[df_milk['trial_type'] ==trial_type]['animal'].unique()
        lick_hist_mean_all = np.empty((len(animals_by_condition),90))
        lick_hist_mean_all[:] = np.nan
        for a, animal in enumerate(animals_by_condition):
            animal_df = df_milk[((df_milk['animal'] ==animal) & (df_milk['trial_type'] ==trial_type))]
            lick_times_list = (animal_df['licks_all'] * scale_factor).tolist()
            lick_hist_dict = lpf.getLickPSTH(lick_times_list, binsize = binsize, total_time_window= 10,
                                       baseline_period = 0, in_seconds = True,  time_window_in_seconds = True)

            if smooth_lick > 10000:
                lick_hist_mean = gaussian_filter1d(lick_hist_dict['mean_hist'], sigma = smooth_lick)
                lick_hist_sem= gaussian_filter1d(lick_hist_dict['sem_hist'], sigma = smooth_lick)
            else:
                lick_hist_mean = lick_hist_dict['mean_hist']
                lick_hist_sem = lick_hist_dict['sem_hist']
            lick_hist_mean_all[a,:] = lick_hist_mean
            if plot_individuals:
                ax_individuals[0, a].plot(lick_hist_dict['bins'][1:], lick_hist_mean, color = color_lick_PSTH[trial_type], linewidth = stroke_PSTH, linestyle = linestyle_lick)
                ax_individuals[0, a].fill_between(lick_hist_dict['bins'][1:], lick_hist_mean - lick_hist_dict['sem_hist'], lick_hist_mean + lick_hist_dict['sem_hist'],
                                                    facecolor= color_lick_PSTH[trial_type], alpha = alpha_PSTH_error, label = label[trial_type] )

        mean_all = np.nanmean(lick_hist_mean_all, axis = 0)
        mean_all = mean_all[1:]
        sem_all = stats.sem(lick_hist_mean_all, axis = 0, ddof = 1)
        sem_all = sem_all[1:]
        if smooth_lick:
            mean_all = gaussian_filter1d(mean_all, sigma = smooth_lick)
            sem_all= gaussian_filter1d(sem_all, sigma = smooth_lick)
        ax_all.plot(lick_hist_dict['bins'][2:], mean_all, color = color_lick_PSTH[trial_type], linewidth = stroke_PSTH, linestyle = linestyle_lick)
        ax_all.fill_between(lick_hist_dict['bins'][2:], mean_all - sem_all, mean_all + sem_all, facecolor= color_lick_PSTH[trial_type], alpha = alpha_PSTH_error, label = label[trial_type] )
    ax_all.legend()
    ax_all.set_ylim([0, None])
    ax_all.set_ylabel('lick rate (Hz)')
    ax_all.set_xlabel('time from reward delivery')
    standardize_plot_graphics(ax_all)
    set_ax_size_inches(axsize[0], axsize[1],  ax_all)
    if plot_individuals:
        ax_individuals[0,0].legend()
        ax_individuals[0,0].set_ylabel('lick rate (Hz)')
        ax_individuals[0,0].set_xlabel('time from reward delivery')
        standardize_plot_graphics(ax_individuals)
        set_ax_size_inches(axsize[0], axsize[1],  ax_individuals)

    if save_fig:
        fig_all.savefig(os.path.join(fig_path, 'lick psth milk vs sucrose.pdf'), transparent = True, bbox_inches = 'tight', )

    return ax_all

def plot_consumption_PSTH_beginning_end(trial_df,
                                        condition_to_plot,
                                        num_to_subset = 6,
                                        color_lick_PSTH= {'beginning':'k',
                                                          'end': 'red'},
                                        stroke_PSTH = 0.5,
                                        alpha_PSTH_error = 0.3,
                                        fontsize_label = 7,
                                        smooth_lick = 0.75,
                                        linewidth_0_lick = 0,
                                        linewidth_0_DA = 0,
                                        alpha0line = 0.5,
                                        align_to_cue = True,
                                        plot_in_sec = True,
                                        plot_individuals = False,
                                        linestyle_lick = 'solid',
                                        turn_ms_into_s = False,
                                        axsize = (1.69, 1),
                                        save_fig = False,
                                        save_png = False,
                                        fig_path ='',
                                        ):
    if turn_ms_into_s:
        scale_factor = 1/1000
        binsize = 100
    else:
        scale_factor = 1
        binsize = 0.1
    df_single_condition = trial_df[trial_df['condition']==condition_to_plot].copy()

    conditions = ['beginning', 'end']

    label = {'beginning':f'first {num_to_subset} trials/session', 'end': f'last {num_to_subset} trials/session'}
    animals = df_single_condition['animal'].unique()
    max_trials = int(df_single_condition['trial_num'].max())
    end_trials_cutoff = max_trials - num_to_subset
    fig, ax = plt.subplots()
    #ax.axhline(0, color = 'gray', linewidth = 1, linestyle = (0,(4,2)))
    for condition in conditions:
        if condition == 'beginning':
            df_condition = df_single_condition[df_single_condition['trial_num'] <= num_to_subset].copy()
        elif condition == 'end':
            df_condition = df_single_condition[df_single_condition['trial_num'] > end_trials_cutoff].copy()
        lick_hist_mean_all = np.empty((len(df_single_condition['animal'].unique()),90))
        lick_hist_mean_all[:] = np.nan
        for a, animal in enumerate(animals):
            animal_df = df_condition[df_condition['animal'] ==animal]
            lick_times_list = (animal_df['licks_all'] * scale_factor).tolist()
            lick_hist_dict = lpf.getLickPSTH(lick_times_list, binsize = binsize, total_time_window= 10,
                                       baseline_period = 0, in_seconds = True, time_window_in_seconds = True)

            lick_hist_mean = lick_hist_dict['mean_hist']
            lick_hist_sem = lick_hist_dict['sem_hist']
            lick_hist_mean_all[a,:] = lick_hist_mean

        mean_all = np.nanmean(lick_hist_mean_all, axis = 0)
        mean_all = mean_all[1:]
        sem_all = stats.sem(lick_hist_mean_all, axis = 0, ddof = 1)
        sem_all =sem_all[1:]
        if smooth_lick:
            mean_all = gaussian_filter1d(mean_all, sigma = smooth_lick)
            sem_all = gaussian_filter1d(sem_all, sigma = smooth_lick)
        ax.plot(lick_hist_dict['bins'][2:], mean_all, color = color_lick_PSTH[condition], linewidth = stroke_PSTH, linestyle = linestyle_lick)
        ax.fill_between(lick_hist_dict['bins'][2:], mean_all - sem_all, mean_all + sem_all, facecolor= color_lick_PSTH[condition], alpha = alpha_PSTH_error, label = label[condition] )
    ax.legend()
    ax.set_ylabel('lick rate (Hz)')
    ax.set_xlabel('time from reward delivery')
    standardize_plot_graphics(ax)
    set_ax_size_inches(axsize[0], axsize[1],  ax)
    if save_fig:
        fig.savefig(os.path.join(fig_path, f'lick psth begin end session{condition_to_plot}.pdf'), transparent = True, bbox_inches = 'tight', )
    return ax


def standardize_plot_graphics(ax):

    if isinstance(ax, np.ndarray):
        ax_list = list(ax.reshape(-1))
    else: #if ax is single ax instance
        ax_list = [ax]
    for single_ax in ax_list:
        [i.set_linewidth(0.35) for i in single_ax.spines.values()]
        single_ax.spines['right'].set_visible(False)
        single_ax.spines['top'].set_visible(False)
    return ax

def change_twinx_together(current_y_left, current_y_right, new_y, new_y_is_left = True):
    # current_scale_factor_pos= current_y_left[1]/current_y_right[1]
    # current_scale_factor_neg= current_y_left[0]/current_y_right[0]

    # current_left_range = abs(current_y_left[0]) + abs(current_y_left[1])
    # current_right_range = abs(current_y_right[0]) + abs(current_y_right[1])
    if new_y_is_left:
        print(new_y)

        new_y =[x if x is not None else y for x, y in zip(new_y, current_y_left)]


        left_ax_new_post_scale_factor_pos = new_y[1]/current_y_left[1]
        left_ax_new_post_scale_factor_neg = new_y[0]/current_y_left[0]
        ax_to_change_top = current_y_right[1] * left_ax_new_post_scale_factor_pos
        ax_to_change_bottom = current_y_right[0] * left_ax_new_post_scale_factor_neg
    else:
        new_y =[x if x is not None else y for x, y in zip(new_y, current_y_right)]

        right_ax_new_post_scale_factor_pos = new_y[1]/current_y_right[1]
        right_ax_new_post_scale_factor_neg = new_y[0]/current_y_right[0]
        ax_to_change_top = current_y_left[1] * right_ax_new_post_scale_factor_pos
        ax_to_change_bottom = current_y_left[0] * right_ax_new_post_scale_factor_neg

    return [ax_to_change_bottom, ax_to_change_top]

def twinx_align0(current_y_left, current_y_right, template_ax_is_left = True, expand_neg = True):

    if template_ax_is_left:
        template_ax = current_y_left
        ax_to_change = current_y_right
    else:
        template_ax = current_y_right
        ax_to_change = current_y_left
    if expand_neg:
        ax_to_change_top = ax_to_change[1]
        ax_to_change_bottom = (ax_to_change[1]/template_ax[1]) * template_ax [0]
    else:
        ax_to_change_top = (template_ax[1]/template_ax[0]) * ax_to_change [0]
        ax_to_change_bottom = ax_to_change[0]

    return [ax_to_change_bottom, ax_to_change_top]


def set_ax_size_inches(width, height, ax, subplot_hspace = 0.1, subplot_wspace = 0.1, print_new_sizes = False):


    if isinstance(ax, np.ndarray):
        try: #if ax is 2d array
            test_ax = ax[0,0]
        except: #if ax is 1d array
            test_ax = ax[0]
        num_row = test_ax.get_gridspec().nrows
        num_col = test_ax.get_gridspec().ncols
    else: #if ax is single ax instance
        num_row = 1
        num_col = 1
        test_ax = ax
    fig = test_ax.get_figure()
    fig.set_constrained_layout(False)
    if fig.get_layout_engine() is None:
        fig.subplots_adjust(hspace = subplot_hspace, wspace = subplot_wspace)
    elif fig.get_layout_engine().adjust_compatible:
        fig.subplots_adjust(hspace = subplot_hspace, wspace = subplot_wspace)
    if fig._suptitle is not None:
        pass
    if fig._supxlabel is not None:
        pass#get extra padding
    if fig._supylabel is not None:
        pass
    # get padding parameters from plot
    l_pad = test_ax.figure.subplotpars.left
    b_pad = test_ax.figure.subplotpars.bottom
    r_pad = test_ax.figure.subplotpars.right
    t_pad = test_ax.figure.subplotpars.top
    wpsace_pad = test_ax.figure.subplotpars.wspace
    hspace_pad = test_ax.figure.subplotpars.hspace
    if print_new_sizes:
        print(l_pad, b_pad, r_pad, t_pad, wpsace_pad, hspace_pad)

    #total ax area
    axes_width = (num_col * width) + ((num_col - 1) * (wpsace_pad * width))
    axes_height = (num_row * height) + ((num_row - 1) * (hspace_pad * height))

    fig_width = float(axes_width)/(r_pad-l_pad)
    fig_height = float(axes_height)/(t_pad-b_pad)
    fig.set_size_inches(fig_width, fig_height)
    test_ax.set_position
    if print_new_sizes:
        print(fig_width)
        print(fig_height)

def plot_with_error(ax,
                    x,
                    y_mean,
                    y_sem = 0,
                    label = '',
                    color = 'black',
                    linewidth = 0.35,
                    elinewidth = 0.25,
                    linestyle = 'solid',
                    marker = 'o',
                    markersize = 5,
                    markeredgewidth = 0.5,
                    alpha=1.0,
                    alpha_error=0.3,
                    plot_error=True,
                    shaded_error = True,
                    plot_individuals = False,
                    df_individuals = None,
                    linewidth_individuals = 0.25,
                    alpha_individuals = 0.6,
                    ):
    if not plot_error:
        elinewidth = 0
        y_sem = 0
        alpha_error = 0
    if shaded_error:
        ax.plot(x, y_mean, label=label, color=color, linewidth=linewidth,
                linestyle=linestyle, marker=marker, markersize=markersize,
                markeredgewidth=markeredgewidth, alpha=alpha)
        ax.fill_between(x, y_mean - y_sem, y_mean + y_sem,
                        facecolor=color, alpha=alpha_error, linewidth=0)
    else:
        ax.errorbar(x, y_mean, yerr=y_sem, color=color,
                    marker=marker, label=label,
                    markersize=markersize, markeredgewidth=markeredgewidth,
                    linewidth=linewidth, linestyle=linestyle,
                    elinewidth=elinewidth)
    if plot_individuals:
        plot_individual_traces(ax,
                                x,
                                df_individuals,
                                color = color,
                                linewidth_individuals=linewidth_individuals,
                                linestyle = linestyle,
                                alpha_individuals= alpha_individuals,
                                )
    return ax



def plot_individual_traces(ax,
                           x,
                           df_individuals,
                           color = 'black',
                           linewidth_individuals=0.35,
                           linestyle = 'solid',
                           alpha_individuals= 0.6,
                           ):

    ax.plot(x, df_individuals, label='_nolegend_',
            color=color, linewidth=linewidth_individuals, alpha=alpha_individuals, linestyle=linestyle)

def prepare_ax_and_save_fig(ax,
                            axsize = (1,1),
                            title = '',
                            save_fig = False,
                            save_png = False,
                            fig_path = '',
                            transparent = True,
                            dpi = 600,
                            bbox_extra_artists = [],
                            ):
    fig = ax.gcf()
    standardize_plot_graphics(ax)
    new_title = lpf.cleanStringForFilename(title)
    if len(axsize)>2:
        set_ax_size_inches(axsize[0],
                               axsize[1],
                               ax,
                               axsize[2],
                               axsize[3],
                               )
    else:
        set_ax_size_inches(axsize[0],
                               axsize[1],
                               ax,
                               )
    if save_fig:
        if save_png:
            fig.savefig(os.path.join(fig_path, f'{new_title}.png'),
                                bbox_inches = 'tight',
                                dpi = dpi,
                                transparent = transparent,
                                )
        fig.savefig(os.path.join(fig_path, f'{new_title}.pdf'),
                            bbox_inches = 'tight',
                            transparent = transparent,
                            )



def plotBarsFromDict(data_by_condition,
                     condition_colors,
                     order_to_plot = [],
                     plot_individuals = True,
                     plot_error = True,
                     ylim =[None,None],
                     bar_alpha = 0.3,
                     markersize = 2.1,
                     plot_sem = True,
                     plot_median_and_IQR= False,
                     bar_width = 0.8,
                     ylabel ='',
                     jitter_individuals = True,
                     jitter_width = 20,
                     logscale = False,
                     save_fig = False,
                     save_png = False,
                     fig_path ='',
                     title='',
                     data_is_nested_dict = True,
                     data_is_regular_dict = False,
                     data_is_df = False,
                     axsize = (0.64, 1),
                     plot_stats = True,
                     linewidth_error = 0.5,
                     color_error = 'k',
                     linewidth_stats_lines = 0.5,
                     stats_assume_equal_var = False,
                     ax_to_plot = None,
                     linearLocator = False,
                     maxNLocator = False,
                     save_stats = False
                     ):
    plot_title = title
    if (not (data_is_df or data_is_regular_dict or data_is_nested_dict)):
        raise Exception('Unclear what format data is in, check data_is_df, data_is_regular_dict, or data_is_nested_dict flags')
    elif data_is_nested_dict:
        data_conditions = {x: list(data_by_condition[x].values()) for x in data_by_condition}

    elif data_is_regular_dict:
        data_conditions = data_by_condition

    elif data_is_df:
        data_conditions = {x: list(data_by_condition.loc[x,:]) for x in data_by_condition.index.get_level_values('condition')}
    num_bars = len(data_conditions)
    if plot_stats or save_stats:

        anova_df = pd.DataFrame({'condition' : np.repeat(list(data_conditions.keys()), [len(v) for k, v in data_conditions.items()]),
                                 'data' : list(itertools.chain.from_iterable(data_conditions.values())), })
        anova_df = anova_df[anova_df['data'].astype(bool)]

        anova_df["data"] = pd.to_numeric(anova_df["data"])

    conditions = list(data_conditions.keys())
    conditions =  lpf.sort_list_by_key(conditions, order_to_plot)
    if logscale:
        bottom = 1
    else:
        bottom = None
    if ax_to_plot is not None:
        bar_ax = ax_to_plot
        bar_fig = bar_ax.get_figure()
    else:
        bar_fig, bar_ax = plt.subplots(figsize =(2,3))
    xtick_labels =[]
    jitter_x_dict ={}
    for con_idx, condition in enumerate(conditions):
        data_conditions[condition] = [x for x in data_conditions[condition] if type(x) is not list]
        if plot_median_and_IQR:
            condition_mean = np.nanmedian(data_conditions[condition])
            IQR = np.array([[condition_mean - np.nanquantile(data_conditions[condition], 0.25)], [np.nanquantile(data_conditions[condition], 0.75)- condition_mean]])
            error = IQR
            mean_error_labels = f'{condition}\n{round(condition_mean, 2)}  [{round(np.nanquantile(data_conditions[condition], 0.25), 1)}, {round(np.nanquantile(data_conditions[condition], 0.75), 1)}]'
            xtick_labels.append(mean_error_labels)
        else:
            condition_mean = np.nanmean(data_conditions[condition])
            if plot_sem:
                condition_error = stats.sem(data_conditions[condition], ddof =1, nan_policy = 'omit')
            else:
                condition_error = np.nanstd(data_conditions[condition], ddof =1)
            mean_error_labels = f'{condition}\n{round(condition_mean, 2)} ± {round(condition_error, 2)}'
            xtick_labels.append(mean_error_labels)
            if plot_error:
                error = condition_error
        #bar plots height from bottom so if bottom set to 1 for log scale need to subtract to plot correctly
        if bottom is not None:
            bar_height_mean = condition_mean - bottom
        else:
            bar_height_mean = condition_mean
        bar_ax.bar(con_idx, bar_height_mean, yerr = error, width = bar_width,
                   color = condition_colors[condition], linewidth = 0, alpha = bar_alpha, ecolor = color_error,
                   error_kw = {'elinewidth': linewidth_error}, bottom = bottom)
        # if plot_median_and_IQR:
        #     bar_ax.errorbar(con_idx, bar_height_mean, yerr = IQR, markersize = 0)

        if plot_individuals:
            if jitter_individuals:
                x_scatter = [con_idx] * len(data_conditions[condition])
                if (ylabel == 'lick rate to cue (Hz)\ntrials 36 - 40'
                    and condition == '60s'
                    ):
                    jittered_x = [0.05, 0.14, 0.07, 0.18, -0.06, -0.16, -0.18, -0.16,
                                  0.03, -0.06, -0.14, 0, 0.11, 0.06, -0.11, -0.05, 0.11,
                                  0.02, 0.19
                                  ]
                    jittered_x = [x + con_idx
                                  for x
                                  in jittered_x
                                  ]
                elif (ylabel == 'lick rate to cue (Hz)\ntrials 36 - 40'
                      and condition == '600s'
                      ):
                    jittered_x = [0.14, -0.16, -0.12, 0.15, -0.15, 0.08, -0.06, 0.06,
                                  -0.01, -0.08, -0.19, 0.0, 0.01, 0.02, 0.0, 0.09, 0.16,
                                  0.19, 0.07
                                  ]
                    jittered_x = [x + con_idx
                                  for x
                                  in jittered_x
                                  ]
                else:
                    jittered_x = [x + random.randrange(-jitter_width, jitter_width, 1)/100 for x in x_scatter]
                jitter_x_dict[condition] = jittered_x
            bar_ax.scatter(jittered_x, data_conditions[condition], marker = 'o', facecolor = condition_colors[condition],
                           s = markersize**2, linewidth = 0, zorder = 7 )

    if logscale:
        bar_ax.set_yscale('log')
        bar_ax.set_ylim([1,None])
        bar_ax.set_ylim(ylim)
        bar_ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

    else:
        bar_ax.set_ylim(ylim)
        if linearLocator:
            bar_ax.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(3))
        elif maxNLocator:
            bar_ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        bar_ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
    bar_ax.set_ylabel(ylabel)
    bar_ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.arange(len(conditions))))
    bar_ax.set_xticklabels(xtick_labels, rotation = 35, ha ='center')
    new_title = lpf.cleanStringForFilename(ylabel)
    data_list = list(data_conditions.values())

    if plot_stats or save_stats:
        if num_bars == 2:
            f_test = sf.FTest(x = data_list[0], y = data_list[1])
            t_test = stats.ttest_ind(a = data_list[0], b =data_list[1], equal_var = stats_assume_equal_var)
            t_test_pingouin = pingouin.ttest(x = data_list[0], y = data_list[1], correction = True)
            t_test_pingouin['scipy_stats_T'] = t_test[0]
            t_test_pingouin['scipy_stats_pval'] = t_test[1]
            t_test_pingouin['FTest_F'] = f_test['f_stat']
            t_test_pingouin['FTest_pval'] = f_test['p_val']
            t_test_pingouin['FTest_ddof1'] = f_test['ddof1']
            t_test_pingouin['FTest_ddof2'] = f_test['ddof2']
            t_test_pingouin[f'mean_{conditions[0]}'] = np.nanmean(data_conditions[conditions[0]])
            t_test_pingouin[f'sem_{conditions[0]}'] =stats.sem(data_conditions[conditions[0]], ddof = 1, nan_policy = 'omit')
            t_test_pingouin[f'var_{conditions[0]}'] = np.nanvar(data_conditions[conditions[0]], ddof = 1)
            t_test_pingouin[f'mean_{conditions[1]}'] = np.nanmean(data_conditions[conditions[1]])
            t_test_pingouin[f'sem_{conditions[1]}'] =stats.sem(data_conditions[conditions[1]], ddof = 1, nan_policy = 'omit')
            t_test_pingouin[f'var_{conditions[1]}'] = np.nanvar(data_conditions[conditions[1]], ddof = 1)
            if plot_stats:
                title = f'mean: p = {np.round(t_test[1], 4)} (t= {round(t_test[0], 4)})\nvariance: p = {round(f_test["p_val"], 4)} (F= {round(f_test["f_stat"], 4)})'
                bar_ax.set_title(title)
                ylim_upper = bar_ax.get_ylim()[1]
                bar_ax.plot([0,1], [ylim_upper,ylim_upper], color = 'k', linewidth = linewidth_stats_lines, markersize = 0)
            if save_stats:
                stats_fig_path = os.path.join(fig_path, 'stats')
                Path(stats_fig_path).mkdir(parents=False, exist_ok=True)
                t_test_pingouin.to_csv(os.path.join(stats_fig_path, f'Stats- {new_title}_'+ '_'.join(conditions)+ f'_{plot_title}' +'.csv'))

        # elif num_bars > 2:
        #     pass
        #     # if stats_assume_equal_var:
        #     #     anova_pingouin = pingouin.anova(dv = 'data', between = 'condition', data = anova_df)
        #     #     anova_scipy = stats.f_oneway(*data_list)
        #     #     anova_pingouin['scipy_stats_T'] = anova_scipy[0]
        #     #     anova_pingouin['scipy_stats_pval'] = anova_scipy[1]
        #     # else:
        #     #     anova_pingouin = pingouin.welch_anova(dv = 'data', between = 'condition', data = anova_df)

        #     anova_pingouin = pd.DataFrame()

        #     for condition in conditions:
        #         anova_pingouin[f'mean_{condition}'] = np.nanmean(data_conditions[condition])
        #         anova_pingouin[f'sem_{condition}'] =stats.sem(data_conditions[condition], ddof = 1, nan_policy = 'omit')
        #         anova_pingouin[f'var_{condition}'] = np.nanvar(data_conditions[condition], ddof = 1)

        #     if plot_stats:
        #         title = f"mean: p = {np.round(anova_pingouin['p-unc'].iloc[0], 4)} (F= {round(anova_pingouin.F.iloc[0], 4)})"
        #         bar_ax.set_title(title)
        #         ylim_upper = bar_ax.get_ylim()[1]
        #         bar_ax.plot([0, len(conditions)-1], [ylim_upper,ylim_upper], color = 'k', linewidth = linewidth_stats_lines, markersize = 0)
        #     if save_stats:
        #         stats_fig_path =os.path.join(fig_path, 'stats')
        #         Path(stats_fig_path).mkdir(parents=False, exist_ok=True)
        #         anova_pingouin.to_csv(os.path.join(stats_fig_path, f'Stats- {new_title}_'+ '_'.join(conditions)+ f'_{plot_title}' +'.csv'))

    # print(bar_ax.figure.subplotpars.right)
    # print(bar_ax.figure.subplotpars.left)
    if axsize is not None:
        set_ax_size_inches(axsize[0], axsize[1], bar_ax, print_new_sizes=False)
        # print(bar_ax.figure.subplotpars.right)
        # print(bar_ax.figure.subplotpars.left)
    standardize_plot_graphics(bar_ax)

    if save_fig:
        if save_png:
            bar_fig.savefig(os.path.join(fig_path, f'{new_title}_'+ '_'.join(conditions) + f'_{plot_title}' +'.png'),
                            dpi = 600,
                            transparent = True,
                            bbox_inches = 'tight')
        bar_fig.savefig(os.path.join(fig_path, f'{new_title}_'+ '_'.join(conditions)+ f'_{plot_title}'+'.pdf'),
                        transparent = True,
                        bbox_inches = 'tight')

    if (plot_stats or save_stats) and num_bars == 2:
        return {'fig':bar_fig, 'ax':bar_ax, 'jitter_x_dict': jitter_x_dict, 'data_conditions':data_conditions, 'p-val': t_test[1] } # 't_test': t_test, 'f_test': f_test }
    else:
        return {'fig':bar_fig, 'ax':bar_ax, }

def plotBoxplotFromDict(data_by_condition,
                         condition_colors,
                         order_to_plot = [],
                         plot_individuals = True,
                         plot_error = True,
                         ylim =[None,None],
                         box_alpha = 0.2,
                         markersize = 2.1,
                         plot_sem = True,
                         plot_median_and_IQR= False,
                         box_width = 0.8,
                         ylabel ='',
                         jitter_individuals = True,
                         jitter_width = 20,
                         logscale = False,
                         save_fig = False,
                         save_png = False,
                         fig_path ='',
                         title='',
                         data_is_nested_dict = True,
                         data_is_regular_dict = False,
                         data_is_df = False,
                         axsize = (0.64, 1),
                         plot_stats = True,
                         linewidth_error = 0.5,
                         color_error = 'k',
                         linewidth_stats_lines = 0.5,
                         stats_assume_equal_var = False,
                         ax_to_plot = None,
                         linearLocator = False,
                         maxNLocator = False,
                         save_stats = False
                         ):
    plot_title = title
    if (not (data_is_df or data_is_regular_dict or data_is_nested_dict)):
        raise Exception('Unclear what format data is in, check data_is_df, data_is_regular_dict, or data_is_nested_dict flags')
    elif data_is_nested_dict:
        data_conditions = {x: list(data_by_condition[x].values()) for x in data_by_condition}

    elif data_is_regular_dict:
        data_conditions = data_by_condition

    elif data_is_df:
        data_conditions = {x: list(data_by_condition.loc[x,:]) for x in data_by_condition.index.get_level_values('condition')}
    num_box = len(data_conditions)
    if plot_stats or save_stats:

        anova_df = pd.DataFrame({'condition' : np.repeat(list(data_conditions.keys()), [len(v) for k, v in data_conditions.items()]),
                                 'data' : list(itertools.chain.from_iterable(data_conditions.values())), })
        anova_df = anova_df[anova_df['data'].astype(bool)]

        anova_df["data"] = pd.to_numeric(anova_df["data"])

    conditions = list(data_conditions.keys())
    conditions =  lpf.sort_list_by_key(conditions, order_to_plot)
    if logscale:
        bottom = 1
    else:
        bottom = None
    if ax_to_plot is not None:
        box_ax = ax_to_plot
        box_fig = box_ax.get_figure()
    else:
        box_fig, box_ax = plt.subplots(figsize =(2,3))
    xtick_labels =[]
    jitter_x_dict ={}
    for con_idx, condition in enumerate(conditions):
        data_conditions[condition] = [x for x in data_conditions[condition] if type(x) is not list]
        if plot_median_and_IQR:
            condition_mean = np.nanmedian(data_conditions[condition])
            IQR = np.array([[condition_mean - np.nanquantile(data_conditions[condition], 0.25)], [np.nanquantile(data_conditions[condition], 0.75)- condition_mean]])
            error = IQR
            mean_error_labels = f'{condition}\n{round(condition_mean, 2)}  [{round(np.nanquantile(data_conditions[condition], 0.25), 1)}, {round(np.nanquantile(data_conditions[condition], 0.75), 1)}]'
            xtick_labels.append(mean_error_labels)
        else:
            condition_mean = np.nanmean(data_conditions[condition])
            if plot_sem:
                condition_error = stats.sem(data_conditions[condition], ddof =1, nan_policy = 'omit')
            else:
                condition_error = np.nanstd(data_conditions[condition], ddof =1)
            mean_error_labels = f'{condition}\n{round(condition_mean, 2)} ± {round(condition_error, 2)}'
            xtick_labels.append(mean_error_labels)
            if plot_error:
                error = condition_error
        #box plots height from bottom so if bottom set to 1 for log scale need to subtract to plot correctly
        if bottom is not None:
            box_height_mean = condition_mean - bottom
        else:
            box_height_mean = condition_mean
        boxplot =box_ax.boxplot(data_conditions[condition], notch = False,# whis = [0, 100],
                   patch_artist=True,
                   positions = [con_idx],
                   widths = [box_width],
                   medianprops = {'color': 'k', 'linewidth': 0.5},
                   whiskerprops ={'linewidth': 0.25},
                   capprops={'linewidth': 0},
                   showcaps = False,
                   showfliers = False)  # fill with color)
        for patch in boxplot['boxes']:
            patch.set_facecolor(condition_colors[condition])
            patch.set_linewidth(0)
            patch.set_alpha(box_alpha)
        # if plot_median_and_IQR:
        #     box_ax.errorbar(con_idx, box_height_mean, yerr = IQR, markersize = 0)

        if plot_individuals:
            if jitter_individuals:
                x_scatter = [con_idx] * len(data_conditions[condition])
                if (ylabel == 'lick rate to cue (Hz)\ntrials 36 - 40'
                    and condition == '60s'
                    ):
                    jittered_x = [0.05, 0.14, 0.07, 0.18, -0.06, -0.16, -0.18, -0.16,
                                  0.03, -0.06, -0.14, 0, 0.11, 0.06, -0.11, -0.05, 0.11,
                                  0.02, 0.19
                                  ]
                    jittered_x = [x + con_idx
                                  for x
                                  in jittered_x
                                  ]
                elif (ylabel == 'lick rate to cue (Hz)\ntrials 36 - 40'
                      and condition == '600s'
                      ):
                    jittered_x = [0.14, -0.16, -0.12, 0.15, -0.15, 0.08, -0.06, 0.06,
                                  -0.01, -0.08, -0.19, 0.0, 0.01, 0.02, 0.0, 0.09, 0.16,
                                  0.19, 0.07
                                  ]
                    jittered_x = [x + con_idx
                                  for x
                                  in jittered_x
                                  ]
                else:
                    jittered_x = [x + random.randrange(-jitter_width, jitter_width, 1)/100 for x in x_scatter]
                jitter_x_dict[condition] = jittered_x
            box_ax.scatter(jittered_x, data_conditions[condition], marker = 'o', facecolor = condition_colors[condition],
                           s = markersize**2, linewidth = 0, zorder = 7 )

    if logscale:
        box_ax.set_yscale('log')
        box_ax.set_ylim([1,None])
        box_ax.set_ylim(ylim)
        box_ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

    else:
        box_ax.set_ylim(ylim)
        if linearLocator:
            box_ax.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(3))
        elif maxNLocator:
            box_ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        box_ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
    box_ax.set_ylabel(ylabel)
    box_ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.arange(len(conditions))))
    box_ax.set_xticklabels(xtick_labels, rotation = 35, ha ='center')
    new_title = lpf.cleanStringForFilename(ylabel)
    data_list = list(data_conditions.values())

    if plot_stats or save_stats:
        if num_box == 2:
            f_test = sf.FTest(x = data_list[0], y = data_list[1])
            t_test = stats.ttest_ind(a = data_list[0], b =data_list[1], equal_var = stats_assume_equal_var)
            t_test_pingouin = pingouin.ttest(x = data_list[0], y = data_list[1], correction = True)
            t_test_pingouin['scipy_stats_T'] = t_test[0]
            t_test_pingouin['scipy_stats_pval'] = t_test[1]
            t_test_pingouin['FTest_F'] = f_test['f_stat']
            t_test_pingouin['FTest_pval'] = f_test['p_val']
            t_test_pingouin['FTest_ddof1'] = f_test['ddof1']
            t_test_pingouin['FTest_ddof2'] = f_test['ddof2']
            t_test_pingouin[f'mean_{conditions[0]}'] = np.nanmean(data_conditions[conditions[0]])
            t_test_pingouin[f'sem_{conditions[0]}'] =stats.sem(data_conditions[conditions[0]], ddof = 1, nan_policy = 'omit')
            t_test_pingouin[f'var_{conditions[0]}'] = np.nanvar(data_conditions[conditions[0]], ddof = 1)
            t_test_pingouin[f'mean_{conditions[1]}'] = np.nanmean(data_conditions[conditions[1]])
            t_test_pingouin[f'sem_{conditions[1]}'] =stats.sem(data_conditions[conditions[1]], ddof = 1, nan_policy = 'omit')
            t_test_pingouin[f'var_{conditions[1]}'] = np.nanvar(data_conditions[conditions[1]], ddof = 1)
            if plot_stats:
                title = f'mean: p = {np.round(t_test[1], 4)} (t= {round(t_test[0], 4)})\nvariance: p = {round(f_test["p_val"], 4)} (F= {round(f_test["f_stat"], 4)})'
                box_ax.set_title(title)
                ylim_upper = box_ax.get_ylim()[1]
                box_ax.plot([0,1], [ylim_upper,ylim_upper], color = 'k', linewidth = linewidth_stats_lines, markersize = 0)
            if save_stats:
                stats_fig_path = os.path.join(fig_path, 'stats')
                Path(stats_fig_path).mkdir(parents=False, exist_ok=True)
                t_test_pingouin.to_csv(os.path.join(stats_fig_path, f'Stats- {new_title}_'+ '_'.join(conditions)+ f'_{plot_title}' +'.csv'))

        # elif num_box > 2:
        #     pass
        #     # if stats_assume_equal_var:
        #     #     anova_pingouin = pingouin.anova(dv = 'data', between = 'condition', data = anova_df)
        #     #     anova_scipy = stats.f_oneway(*data_list)
        #     #     anova_pingouin['scipy_stats_T'] = anova_scipy[0]
        #     #     anova_pingouin['scipy_stats_pval'] = anova_scipy[1]
        #     # else:
        #     #     anova_pingouin = pingouin.welch_anova(dv = 'data', between = 'condition', data = anova_df)

        #     anova_pingouin = pd.DataFrame()

        #     for condition in conditions:
        #         anova_pingouin[f'mean_{condition}'] = np.nanmean(data_conditions[condition])
        #         anova_pingouin[f'sem_{condition}'] =stats.sem(data_conditions[condition], ddof = 1, nan_policy = 'omit')
        #         anova_pingouin[f'var_{condition}'] = np.nanvar(data_conditions[condition], ddof = 1)

        #     if plot_stats:
        #         title = f"mean: p = {np.round(anova_pingouin['p-unc'].iloc[0], 4)} (F= {round(anova_pingouin.F.iloc[0], 4)})"
        #         box_ax.set_title(title)
        #         ylim_upper = box_ax.get_ylim()[1]
        #         box_ax.plot([0, len(conditions)-1], [ylim_upper,ylim_upper], color = 'k', linewidth = linewidth_stats_lines, markersize = 0)
        #     if save_stats:
        #         stats_fig_path =os.path.join(fig_path, 'stats')
        #         Path(stats_fig_path).mkdir(parents=False, exist_ok=True)
        #         anova_pingouin.to_csv(os.path.join(stats_fig_path, f'Stats- {new_title}_'+ '_'.join(conditions)+ f'_{plot_title}' +'.csv'))

    # print(box_ax.figure.subplotpars.right)
    # print(box_ax.figure.subplotpars.left)
    if axsize is not None:
        set_ax_size_inches(axsize[0], axsize[1], box_ax, print_new_sizes=False)
        # print(box_ax.figure.subplotpars.right)
        # print(box_ax.figure.subplotpars.left)
    standardize_plot_graphics(box_ax)

    if save_fig:
        if save_png:
            box_fig.savefig(os.path.join(fig_path, f'{new_title}_'+ '_'.join(conditions) + f'_{plot_title}' +'.png'),
                            dpi = 600,
                            transparent = True,
                            bbox_inches = 'tight')
        box_fig.savefig(os.path.join(fig_path, f'{new_title}_'+ '_'.join(conditions)+ f'_{plot_title}'+'.pdf'),
                        transparent = True,
                        bbox_inches = 'tight')

    if (plot_stats or save_stats) and num_box == 2:
        return {'fig':box_fig, 'ax':box_ax, 'jitter_x_dict': jitter_x_dict, 'data_conditions':data_conditions, 'p-val': t_test[1] } # 't_test': t_test, 'f_test': f_test }
    else:
        return {'fig':box_fig, 'ax':box_ax, }

def plot_paired_bars_from_dicts_or_list(data_paired,
                                condition_colors,
                                labels = [],
                                plot_individuals = True,
                                plot_connecting_lines = True,
                                linewidth_connection = 0.5,
                                alpha_connecting_lines = 0.5,
                                plot_error = True,
                                ylim =[None,None],
                                bar_alpha = 0.3,
                                markersize = 2.1,
                                plot_sem = True,
                                bar_width = 0.8,
                                ylabel ='',
                                jitter_individuals = False,
                                jitter_width = 20,
                                logscale = False,
                                save_fig = False,
                                save_png = False,
                                fig_path ='',
                                title='',
                                data_is_nested_dict = True,
                                data_is_regular_dict = False,
                                data_is_df = False,
                                data_is_nested_list = False,
                                axsize = (0.64, 1),
                                plot_stats = True,
                                linewidth_error = 1,
                                color_error = 'k',
                                linewidth_stats_lines = 0.5,
                                linearLocator = False,
                                maxNLocator = False,
                                save_stats = False
                                ):

    if (not (data_is_df or data_is_regular_dict or data_is_nested_dict)):
        raise Exception('Unclear what format data is in, check data_is_df, data_is_regular_dict, or data_is_nested_dict flags')
    elif data_is_nested_dict:
        data_conditions = {x: list(data_paired[x].values()) for x in data_paired}

    elif data_is_regular_dict:
        data_conditions = data_paired
    # elif data_is_df:
    #     data_conditions = {x: list(data_paired.loc[x,:]) for x in data_paired.index.get_level_values('condition')}
    num_bars = 2
    # if plot_stats or save_stats:

    #     anova_df = pd.DataFrame({'condition' : np.repeat(list(data_conditions.keys()), [len(v) for k, v in data_conditions.items()]),
    #                              'data' : list(itertools.chain.from_iterable(data_conditions.values())), })
    #     anova_df = anova_df[anova_df['data'].astype(bool)]

    #     anova_df["data"] = pd.to_numeric(anova_df["data"])

    # conditions = list(data_conditions.keys())
    # conditions =  sort_list_by_key(conditions, order_to_plot)
    if logscale:
        bottom = 1
    else:
        bottom = None
    bar_fig, bar_ax = plt.subplots(figsize =(2,3))
    xtick_labels =[]
    jitter_x_dict ={}
    for idx, label in enumerate(labels):
        data_conditions[label] = [x for x in data_conditions[label] if type(x) is not list]
        condition_mean = np.nanmean(data_conditions[label])
        if plot_sem:
            condition_error = stats.sem(data_conditions[label], ddof =1, nan_policy = 'omit')
        else:
            condition_error = np.nanstd(data_conditions[label], ddof =1)
        mean_error_labels = f'{label}\n{round(condition_mean, 2)} ± {round(condition_error, 2)}'
        xtick_labels.append(mean_error_labels)
        if plot_error:
            error = condition_error
        #bar plots height from bottom so if bottom set to 1 for log scale need to subtract to plot correctly
        if bottom is not None:
            bar_height_mean = condition_mean - bottom
        else:
            bar_height_mean = condition_mean
        bar_ax.bar(idx, bar_height_mean, yerr = error, width = bar_width,
                   color = condition_colors[label], linewidth = 0, alpha = bar_alpha, ecolor = color_error,
                   error_kw = {'elinewidth': linewidth_error}, bottom = bottom)


        if plot_individuals:
            #if jitter_individuals:
            x_scatter = [idx] * len(data_conditions[label])
                # jittered_x = [x + random.randrange(-jitter_width, jitter_width, 1)/100
                #               for x
                #               in x_scatter
                #               ]
                # jitter_x_dict[label] = jittered_x
            bar_ax.scatter(x_scatter, data_conditions[label], marker = 'o', facecolor = condition_colors[label],
                           s = markersize**2, linewidth = 0, zorder = 7 )
    if plot_individuals:
        if plot_connecting_lines:
            bar_ax.plot([0,1],
                        [data_conditions[labels[0]],
                         data_conditions[labels[1]]],
                        color = 'k',
                        linewidth = linewidth_connection,
                        markersize = 0,
                        alpha = alpha_connecting_lines,
                        )

    if logscale:
        bar_ax.set_yscale('log')
        bar_ax.set_ylim([1,None])
        bar_ax.set_ylim(ylim)
        bar_ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

    else:
        bar_ax.set_ylim(ylim)
        if linearLocator:
            bar_ax.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(3))
        elif maxNLocator:
            bar_ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        bar_ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
    bar_ax.set_ylabel(ylabel)
    bar_ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0, 1]))
    bar_ax.set_xticklabels(xtick_labels, rotation = 35, ha ='center')
    new_title = lpf.cleanStringForFilename(ylabel)

    if plot_stats or save_stats:
        f_test = sf.FTest(x = data_conditions[labels[0]], y = data_conditions[labels[1]])
        t_test = stats.ttest_rel(a = data_conditions[labels[0]], b =data_conditions[labels[1]])
        t_test_pingouin = pingouin.ttest(x =data_conditions[labels[0]], y = data_conditions[labels[1]], paired = True)
        t_test_pingouin['scipy_stats_T'] = t_test[0]
        t_test_pingouin['scipy_stats_pval'] = t_test[1]
        t_test_pingouin['FTest_F'] = f_test['f_stat']
        t_test_pingouin['FTest_pval'] = f_test['p_val']
        t_test_pingouin['FTest_ddof1'] = f_test['ddof1']
        t_test_pingouin['FTest_ddof2'] = f_test['ddof2']
        for label in labels:
            t_test_pingouin[f'mean_{label}'] = np.nanmean(data_conditions[label])
            t_test_pingouin[f'sem_{label}'] =stats.sem(data_conditions[label], ddof = 1, nan_policy = 'omit')
            t_test_pingouin[f'var_{label}'] = np.nanvar(data_conditions[label], ddof = 1)
        if plot_stats:
            title = f'mean: p = {np.round(t_test[1], 4)} (t= {round(t_test[0], 4)})\nvariance: p = {round(f_test["p_val"], 4)} (F= {round(f_test["f_stat"], 4)})'
            bar_ax.set_title(title)
            ylim_upper = bar_ax.get_ylim()[1]
            bar_ax.plot([0,1], [ylim_upper,ylim_upper], color = 'k', linewidth = linewidth_stats_lines, markersize = 0)
        if save_stats:
            stats_fig_path = os.path.join(fig_path, 'stats')
            Path(stats_fig_path).mkdir(parents=False, exist_ok=True)
            t_test_pingouin.to_csv(os.path.join(stats_fig_path, f'Stats- {new_title}_'+ '_'.join(labels) +'.csv'))

    if axsize is not None:
        set_ax_size_inches(axsize[0], axsize[1], bar_ax, print_new_sizes=False)
    standardize_plot_graphics(bar_ax)

    if save_fig:
        if save_png:
            bar_fig.savefig(os.path.join(fig_path, f'{new_title}_'+ '_'.join([str(x) if not isinstance(x, str) else x for x in labels]) +'.png'),
                            dpi = 600,
                            transparent = True,
                            bbox_inches = 'tight')
        bar_fig.savefig(os.path.join(fig_path, f'{new_title}_'+ '_'.join([str(x) if not isinstance(x, str) else x for x in labels]) +'.pdf'),
                        transparent = True,
                        bbox_inches = 'tight')

    if plot_stats or save_stats:
        return {'fig':bar_fig, 'ax':bar_ax, }#'jitter_x_dict': jitter_x_dict, 'data_conditions':data_conditions} # 't_test': t_test, 'f_test': f_test }
    else:
        return {'fig':bar_fig, 'ax':bar_ax, }

def plotBarsFromDict_brokenAxis(data_by_condition,
                                condition_colors,
                                order_to_plot = [],
                                plot_individuals = True,
                                plot_error = True,
                                ylim1 =[None, None],
                                ylim2 = [None, None],
                                hspace=0.02,
                                height_ratios=[1, 6],
                                d = 0.5,
                                MultipleLocator = None,
                                bar_alpha = 0.3,
                                markersize = 2.1,
                                plot_sem = True,
                                plot_median_and_IQR= False,
                                bar_width = 0.8,
                                ylabel ='',
                                jitter_individuals = True,
                                jitter_width = 20,
                                logscale = False,
                                save_fig = False,
                                save_png = False,
                                fig_path ='',
                                title='',
                                data_is_nested_dict = True,
                                data_is_regular_dict = False,
                                data_is_df = False,
                                axsize = (0.64, 1),
                                plot_stats = True,
                                linewidth_error = 0.5,
                                color_error = 'k',
                                linewidth_stats_lines = 0.5,
                                stats_assume_equal_var = False,
                                ax_to_plot = None,
                                linearLocator = False,
                                maxNLocator = False,
                                save_stats = False
                                ):

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, height_ratios = height_ratios)
    fig.subplots_adjust(hspace = hspace)  # adjust space between Axes

    # plot the same data on both Axes
    figax_learned_trial = plotBarsFromDict(data_by_condition,
                                               condition_colors = condition_colors,
                                               ylabel = '',
                                               axsize = axsize,
                                               data_is_nested_dict= data_is_nested_dict,
                                               data_is_regular_dict = data_is_regular_dict,
                                               data_is_df = data_is_df,
                                               ylim = [None, None],
                                               ax_to_plot = ax1,
                                               save_fig = False,
                                               fig_path = '',
                                               )
    figax_learned_trial =  plotBarsFromDict(data_by_condition,
                                               condition_colors = condition_colors,
                                               ylabel = ylabel,
                                               plot_stats = True,
                                               axsize = axsize,
                                               data_is_nested_dict= data_is_nested_dict,
                                               data_is_regular_dict = data_is_regular_dict,
                                               data_is_df = data_is_df,
                                               ylim = [None, None],
                                               ax_to_plot = ax2,
                                               save_fig = False,
                                               fig_path = '',
                                               )


    ax1.set_ylim(ylim1)
    ax2.set_ylim(ylim2)
    if MultipleLocator is not None:
        ax1.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(MultipleLocator))
    ax1.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
    # hide the spines between ax and ax2
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.tick_params(axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False)
    ax1.tick_params(labeltop=False)
    ax2.xaxis.tick_bottom()


    d = .5  # vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=4,
                  linestyle="none", linewidth = 0.35, color='k', mec='k', mew=0.35, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

    if save_fig:
        fig.savefig(os.path.join(fig_path, f'{ylabel} for {data_by_condition.keys()}.pdf'),
                        transparent = True,
                        bbox_inches = 'tight')
    return {'fig': fig, 'ax1':ax1, 'ax2': ax2}