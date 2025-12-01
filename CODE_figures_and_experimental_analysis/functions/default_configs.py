# -*- coding: utf-8 -*-
"""
@author: Dennis A. Burke (dennis.burke@ucsf.edu ; permanent address: dennis.a.burke AT gmail)

from Burke et al. - Duration between rewards controls the rate of behavioral and dopaminergic learning

HELPER FUNCTIONS
"""

import matplotlib.pyplot as plt
import matplotlib


plt.rcParams['axes.titlesize'] = 8
plt.rcParams['axes.labelsize'] = 7
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6
plt.rcParams['legend.fontsize'] = 6
plt.rcParams['legend.labelspacing'] = 0.2
plt.rcParams['axes.labelpad'] = 2
plt.rcParams['axes.linewidth'] = 0.35
plt.rcParams['xtick.major.size'] = 1.5
plt.rcParams['xtick.minor.size'] = 0.75
plt.rcParams['xtick.major.width'] = 0.35
plt.rcParams['xtick.minor.width'] = 0.35
plt.rcParams['xtick.major.pad'] = 2
plt.rcParams['ytick.major.size'] = 1.5
plt.rcParams['ytick.minor.size'] = 0.75
plt.rcParams['ytick.major.width'] = 0.35
plt.rcParams['ytick.minor.width'] = 0.35
plt.rcParams['ytick.major.pad'] = 2
plt.rcParams['lines.scale_dashes'] = False
plt.rcParams['lines.dashed_pattern'] = (2, 1)
plt.rcParams['font.sans-serif'] = ['HelveticaLTStd-Light']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['text.color'] = 'k'
plt.rcParams['figure.max_open_warning']= 50
matplotlib.rcParams['figure.dpi'] = 144


colors_for_conditions = {'600s': '#7A4A9D',
                        '60s': '#CDAE2C',
                        '60s-few': 'red',
                        '60s-CSminus': '#45b97c',
                        '60s-few-ctxt': '#CE768F',
                        '30s': '#f89a1c',
                        '300s': '#be5aa2',
                        '3600s': '#4362AD',
                        '600s-bgdmilk': '#5F5555',
                        '60s-50%': '#ad2372',
                        '60s-10%': '#906CAF',
                        '30 s ITI': '#f89a1c',
                        '60 s ITI': '#CDAE2C',
                        '300 s ITI': '#be5aa2',
                        '600 s ITI': '#7A4A9D',
                        '3600 s ITI': '#4362AD',
                        }



colors_for_conditions_DA = colors_for_conditions
# colors_for_conditions_DA = {'600s': '#503B72' ,
#                             '60s':  '#907C2F',
#                             '3600s': 'darkblue',
#                             '60s-few': 'darkred',
#                             '60s-50%': 'AD2372',
#                             }
colors_for_conditions_DA_reward = colors_for_conditions
# colors_for_conditions_DA_reward = {'600s':'#2B213F',
#                                    '60s': '#605216',
#                                    '3600s': 'midnightblue',
#                                    '60s-few': 'maroon',
#                                    '60s-50%': '#AD2372',
#                                    }

"""
SET:
Example animals
"""

cumsum_examples_behavior = {'600s':'600s_F1',
                            '60s':'60sD_M8',
                            '30s': '30s_M3',
                            '300s': '300s_M2',
                            '3600s': '3600s_F2',
                            }
cumsum_examples_behavior_supp = {'600s':'600s_M5',
                                 '60s':'60s_F3',
                                 }
cumsum_examples_DA = {'600s':'600sD_F8',
                      '60s':'60sD_F7',
                      '3600s': '3600sD_F2',
                      '60s-50%':'60s-50%D_M3',
                      '60s-10%': '60s-10%D_F5'}
cumsum_examples_DA_supp = {'600s':'600sD_M8',
                           '60s':'60sD_M8',
                           }



"""
SET:
Different axsizes for different types of figures

"""

axsize_timecourse = (1, 1)
axsize_timecourse_inset = (0.7, 0.7)
axsize_cumsum_examples = (1,1)
axsize_cumsum_all_individuals = (0.65, 0.65, 0.5, 0.1)
axsize_cumsum_all_individuals_DA = (0.65, 0.65, 0.2, 0.2)
axsize_cumsum_all_individuals_DA_nosharey = (0.65, 0.65, 0.5, 0.5)
axsize_raster_PSTH_panel = (0.5, 0.35)
axsize_bars_1 = (0.28, 1)
axsize_bars_2 = (0.64, 1)
axsize_bars_3 = (0.96, 1)
axsize_bars_4 = (1.28, 1)
axsize_bars_5 = (1.60, 1)

fontsize_title = 7
fontsize_label = 7
fontsize_ticks = 6

linestyle_lick = 'solid'
linestyle_DA = 'dashed'

alpha_timecourse_error = 0.3

markersize_bargraphs = 2.1

#example PSTH parameters
color_cue_shade = '#939598'
alpha_cue_shade = 0.5
stroke_raster = 0.35
stroke_PSTH = 0.5
alpha_PSTH_error = 0.3
linewidth_0_lick = 0
alpha0line = 0.5
linewidth_0_DA = 0
stroke_cue_reward_vertical = 0.25
xlim_sec_examplePSTH = [-2.5, 7.5]
smooth_lick = 0.75
plot_in_sec = True
align_to_cue = True
use_DA_cue_full = False
plot_symbols = False

bout_threshold = 0.5
#DA normalization
norm_to_max_individual_rewards = 3
norm_to_max_individual_antic = 0

#cumsum
linestyle_learned_trial = 'solid'
color_DA_trial = 'k'