# -*- coding: utf-8 -*-
"""
@author: Dennis A. Burke (dennis.burke@ucsf.edu ; permanent address: dennis.a.burke AT gmail)

from Burke et al. - Duration between rewards controls the rate of behavioral and dopaminergic learning

HELPER FUNCTIONS
"""
import scipy.stats as stats
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics import roc_auc_score as auROC
from scipy.optimize import curve_fit



def extractLickTimesAroundEvent(lick_times, event_times, total_time_window: int = 14000,
                           baseline_period: int = 7000):
    '''
    given a list or array of lick time stamps, extract all lick times around
    an event time (event could be a cue or reward or other event with timestamps)

    returns a list of arrays for each "trial" extracted around lick event

    Parameters
    ----------
    lick_times : numpy array of lick timestamps
        DESCRIPTION.
    event_times : TYPE
        DESCRIPTION.
    total_time_window : int, optional
        DESCRIPTION. The default is 13000.
    baseline_period : int, optional
        DESCRIPTION. The default is 7000.

    Returns
    -------
    trials_all : TYPE
        DESCRIPTION.

    '''
    after_stimulus= total_time_window - baseline_period

    if isinstance(event_times, (list, tuple, set, np.ndarray)): #otherwise 'len' will throw error if array input is a single scalar int
        trials_all = np.zeros(shape=(np.size(event_times, 0),1)).tolist()
    else:
        trials_all = np.zeros(shape=(1,1)).tolist()



    for index, event in enumerate(event_times):
        current_trial = lick_times[((lick_times > (event-baseline_period))
                                    & (lick_times<(event + after_stimulus)))]

        #current_trial= (current_trial - stim)/1000
        trials_all[index]=current_trial

    return trials_all



def extractLickOffTimesAroundEvent(lick_on, lick_off, event_times, total_time_window: int = 14000,
                           baseline_period: int = 7000):
    '''
    given a list or array of lick time stamps, find all lick ON times around and event and then return the corresponding
    lick OFF times (event could be a cue or reward or other event with timestamps)

    returns a list of arrays for each "trial" extracted around lick event

    Parameters
    ----------
    lick_times : numpy array of lick timestamps
        DESCRIPTION.
    event_times : TYPE
        DESCRIPTION.
    total_time_window : int, optional
        DESCRIPTION. The default is 13000.
    baseline_period : int, optional
        DESCRIPTION. The default is 7000.

    Returns
    -------
    trials_all : TYPE
        DESCRIPTION.

    '''

    if len(lick_on) > len(lick_off):
        lick_on = lick_on[:-1]
    after_stimulus= total_time_window - baseline_period

    if isinstance(event_times, (list, tuple, set, np.ndarray)): #otherwise 'len' will throw error if array input is a single scalar int
        trials_all = np.zeros(shape=(np.size(event_times, 0),1)).tolist()
    else:
        trials_all = np.zeros(shape=(1,1)).tolist()



    for index, event in enumerate(event_times):
        current_trial = lick_off[((lick_on > (event-baseline_period))
                                    & (lick_on<(event + after_stimulus)))]

        #current_trial= (current_trial - stim)/1000
        trials_all[index]=current_trial

    return trials_all
def alignLickTimesToTrialEvents(lick_trials, event_times):

    if len(lick_trials) != len(event_times):
        raise Exception("lick trials and events to align to have dif lengths")

    event_aligned_licks = [lick - event for lick, event in zip(lick_trials, event_times)]

    return event_aligned_licks

def convertLickTimesToSeconds(lick_times, licks_are_trials = True):


    #if licks are already extracted in trials, ie output of "extractLickAroundEvent"
    if licks_are_trials:
        licks_in_sec = [((lick)/1000) for lick in lick_times]
    #else if licks are just in a single array
    else:
        licks_in_sec = lick_times/1000

    return licks_in_sec



def extractLickRasterAroundEvent(lick_times, event_times, total_time_window: int = 14000,
                           baseline_period: int = 7000, in_seconds = False):


    time = np.arange(total_time_window)

    lick_trials_raw = extractLickTimesAroundEvent(lick_times, event_times, total_time_window,
                                              baseline_period)
    lick_trials_aligned = alignLickTimesToTrialEvents(lick_trials_raw, event_times)
    x_time = time - baseline_period

    if in_seconds:
        lick_trials = convertLickTimesToSeconds(lick_trials_aligned, licks_are_trials =True)
        x_time = x_time/1000
    else:
        lick_trials = lick_trials_aligned



    trials_all = np.zeros(shape=(np.size(lick_trials, 0),total_time_window))




    for index, trial in enumerate(lick_trials):
        #current_trial = trial[

        #current_trial= (current_trial - stim)/1000
        trials_all[index,:][trial]=1

    return x_time, trials_all

def getLickPSTH(lick_trials_aligned, binsize= 100, total_time_window: int = 14000,
                           baseline_period: int = 7000, in_seconds = False, time_window_in_seconds = False):
    if ((in_seconds) and (not time_window_in_seconds)):
        total_time_window = total_time_window/1000
        baseline_period =baseline_period/1000
        binsize = binsize/1000
    time = np.arange(total_time_window)

    aligned_time = time - baseline_period


    bins = np.arange(aligned_time[0],aligned_time[-1]+binsize, binsize)

    hist_by_trial = np.zeros(shape=(len(lick_trials_aligned),len(bins)-1))

    for index, trial in enumerate(lick_trials_aligned):
        hist, edges = np.histogram(trial, bins)

        #hist_rate = hist/binsize


        if in_seconds:
            hist_rate = hist/binsize
        else:
            hist_rate = hist/(binsize/1000)
        hist_by_trial[index,:] = hist_rate


    # raw_hist, bin_edges = np.histogram(np.concatenate(lick_trials_aligned, axis = 0), bins)
    # raw_hist = raw_hist/(len(lick_trials_aligned))


    hist_dict = {'hist_by_trials': hist_by_trial,
                 'bins': bins,
                 'mean_hist': np.mean(hist_by_trial, axis = 0),
                 'sem_hist': stats.sem(hist_by_trial, ddof =1),
                 }
    return hist_dict


#photo functions

def renameDoricCSVColumns(df):
    descriptive_name_dict = {'Time(s)': 'photo_time_s', 'doric_time': 'photo_time_s', 'AIn-1 - Dem (AOut-1)': '405',
                             'AIn-1 - Dem (AOut-2)': '470', 'DI/O-1': 'matlab_on_TTL',
                             'DI/O-2': 'event_TTL'}

    descriptive_name_dict_decimated = {'Time(s)': 'photo_time_s', 'doric_time': 'photo_time_s','AIn-1 - Dem (AOut-1) Decimation': '405',
                             'AIn-1 - Dem (AOut-2) Decimation': '470', 'DI/O-1 Decimation': 'matlab_on_TTL',
                             'DI/O-2 Decimation': 'event_TTL'}
    renamed_df = df.rename(columns = descriptive_name_dict)

    decimated_renamed_df = renamed_df.rename(columns = descriptive_name_dict_decimated)
    return decimated_renamed_df



def detrend(iso, dlight, time = None, start = None, end = None):
    '''
    use the isosbestic signal to detrend (correct for photobleaching downward
    slope) the signal of interest and return a flattened sigal

    time, start, end are optional parameters to detrend based on a segment of the
    overall signal (i.e. to ignore artifact area at beginning or end
    or baseline before a drug injection)

    Parameters
    ----------
    iso : array or series
        isobestic/405nm channel signal.
    dlight : array or series
        signal/470nm channel signal.
    start : int, optional
        The default is None.
    end : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    detrendedSignal : TYPE
        DESCRIPTION.

    '''
    if isinstance(iso, np.ndarray):
        x = iso
    elif isinstance(iso, pd.Series):
        x = iso.to_numpy()


    if isinstance(dlight, np.ndarray):
        y = dlight
    elif isinstance(dlight, pd.Series):
        y = dlight.to_numpy()

    #if do not want to linear fit based on whole curves and time and (start  or stop) given
    if time is not None:
        if isinstance(iso, np.ndarray):
            time = time
        elif isinstance(iso, pd.Series):
            time = time.to_numpy()

        time_subset = time
        if start is not None:

            time_subset = time_subset[time_subset>start]
        if end is not None:
            time_subset = time_subset[time_subset< end]
        indices_to_fit = np.flatnonzero(np.isin(time, time_subset))

        x = x[indices_to_fit]
        y = y[indices_to_fit]


    # NOTE**** Polyfit throws error for line fitting and does a bad job,
    # linregress is much better

    #remove NaNs
    mask = ~np.isnan(x) & ~np.isnan(y)
    bls = stats.linregress(x[mask], y[mask])
    Y_fit_all = np.multiply(bls[0], iso) + bls[1]
    Y_dF_all = dlight - Y_fit_all

    detrendedSignal = np.multiply(100, np.divide(Y_dF_all, Y_fit_all))

    return detrendedSignal


def getDoricTTLStartStopOnTimes(photo_df, event_TTL= True, use_matlab_time = False, time_in_ms = True):

    '''
    given an arbitrary array that represents TTL signals, get the index of when
    they first go on, they first go low, and every index that is high for the
    signal, use that to index time array to get times
    '''
    if event_TTL:
        ttl = 'event_TTL'
    else:
        ttl = 'matlab_on_TTL'

    if use_matlab_time:
        time = 'matlab_time_ms'
    else:
        if time_in_ms:
            time = 'photo_time_ms'
        else:
            time = 'photo_time_s'

    onTimes = photo_df[time][np.flatnonzero(photo_df[ttl]==1)].to_numpy()

    difs = np.diff(photo_df[ttl])
    difs = np.insert(difs, 0, 0)

    startTimes = photo_df[time][np.flatnonzero(difs>0)].to_numpy()
    stopTimes = photo_df[time][np.flatnonzero(difs<0)].to_numpy()

    return startTimes, stopTimes, onTimes

def getTTLStartStopOnTimes(TTL_data, TTL_time):

    '''
    given an arbitrary array that represents TTL signals, get the index of when
    they first go on, they first go low, and every index that is high for the
    signal
    '''

    onTimes = TTL_time[TTL_data]

    difs = np.diff(TTL_data)
    difs = np.insert(difs, 0, 0)

    startTimes = TTL_time[np.flatnonzero(difs>0)]
    stopTimes = TTL_time[np.flatnonzero(difs<0)]

    return startTimes, stopTimes, onTimes



def extractEpoch(time, data, timestamps_array, total_time_window: int = 15000, baseline_period: int = 7000):
    '''


    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    timestamps_array : TYPE
        DESCRIPTION.
    total_time_window : int, optional
        DESCRIPTION. The default is 4000.
    baseline_period : int, optional
        DESCRIPTION. The default is 1000.
    smplFreq : TYPE, optional
        DESCRIPTION. The default is 1017.2526245117188.
    use405CorrectedSignal : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    x_time : TYPE
        DESCRIPTION.
    epochAll : TYPE
        DESCRIPTION.
    epochMean : TYPE
        DESCRIPTION.
    epochSEM : TYPE
        DESCRIPTION.

    '''

    #turn pandas series into numpy arrays for consistent datatypes on return values
    if isinstance(time, pd.Series):
        time = time.to_numpy()
    if isinstance(data, pd.Series):
        data= data.to_numpy()

    timesteps = np.diff(time)
    timestep_mean = np.mean(timesteps)
    smplFreq = 1/timestep_mean


    # take an array of indices (i.e water delivery indices) and extract dFF signal
    #from around that time point normalized to period before TTL index
    # if "use405CorrectedSignal is false there is no normalization done to the raw signal before baselining in this function
    #if true, we use the corrected/detrended signal to generate the arrays and plots and dFF refers to change over 405 estimated baseline



    after_stimulus= total_time_window - baseline_period

    interp = False
    if isinstance(timestamps_array, (list, tuple, set, np.ndarray)): #otherwise 'len' will throw error if array input is a single scalar int/not interable

        first_doric_after_stim = [time[time>x][0] for x in timestamps_array]
        list_of_xtimes = [time[np.flatnonzero((time>=x-baseline_period) & (time<= x + after_stimulus))]
                      for x in first_doric_after_stim]


        len_xtimes_max = max(map(len, list_of_xtimes))
        len_xtimes_min = min(map(len, list_of_xtimes))

        if len_xtimes_max != len_xtimes_min:

            interpolated_xtimes = np.linspace(-baseline_period, after_stimulus, len_xtimes_max )
            interp = True #flag that values will need to be interpolated later for consistent array sizes
            x_time = interpolated_xtimes

        else:
            #if no need to interpolate, just zero stimulus time in first 'x_time' calculated above to get general x_time
            x_time = list_of_xtimes[0] - first_doric_after_stim[0]

        epoch_len =  len_xtimes_max
        epochAll = np.zeros(shape=(np.size(timestamps_array, 0), epoch_len))

    else:
        #if only single scalar timestamp to extract, no need to worry about interpolation
        first_doric_after_stim = time[time > timestamps_array][0]


        epoch_time = time[(time>(first_doric_after_stim- baseline_period)) & (time < (first_doric_after_stim + after_stimulus))]
        x_time = epoch_time-first_doric_after_stim
        epoch_len = len(x_time)
        first_doric_after_stim=[first_doric_after_stim] #turn scalar into list so no error is thrown in for loop below
        epochAll = np.zeros(shape=(1, epoch_len))



    for index, value in enumerate(first_doric_after_stim):

        #check for edge case for if stimulus at the end of a session and extracted epoch will overrun total time
        if (value + after_stimulus) <= np.max(time):
            epochIndicies =  np.flatnonzero((time<= (value + after_stimulus)) & (time >= (value - baseline_period)) )

            np.arange((value-baseline_period),(value + after_stimulus))
            currentEpoch=data[epochIndicies]

            if interp:
                currentEpoch_time = time[epochIndicies] - value
                currentEpoch_interp = np.interp(interpolated_xtimes, currentEpoch_time, currentEpoch)
                currentEpoch = currentEpoch_interp

        else:
        #if session ends before last values to extract, pad the array with the last value
        #maybe better to do NaNs, but this will likely not come up often and can change code later if needed
            epochIndicies = np.flatnonzero((time >= (value - baseline_period)))
            sizeOfShortArray=len(epochIndicies)
            currentEpoch=np.pad(data[epochIndicies], [(0,(epoch_len - sizeOfShortArray))], 'edge')


        #have to pad array to fit into big array
        epochAll[index,:]=currentEpoch



    epochMean = np.nanmean(epochAll,axis=0)
    epochSEM = stats.sem(epochAll,axis=0, nan_policy ='omit')

    #return epoch x axis, individual dff in 2d array, epoch dff mean, and epoch dff sem
    return x_time, epochAll, epochMean, epochSEM




def zScoreEpoch(epoch, center, spread, robust=True):

    if robust:
        epochZscore = ((epoch-center)/(np.multiply(spread, 1.4826)))
    else:
        epochZscore = ((epoch-center)/(spread))

    return epochZscore


def calcAUROCscore(test, bsln):
    test = test[np.isfinite(test)]
    bsln = bsln[np.isfinite(bsln)]
    data = np.concatenate((test, bsln))
    labels = np.concatenate((np.ones(test.size,), np.zeros(bsln.size,)))
    return 2*auROC(labels, data)-1





def boutStartStop(events_on, events_off = (), maxIBI = 1000, use_events_off = True):
    '''


    Parameters
    ----------
    events_on : list or array of timestamps
        DESCRIPTION.
    maxIBI : TYPE, optional
        DESCRIPTION. The default is 1000.

    Returns
    -------
    bout_start : TYPE
        DESCRIPTION.
    bout_stop : TYPE
        DESCRIPTION.

    '''
    if use_events_off:
        inter_event_intervals =np.array([evon - evoff
                                         for (evon,
                                              evoff)
                                         in zip(events_on[1:],
                                                events_off[:-1])
                                         ]
                                        )


        bout_threshold = 1000

        bout_stop = events_off[np.flatnonzero(inter_event_intervals > maxIBI)]

        #the last timestamp in events list by definition is the last end of the last bout
        bout_stop = np.append(bout_stop, events_off[-1])

        inter_event_intervals = np.insert(inter_event_intervals, 0, maxIBI+10)
        bout_start = events_on[np.flatnonzero(inter_event_intervals > maxIBI)]

    else:
        inter_event_intervals = np.diff(events_on) #get time interval between each event

        # in array of index of each event with an
        bout_stop = events_on[np.flatnonzero(inter_event_intervals > maxIBI)]

        #the last timestamp in events_on list by definition is the last end of the last bout
        bout_stop = np.append(bout_stop, events_on[-1])


        #ensures that if first TTL is a single pulse it will be picked up
        inter_event_intervals = np.insert(inter_event_intervals, 0, maxIBI+10)
        bout_start = events_on[np.flatnonzero(inter_event_intervals > maxIBI)]



    return bout_start, bout_stop




def point_to_line(pt, p1, p2):
    d = np.empty([len(pt),1])
    d.fill(np.nan)

    line = p2 - p1;
    for i in range(len(pt)):
        b = p1 - pt[i,:]
        d[i] = np.linalg.norm(np.cross(line, b))/np.linalg.norm(line)
    return d


def getCumsumChangePoint(trials_list_x, cumsum_data_y, percent_max_dist = 0.75, data_direction = 'increase',  animal_name ='',):
    '''


    Parameters
    ----------
    trials_list_x : numpy array
        DESCRIPTION.
    cumsum_data_y : numpy array
        DESCRIPTION.
    percent_max_dist : float between 0 and 1, optional
        DESCRIPTION. The default is 0.75.
    data_direction : TYPE, optional
        DESCRIPTION. The default is 'increase'.

    Returns
    -------
    learned_trial_params : dict
        DESCRIPTION.

    '''


    full_trials_x = trials_list_x #np.insert(trials_list_x, 0, 0)
    full_trials_y = cumsum_data_y #np.insert(cumsum_data_y, 0, 0) #cumsum_data_y


    diagonal_y = np.linspace(0, cumsum_data_y[-1], len(full_trials_x)) #draw diagonal from 0 to end of y that's length of data/trials

    cumsum_data_all = np.stack((full_trials_x, full_trials_y), axis=1).reshape(-1, 2)
    dist_from_diag = np.ravel(point_to_line(cumsum_data_all, cumsum_data_all[0,:], cumsum_data_all[-1,:])) # calc distance between diagonal and datapoints

    max_dist = np.max(dist_from_diag)
    maxdist_cutoff = trials_list_x[np.flatnonzero(dist_from_diag>=(max_dist*percent_max_dist))] #find the first point that is within 75% of max distance from diagonal

    learned_trial = maxdist_cutoff[0]

    diagonal_x = trials_list_x #np.insert(trials_list_x, 0, 0)

    dist_at_learned_trial = dist_from_diag[np.flatnonzero(cumsum_data_all[:,0] == learned_trial)][0]

    try:
        if data_direction == 'increase':
            # if diagonal is under data at 'learned trial' (ie change trial captured is for decreasing data)
            # then iterate backwards over trials and data applying same algorithm until diagonal is above data at learned trial
            if (cumsum_data_y[np.flatnonzero(trials_list_x==learned_trial)][0]) > (diagonal_y[np.flatnonzero(trials_list_x == learned_trial)][0]):

                for last_x, last_y in zip (np.flip(trials_list_x[trials_list_x <= learned_trial]), np.flip(cumsum_data_y[trials_list_x <= learned_trial])):
                    diagonal_y = np.linspace(0, last_y, len(trials_list_x[trials_list_x <= last_x]))
                    diagonal_x = trials_list_x[trials_list_x <= last_x]
                    cumsum_data_all = np.stack((trials_list_x[trials_list_x <= last_x], cumsum_data_y[trials_list_x <= last_x]), axis=1).reshape(-1, 2)
                    dist_from_diag = np.ravel(point_to_line(cumsum_data_all, cumsum_data_all[0,:], cumsum_data_all[-1,:]))
                    max_dist = np.max(dist_from_diag)
                    maxdist_cutoff = trials_list_x[:last_x][np.flatnonzero(dist_from_diag >= (max_dist*percent_max_dist))]
                    learned_trial = maxdist_cutoff[0]

                    dist_at_learned_trial = dist_from_diag[np.flatnonzero(cumsum_data_all[:,0] == learned_trial)][0]
                    if (cumsum_data_y[np.flatnonzero(trials_list_x == learned_trial)][0]) < (diagonal_y[np.flatnonzero(trials_list_x == learned_trial)][0]):
                        break
                else: #in case loop never breaks out
                    raise ValueError(f'no inflection point found for {animal_name} on correct side of diagonal')

        elif data_direction == 'decrease':
            if (cumsum_data_y[np.flatnonzero(trials_list_x==learned_trial)][0]) < (diagonal_y[np.flatnonzero(trials_list_x == learned_trial)][0]):

                for last_x, last_y in zip (np.flip(trials_list_x[trials_list_x <= learned_trial]), np.flip(cumsum_data_y[trials_list_x <= learned_trial])):
                    diagonal_y = np.linspace(0, last_y, len(trials_list_x[trials_list_x <= last_x]))
                    diagonal_x = trials_list_x[trials_list_x <= last_x]
                    cumsum_data_all = np.stack((trials_list_x[trials_list_x <= last_x], cumsum_data_y[trials_list_x <= last_x]), axis=1).reshape(-1, 2)
                    dist_from_diag = np.ravel(point_to_line(cumsum_data_all, cumsum_data_all[0,:], cumsum_data_all[-1,:]))
                    max_dist = np.max(dist_from_diag)
                    maxdist_cutoff = trials_list_x[:last_x][np.flatnonzero(dist_from_diag >= (max_dist*percent_max_dist))]
                    learned_trial = maxdist_cutoff[0]

                    dist_at_learned_trial = dist_from_diag[np.flatnonzero(cumsum_data_all[:,0] == learned_trial)][0]
                    if (cumsum_data_y[np.flatnonzero(trials_list_x == learned_trial)][0]) > (diagonal_y[np.flatnonzero(trials_list_x == learned_trial)][0]):
                        break
                else: #in case loop never breaks out
                    raise ValueError(f'no inflection point found for {animal_name} on correct side of diagonal')
    except:
        print(f'Issue with change point detection for {animal_name}')
    #recalculate max distance in cordinates where y is normalized to 1
    cumsum_data_normed = cumsum_data_all
    cumsum_data_normed[:,1] = cumsum_data_normed[:,1]/ cumsum_data_normed[-1,1]
    dist_from_diag_norm = np.ravel(point_to_line(cumsum_data_normed, cumsum_data_normed[0,:], cumsum_data_normed[-1,:])) # calc distance between diagonal and datapoints
    max_dist_norm = np.max(dist_from_diag_norm)
    dist_at_learned_trial_norm = dist_from_diag_norm[np.flatnonzero(cumsum_data_normed[:,0] == learned_trial)][0]

    learned_trial_params = {'learned_trial': learned_trial, 'diag_y': diagonal_y, 'diag_x': diagonal_x, 'dist_from_diag': dist_from_diag, 'max_dist': max_dist,
                            'dist_at_learned_trial': dist_at_learned_trial, 'max_dist_norm':max_dist_norm, 'dist_at_learned_trial_norm': dist_at_learned_trial_norm }
    return learned_trial_params











# The double exponential curve we are going to fit.
def double_exponential(t, const, amp_fast, amp_slow, tau_slow, tau_multiplier):
    '''Compute a double exponential function with constant offset.
    Parameters:
    t       : Time vector in seconds.
    const   : Amplitude of the constant offset.
    amp_fast: Amplitude of the fast component.
    amp_slow: Amplitude of the slow component.
    tau_slow: Time constant of slow component in seconds.
    tau_multiplier: Time constant of fast component relative to slow.
    '''
    tau_fast = tau_slow*tau_multiplier
    return const+amp_slow*np.exp(-t/tau_slow)+amp_fast*np.exp(-t/tau_fast)


def curve_fit_double_exponential(signal_to_fit, time_signal, signal_in_ms = True):
    '''
    function from Thomas Akam
    Akam, Thomas, and Mark E. Walton. "pyPhotometry: Open source Python based
    hardware and software for fiber photometry data acquisition."
    Scientific reports 9.1 (2019): 3521.

    Parameters
    ----------
    signal_to_fit : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    if signal_in_ms:
        time_s = time_signal/1000
    else:
        time_s = time_signal


    max_sig = np.max(signal_to_fit)
    inital_params = [max_sig/2, max_sig/4, max_sig/4, 3600, 0.1]
    bounds = ([0      , 0      , 0      , 600  , 0],
              [max_sig, max_sig, max_sig, 36000, 1])
    signal_parms, parm_cov = curve_fit(double_exponential, time_s, signal_to_fit,
                                  p0=inital_params, bounds=bounds, maxfev=1000)
    signal_expfit = double_exponential(time_s, *signal_parms)

    return signal_expfit

def get_nonlearners (trial_df, learning_threshold = 0.5, day_above_threshold = 2, conditions_to_exclude =[] ):
    '''


    Parameters
    ----------
    trial_df : DataFrame
        DESCRIPTION. Dataframe where each row is data for a single trial for a single animal
    learning_threshold : int or float, optional
        DESCRIPTION. The average daily mean cue evoked lick rate in Hz over which
        learning is considered to have occured. The default is 0.5.
    day_above_threshold : TYPE, optional
        DESCRIPTION. Number of days average cue evoked lick rate needs to above
        threshold to count as learning. NOT CONSECUTIVE. The default is 2.

    Returns
    -------
    list of animal names that are classified as "nonlearners".

    '''
    excluded_df = trial_df[~trial_df['condition'].isin(conditions_to_exclude)].copy()
    lick_group = excluded_df.groupby(['condition', 'animal', 'day_num'], as_index = False)['antic_norm_rate_change'].mean()
    df_lick = lick_group.pivot(index = ['day_num'], columns = ['condition', 'animal'],
                               values = 'antic_norm_rate_change')
    df_lick_thresholded = df_lick[df_lick>=learning_threshold]

    learned_days_count = df_lick_thresholded[df_lick_thresholded.notnull()].count()

    learned_days_above_threshold = learned_days_count[learned_days_count < day_above_threshold]

    nonlearners_list = list(learned_days_above_threshold.index.get_level_values(1))
    nonlearners_conditions_list = list(learned_days_above_threshold.index.get_level_values(1))

    nonlearner_df = learned_days_above_threshold.reset_index().drop(0, axis = 1)
    return nonlearners_list



def cumLickTrialCount(trial_df, grouping_var = ['animal', 'cue_type']):

    trial_df['cue_trial_num'] = trial_df.groupby(grouping_var).cumcount()+1
    trial_df['rew_trial_num'] = trial_df.groupby(['animal', 'trial_type']).cumcount()+1
    trial_df['cumsum_antic_norm'] = trial_df.groupby(grouping_var)['nlicks_antic_norm'].cumsum()
    if 'epoch_dff_auc_antic_norm_500ms' in trial_df.columns:
        trial_df['cumsum_antic_dff_auc_norm_500ms'] = trial_df.groupby(grouping_var)['epoch_dff_auc_antic_norm_500ms'].cumsum()
        trial_df['cumsum_antic_dff_peak_norm_500ms'] = trial_df.groupby(grouping_var)['epoch_dff_peak_antic_norm_500ms'].cumsum()
        trial_df['cumsum_consume_dff_auc_norm_500ms'] = trial_df.groupby(grouping_var)['epoch_dff_auc_consume_norm_lickaligned_500ms'].cumsum()
        trial_df['cumsum_consume_dff_peak_norm_500ms'] = trial_df.groupby(grouping_var)['epoch_dff_peak_consume_norm_lickaligned_500ms'].cumsum()
        # trial_df['cumsum_antic_dff_auc_norm'] = trial_df.groupby(grouping_var)['epoch_dff_auc_antic_norm'].cumsum()
        # trial_df['cumsum_antic_dff_peak_norm'] = trial_df.groupby(grouping_var)['epoch_dff_peak_antic_norm'].cumsum()
        # trial_df['cumsum_consume_dff_auc_norm'] = trial_df.groupby(grouping_var)['epoch_dff_auc_consume_norm_lickaligned'].cumsum()
        # trial_df['cumsum_consume_dff_peak_norm'] = trial_df.groupby(grouping_var)['epoch_dff_peak_consume_norm_lickaligned'].cumsum()

        consume_measurement_wind_list = [1, 1.5, 2, 2.5, 3]
        consume_measurement_wind_list = [2]
        consume_measurement_wind_list_str = [str(x).replace('.', '') for x in consume_measurement_wind_list]
        for consume_meaure_wind in consume_measurement_wind_list_str:
            trial_df[f'cumsum_consume_dff_auc_norm_{consume_meaure_wind}s'] = trial_df.groupby(grouping_var)[f'epoch_dff_auc_consume_norm_{consume_meaure_wind}s'].cumsum()
            trial_df[f'cumsum_consume_dff_peak_norm_{consume_meaure_wind}s'] = trial_df.groupby(grouping_var)[f'epoch_dff_peak_consume_norm_{consume_meaure_wind}s'].cumsum()
            trial_df[f'cumsum_consume_dff_auc_norm_lickaligned_{consume_meaure_wind}s'] = trial_df.groupby(grouping_var)[f'epoch_dff_auc_consume_norm_lickaligned_{consume_meaure_wind}s'].cumsum()
            trial_df[f'cumsum_consume_dff_peak_norm_lickaligned_{consume_meaure_wind}s'] = trial_df.groupby(grouping_var)[f'epoch_dff_peak_consume_norm_lickaligned_{consume_meaure_wind}s'].cumsum()


    return trial_df



def search_string_list(string_list, search_terms):
    """
    Searches a list of strings for elements that match all specified conditions.

    Args:
        string_list: A list of strings to search.
        conditions: A list of conditions (strings or regular expressions).
                    Each condition must be present in a matching string.

    Returns:
        A list of strings that match all the given conditions.
    """
    matching_strings = []
    for string in string_list:
        if all(search_terms in string for search_terms in search_terms):
            matching_strings.append(string)
    return matching_strings

def get_recursively(search_dict, field):
    """Takes a dict with nested lists and dicts,
    and searches all dicts for a key of the field
    provided.
    """
    fields_found = []

    for key, value in search_dict.items():

        if key == field:
            fields_found.append(value)

        elif isinstance(value, dict):
            results = get_recursively(value, field)
            for result in results:
                fields_found.append(result)

        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    more_results = get_recursively(item, field)
                    for another_result in more_results:
                        fields_found.append(another_result)


def dropInitialNoLickTrials(trial_df, trials_to_drop = 'consume', conditions_to_exclude = None):

    if isinstance(conditions_to_exclude, str):
        df_drop_trials = trial_df[trial_df['condition'] != conditions_to_exclude].copy()
        df_exclude = trial_df[trial_df['condition'] == conditions_to_exclude].copy()
    elif isinstance(conditions_to_exclude, list):
        df_drop_trials = trial_df[~trial_df['condition'].isin(conditions_to_exclude)].copy()
        df_exclude = trial_df[trial_df['condition'].isin(conditions_to_exclude)].copy()
    elif conditions_to_exclude is None:
        df_drop_trials = trial_df.copy()
        df_exclude = pd.DataFrame()
    else:
        print('Conditions_to_exclude needs to be string or list')
    try:
        df_drop_trials = df_drop_trials.sort_values(by = [ 'animal', 'day_num', 'cue_trial_num'])
    except:
        print("No 'cue_trial_num' col in df, sorting by 'trial_num'")
        df_drop_trials = df_drop_trials.sort_values(by = [ 'animal', 'day_num', 'trial_num'])
    if trials_to_drop == 'consume' :
        consume_mask = df_drop_trials['nlicks_consume'].where(df_drop_trials['nlicks_consume']>0).groupby(df_drop_trials['animal']).ffill()

    no_lick_df = df_drop_trials[consume_mask.notnull()]

    onlyNoLick = df_drop_trials[consume_mask.isnull()]
    df_combined = pd.concat([no_lick_df, df_exclude], axis = 0, ignore_index = True)

    return df_combined #, df_exclude no_lick_df, consume_mask


def calculate_time_to_learn_from_learned_trials(learned_trials_dict, trial_df, data_in_ms = False):

    time_to_learn_dict = dict.fromkeys(learned_trials_dict.keys(),[])
    for condition in learned_trials_dict.keys():
        time_to_learn_dict[condition] = dict.fromkeys(learned_trials_dict[condition].keys(),[])
        for animal, trial in learned_trials_dict[condition].items():
            df_subset = trial_df[((trial_df['animal'] == animal)
                                                            & (trial_df['cue_trial_num'] <= (trial+1))) ]
            total_ITI_before_learning = df_subset['preceding_ITI'].sum()
            total_trial_time_before_learning = (df_subset['preceding_ITI'].count()-1)*df_subset['antic_dur'].median()
            total_time_before_learning = total_ITI_before_learning + total_trial_time_before_learning
            if data_in_ms:
               total_time_before_learning = (total_time_before_learning/1000)
            time_to_learn_dict[condition][animal] = total_time_before_learning

    return time_to_learn_dict

def calculate_rewards_to_learn_from_learned_trials(learned_trials_dict, trial_df, conditions_to_get = 'all', animals_to_exclude = None):
    conditions = get_conditions_as_list(conditions_to_get,
                               trial_df_or_data_dict = trial_df,
                               )
    if animals_to_exclude is not None:
        if isinstance(animals_to_exclude, list):
            exclusions_list = animals_to_exclude
        elif isinstance(animals_to_exclude, str):
            exclusions_list = [animals_to_exclude]
    else:
        exclusions_list = []
    rewards_to_learn_dict = dict.fromkeys(conditions, {})
    for condition in conditions:
        rewards_to_learn_dict[condition] = dict.fromkeys(learned_trials_dict[condition].keys(), [])
        for key, value in learned_trials_dict[condition].items():
            if value:
                if key not in exclusions_list:
                    df_animal_subset = trial_df[((trial_df['animal'] == key) & (trial_df['cue_trial_num'] <= value))].copy()
                    rewards_to_learn = df_animal_subset['reward_del'].sum()
                    rewards_to_learn_dict[condition][key] = rewards_to_learn
    return rewards_to_learn_dict


def predict_trials_to_learn_from_IRI(IRI_to_predict, regression_fit):

    predicted_trials_to_learn = (np.power(10, regression_fit.intercept)
                                *np.power(IRI_to_predict,
                                          regression_fit.slope)
                                )
    return predicted_trials_to_learn

def fit_sigmoid_to_data(x_data, y_data, curve_goes_negative = False):

    def sigmoid(x, L, x0, k, b):
        y = L / (1 + np.exp(-k*(x - x0))) + b
        return y

    p0 = [max(y_data), np.median(x_data), 1, min(y_data)]
    popt, pcov = curve_fit(sigmoid, x_data, y_data, p0, method='trf')
    L, x0, k, b = popt
    y_fit = sigmoid(x_data, L, x0, k, b)
    if curve_goes_negative:
        max_yfit = np.min(y_fit)
        half_rise_trial = x_data[np.flatnonzero((y_fit-y_fit[0])<= ((max_yfit-y_fit[0])/2))[0]]
        nintey5percent_rise_trial = x_data[np.flatnonzero((y_fit-y_fit[0])<= ((max_yfit-y_fit[0])*0.95))[0]]
        results_dict = {'y_fit': y_fit,
                    'half_rise_trial': half_rise_trial,
                    '95%_rise_trial': nintey5percent_rise_trial,
                    'popt': popt,
                    }
    else:
        max_yfit = np.max(y_fit)
        half_rise_trial = x_data[np.flatnonzero((y_fit-y_fit[0]) >= ((max_yfit-y_fit[0])/2))[0]]
        nintey5percent_rise_trial = x_data[np.flatnonzero((y_fit-y_fit[0]) >= ((max_yfit-y_fit[0])*0.95))[0]]
        results_dict = {'y_fit': y_fit,
                    'half_rise_trial': half_rise_trial,
                    '95%_rise_trial': nintey5percent_rise_trial,
                    'popt': popt,
                    }
    return results_dict


def savePickle(variable, filename):
    with open(filename, 'wb') as f:
        pickle.dump(variable, f, -1)


def openPickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    return data

def get_conditions_as_list(conditions_to_get,
                           trial_df_or_data_dict = None,
                           ):
    """
    Parameters
    ----------
    conditions_to_get : str, or list
        DESCRIPTION.
    trial_df_or_data_dict : TYPE, optional
        Dataframe or dictionary of data used to extract condtions from if "all" is called.
        The default is None.

    Raises
    ------
    TypeError
        DESCRIPTION.

    Returns
    -------
    list
        DESCRIPTION.

    """
    if conditions_to_get == 'all':
        if isinstance(trial_df_or_data_dict, dict):
            return list(trial_df_or_data_dict.keys())
        elif isinstance(trial_df_or_data_dict, pd.DataFrame):
            return list(trial_df_or_data_dict['condition'].unique())
        else:
            raise TypeError('Did not pass a dict or dataframe with "all" flag for conditions subsetting')
    if isinstance(conditions_to_get, list):
        return conditions_to_get
    return [conditions_to_get]

def group_and_pivot(df,
                    group_vars,
                    value_col,
                    ):
    grouped = df.groupby(group_vars, as_index=False)[value_col].mean()
    return grouped.pivot(index=group_vars[-1],
                         columns=group_vars[:-1],
                         values=value_col,
                         )

def sort_list_by_key(list_to_sort, list_as_key):
    key_list_dict = {x: index for (index, x) in enumerate(list_as_key)}

    sorted_list = sorted(list_to_sort, key=lambda x: key_list_dict[x] if x in key_list_dict else float('inf'))

    return sorted_list


def binData(dataToBin, binSize, xvalues = False, mean = False):
    bin_size = binSize

    x_bins = (np.arange(len(dataToBin)/bin_size)*bin_size)+bin_size

    if mean:
        binnedData = dataToBin[:(dataToBin.size // bin_size) * bin_size].reshape(-1, bin_size).mean(axis=1)
    else:
        binnedData = dataToBin[:(dataToBin.size // bin_size) * bin_size].reshape(-1, bin_size)

    if xvalues:
        return x_bins, binnedData
    else:
        return binnedData

def check_flags(*flags, allow: str = 'at_least_one'):
    """


    Args:
        *flags (TYPE): boolean-ish flags.
        allow (str, optional): DESCRIPTION. Defaults to 'at_least_one'.
                    - "at_least_one" → require at least one True
                    - "exactly_one" → require exactly one True
                    - "at_most_one" → require at most one True

    Raises:
        ValueError: if flags do not match rule allowed, raise error.

    Returns:
        None.

    """
    flag_count = sum(bool(f) for f in flags)
    if allow == "at_least_one":
        if flag_count < 1:
            raise ValueError('At least one flag must be True.Check {' ,'.join(flags)}')
    elif allow == "exactly_one":
        if flag_count != 1:
            raise ValueError(f'Exactly one flag must be True. Check {' ,'.join(flags)}')
    elif allow == "at_most_one":
        if flag_count > 1:
            raise ValueError('At most one flag can be True.Check {','.join(flags)}')
    else:
        raise ValueError(f'Unknown rule for flag validation: entered {allow}. Must enter "at_least_one", "exactly_one", or "at_most_one"')


def cleanStringForFilename(filename):
    '''


    Parameters
    ----------
    title : string
        take in a string that may contain characters that won't be legal for filenames.
        commas, parenthesies, slashes, periods etc.

    Returns
    -------
    new_title : string
        string without them, mostly replacing with hyphens and spaces.

    '''
    new_filename = filename.lstrip()
    new_filename = new_filename.replace( '\n', '_')
    new_filename = new_filename.replace( '\\', '-')
    new_filename = new_filename.replace( '/', '-')
    new_filename = new_filename.replace( ',', '-')
    new_filename = new_filename.replace( '%', 'percent')
    new_filename = new_filename.replace( ':', '-')
    new_filename = new_filename.replace( '>', ' greater than ')
    new_filename = new_filename.replace( '<', ' less than ')
    new_filename = new_filename.replace( '[', '')
    new_filename = new_filename.replace( ']', '')
    new_filename = new_filename.replace( "'", "")
    return new_filename