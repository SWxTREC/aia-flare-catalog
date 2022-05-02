import numpy as np
import scipy.signal as ss
import scipy.ndimage as sn
from datetime import datetime, timedelta
import pandas as pd


def detect_peaks(x, signal, thresh=0.01, positiveonly=False, returndiff=False):
    """Detect kinks in the input signal, returning a list of all x locations.
       Assumes an even grid.  Note that this doesn't handle jumps- these 
       should probably be found first, and possibly removed, before kink detection.
       :param x: locations of the signal data points
       :param signal: data values to detect kinks
       :param thresh: relative scale factor for kink detection
       :param returninds: flag to return the kink indices, rather than x locations
    """
    sos = ss.butter(2,0.5,output='sos')
    if len(signal) > 9:
        signal_smooth = sn.maximum_filter1d(signal,3)
        # signal_smooth = ss.sosfiltfilt(sos,signal)
    else:
        signal_smooth = signal

    # Data point spacing
    x = np.array(x)
    dx = np.diff(x)
    ds = np.diff(signal_smooth)

    # Compute the difference betwen the derivative
    # on the right and on the left of each point
    n = len(ds)
    frwd, bkwd = ds[1:]/dx[1:], ds[:n-1]/dx[:n-1]
    if positiveonly:
        diff = frwd-bkwd
    else:
        diff = abs(frwd-bkwd)
    
    # normalize signal and derivatives
    diff /= max(diff)
    # signal /= max(signal)
    frwd /= max(abs(frwd))
    bkwd /= max(abs(bkwd))

    # Run a median filter on the difference signal
    diffmed = ss.medfilt(diff, 11)
    signalmed = ss.medfilt(signal,121)

    # If both positive curvature and positive derivative
    # then we have a kink
    if positiveonly:
        # print(np.shape([frwd[:-2],frwd[1:-1],frwd[2:]]))
        inds = np.where(np.logical_and(diff[:-2]>0,np.min([frwd[:-2],frwd[1:-1],frwd[2:]],axis=0))>0.02)[0]
        # inds = np.where(np.logical_and(signal[3:-1]-1.3*signal[1:-3]>0,np.logical_and(diff[:-2]>0,np.min([frwd[:-2],frwd[1:-1],frwd[2:]],axis=0)>0)))[0]
    else:
        inds = np.where(diff>diffmed+thresh)[0]
    # inds,_ = ss.find_peaks(diff)
    # inds = inds[diff[inds]>diffmed[inds]+thresh]

    # Check for duplicate entries for kinks off
    # sample boundaries
    dups = np.where(np.diff(inds)==1)[0]
    if len(dups)>0:
        inds = np.delete(inds, dups+1)        

    # If we found kinks, bump the index offset
    if len(inds) != 0:
        inds = inds

    # Now find associated peak and discard extra kink values
    inds_peaks = []
    inds_ends = []
    inds_starts = []
    for ind in inds:
        if ind > 2 and np.min(signal[ind-2:ind])<signal[ind]:
            ind = ind-2+np.argmin(signal[ind-2:ind])
        if len(inds_ends) > 0 and ind < inds_ends[-1]:
            continue
        peak = signal[ind]
        ind_peak = ind
        for j in range(ind+1,np.min([len(x),ind+121])):
            if signal[j] > peak:
                peak = signal[j]
                ind_peak = j
            elif j>ind+2 and signal[j]<=signal[ind]:
                break
            elif (j>ind+2 and peak>signal[ind] and (signal[j] <= (peak - 0.8*(peak-signal[ind])))):
                if peak > 1.1*signalmed[ind]:
                    inds_starts.append(ind)
                    inds_peaks.append(ind_peak)
                    inds_ends.append(j)
                break
                

    if returndiff:
        return inds_starts, inds_peaks, inds_ends
    else:
        return inds

def verify_peak(start_time,peak_time,end_time,aia_data,aia_times,lams,starts,peaks,ends,sharp_flare_data):
    peak_data = {}
    true_peak_inds = np.nan*np.zeros(len(lams))
        
    for l in range(len(lams)):
        times = np.array(aia_times[l])[peaks[l]]
        inds = np.where(np.logical_and(times>=start_time-timedelta(minutes=20), times <= end_time+timedelta(minutes=20)))[0]
        if len(inds)>0:
            ind = inds[np.argmin(np.abs(times[inds]-peak_time))]
            true_peak_inds[l] = ind
            # might need this to actually be the closest time if multiple peaks
            peak_data[lams[l]+'_peak_time'] = aia_times[l][np.array(peaks[l])[ind]]
            peak_data[lams[l]+'_start_time'] = aia_times[l][np.array(starts[l])[ind]]
            peak_data[lams[l]+'_end_time'] = aia_times[l][np.array(ends[l])[ind]]
            peak_data[lams[l]+'_magnitude'] = aia_data[l][np.array(peaks[l])[ind]]
            peak_data[lams[l]+'_prominence'] = aia_data[l][np.array(peaks[l])[ind]]-aia_data[l][np.array(starts[l])[ind]]

        else: # no peak found so add NaN instead
            peak_data[lams[l]+'_peak_time'] = np.nan
            peak_data[lams[l]+'_start_time'] = np.nan
            peak_data[lams[l]+'_end_time'] = np.nan
            peak_data[lams[l]+'_magnitude'] = np.nan
            peak_data[lams[l]+'_prominence'] = np.nan

    start_time = np.min([peak_data[lam+'_start_time'] for lam in lams[2:] if peak_data[lam+'_start_time'] is not np.nan])
    end_time = np.max([peak_data[lam+'_end_time'] for lam in lams[2:] if peak_data[lam+'_end_time'] is not np.nan])
    peak_data['aia_min_start_time'] = start_time
    peak_data['aia_max_end_time'] = end_time

    # find associated flare from GOES
    flare_data = sharp_flare_data[(sharp_flare_data['STARTTIME']<=end_time) & (sharp_flare_data['ENDTIME']>=start_time)]
    # choose largest flare
    if len(flare_data)>0:
        im = np.argmax(flare_data['INTENSITY'])
        peak_data['goes_flare_ind'] = int(flare_data['flare_ind'].iloc[im])
        peak_data['CMX'] = flare_data['CMX'].iloc[im]
        peak_data['CMX_VALUE'] = flare_data['CMX_VALUE'].iloc[im]
        peak_data['goes_magnitude'] = flare_data['INTENSITY'].iloc[im]/10
        peak_data['goes_start_time'] = flare_data['STARTTIME'].iloc[im]
        peak_data['goes_end_time'] = flare_data['ENDTIME'].iloc[im]
    else:
        peak_data['goes_flare_ind'] = None
        peak_data['CMX'] = None
        peak_data['CMX_VALUE'] = None
        peak_data['goes_magnitude'] = None
        peak_data['goes_start_time'] = None
        peak_data['goes_end_time'] = None

    if np.sum(np.isnan(true_peak_inds[2:])) <=1: # corresponds to one of AIA 304,1600,131 not peaking   
        return peak_data
    else:
        return {}

def verify_peaks(aia_data,aia_times,lams,starts,peaks,ends,sharp_flare_data):
    peaks_data = []
    # go through the list of peaks for each wavelength and remove entries where the others don't peak
    for j in range(len(lams)-1,1,-1):
        for k in range(len(peaks[j])):
            start_time = aia_times[j][starts[j][k]]
            peak_time = aia_times[j][peaks[j][k]]
            end_time = aia_times[j][ends[j][k]]
            peak_data = verify_peak(start_time,peak_time,end_time,aia_data,aia_times,lams,starts,peaks,ends,sharp_flare_data)
            if (len(peak_data)>0) and peak_data['aia_min_start_time'] not in [peak['aia_min_start_time'] for peak in peaks_data] and peak_data['aia_max_end_time'] not in [peak['aia_max_end_time'] for peak in peaks_data]: 
            # if len(peak_data)>0 and peak_data not in peaks_data:
                # now we've verified this is a peak and we want to add it to the true peak lists
                peaks_data.append(peak_data)

    return peaks_data

def generate_peak_data(aia_data,aia_times,lams,sharp_flare_data):
    sharp_flare_data['STARTTIME'] = pd.to_datetime('20' + sharp_flare_data['DATE'].astype('str') + '_' + 
                                    sharp_flare_data['START_TIME'].astype('str').str.zfill(4) + '00', format='%Y%m%d_%H%M%S')
    sharp_flare_data['ENDTIME'] = pd.to_datetime('20' + sharp_flare_data['DATE'].astype('str') + '_' + 
                                    sharp_flare_data['END_TIME'].astype('str').str.zfill(4) + '00', format='%Y%m%d_%H%M%S')
    sharp_flare_data['ENDTIME'][sharp_flare_data['ENDTIME']<sharp_flare_data['STARTTIME']] = sharp_flare_data['ENDTIME'][sharp_flare_data['ENDTIME']<sharp_flare_data['STARTTIME']]+timedelta(days=1)

    starts = []
    peaks = []
    ends = []

    # first find all the peaks in all the wavelengths
    for j in range(len(lams)):
        sum_intensity = np.array(aia_data[j])
        times = np.array(aia_times[j])
        if len(sum_intensity) < 3:
            starts.append([])
            peaks.append([])
            ends.append([])
            continue
        dt = [(time-times[0]).total_seconds() for time in times]
        inds_starts,inds_peaks,inds_ends = detect_peaks(dt,sum_intensity,thresh=0.05,positiveonly=True,returndiff=True)
        starts.append(inds_starts)
        peaks.append(inds_peaks)
        ends.append(inds_ends)

    # then verify and associate them with each other
    peaks_data = verify_peaks(aia_data,aia_times,lams,starts,peaks,ends,sharp_flare_data)
    df = pd.DataFrame(peaks_data)
    
    return df

