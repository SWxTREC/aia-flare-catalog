import numpy as np
import scipy.signal as ss
import scipy.ndimage as sn
from datetime import datetime, timedelta
import pandas as pd


def detect_peaks(x, lam, signal, thresh=0.1, positiveonly=False, returndiff=False):
    """Detect kinks and peaks in the input signal, returning a list or lists of indices.
       :param x: locations of the signal data points
       :param lam: wavelength for selecting peak height and prominence parameters
       :param signal: data values to detect peaks
       :param thresh: relative scale factor for kink detection
       :param positiveonly: flag to detect only positive kinks, true for detecting peaks
       :param returndiff: returns start, peak and end indices rather than kink indices
    """
    sos = ss.butter(2,0.5,output='sos')
    if len(signal) > 9:
        # some smoothing of the signal
        signal_smooth = sn.maximum_filter1d(signal,3)
        kernel = ss.cubic([-1.0,0.0,1.0])
        signal_smooth = ss.convolve(signal_smooth,kernel,mode='same',method='auto')
        # signal_smooth = ss.savgol_filter(signal,window_length=5,polyorder=3)
        # signal_smooth = sn.gaussian_filter1d(signal,2.5)
        # signal_smooth = ss.sosfiltfilt(sos,signal)
    else:
        signal_smooth = signal

    if lam == '193':
        height = 2.5e7
        prominence = 2.5e6
    elif lam == '171':
        height = 1.5e7
        prominence = 9e6
    elif lam == '304':
        height = 9e6
        prominence = 5e6
    elif lam == '1600':
        height = 5e6
        prominence = 2e6
    elif lam == '131':
        height = 1e6
        prominence = 2e5
    elif lam == '94':
        height = 3e5
        prominence = 1e5
    else:
        height = 0
        prominence = 0.05*(np.max(signal)-np.median(signal))

    peaks, props = ss.find_peaks(signal_smooth,height=height,prominence=prominence,width=3,rel_height=0.8,distance=5)
    width_heights = props['width_heights']
    left = np.round(props['left_ips']).astype(int)
    right = np.round(props['right_ips']).astype(int)

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
    inds_kinks = np.where(np.logical_and(diff>0,frwd>thresh))[0]
    
    # Check for duplicate entries for kinks off
    # sample boundaries
    dups = np.where(np.diff(inds_kinks)==1)[0]
    if len(dups)>0:
        inds_kinks = np.delete(inds_kinks, dups+1)        

    # Now find adjust start point of peaks based on nearest kink 
    # note that left and right indicate the 80% threshold points on either 
    # side of the peak obtained from the scipy peak finding routine
    inds_peaks = []
    inds_ends = []
    inds_starts = []
    for i in range(len(peaks)):
        if len(inds_kinks[inds_kinks<=left[i]])>0:
            # first check for a kink earlier than the given left point of the peak
            ind_start = inds_kinks[np.argmin(abs(left[i]-inds_kinks[inds_kinks<=left[i]]))]
            if x[ind_start]<x[peaks[i]]-60*60:
                # if that is an hour earlier than the peak, try for the closest kink after the left point
                ind_start = inds_kinks[np.argmin(abs(left[i]-inds_kinks[inds_kinks<peaks[i]]))]
                if x[ind_start]<x[peaks[i]]-60*60:
                    # if that is still an hour earlier, try the closest kink to the peak
                    ind_start = inds_kinks[np.argmin(abs(peaks[i]-inds_kinks[inds_kinks<peaks[i]]))]
                    if x[ind_start]<x[peaks[i]]-60*60:
                        # if still an hour earlier, just take the left point or 45 minutes before
                        ind_start = np.max([left[i],peaks[i]-45])
        elif len(inds_kinks[inds_kinks<peaks[i]])>0:
            # if no kinks earlier than the left point, check for the closest after the left point
            ind_start = inds_kinks[np.argmin(abs(left[i]-inds_kinks[inds_kinks<peaks[i]]))]
            if x[ind_start]<x[peaks[i]]-60*60:
                # still an hour earlier, try the closest kink to the peak
                ind_start = inds_kinks[np.argmin(abs(peaks[i]-inds_kinks[inds_kinks<peaks[i]]))]
                if x[ind_start]<x[peaks[i]]-60*60:
                    # if still an hour earlier, just take the left point or 45 minutes before
                    ind_start = np.max([left[i],peaks[i]-45])
        else:
            # if no kinks before the peak, just take the left point or 45 minutes before
            ind_start = np.max([left[i],peaks[i]-45])
        if signal[ind_start]>signal[peaks[i]]:
            ind_start = left[i]
        
        prominence = signal[peaks[i]]-signal[ind_start]

        if right[i]>peaks[i]+120:
            # if the given right point is more than 2 hours after the peak, double check with the adjusted prominence
            if i < len(peaks)-1:
                ind_end = peaks[i]+np.argmin(np.abs(signal[peaks[i]:np.min([peaks[i+1],peaks[i]+120])]-signal[peaks[i]]+0.8*prominence))
            else:
                ind_end = peaks[i]+np.argmin(np.abs(signal[peaks[i]:np.min([len(signal),peaks[i]+120])]-signal[peaks[i]]+0.8*prominence))
        else:
            ind_end = right[i]

        if signal[ind_start]<np.median(signal)/100:
            # too small to be a peak
            continue
            
        inds_starts.append(ind_start)
        inds_peaks.append(peaks[i])
        inds_ends.append(ind_end)   


    if returndiff:
        return inds_starts, inds_peaks, inds_ends
    else:
        return inds_kinks

def verify_peak(start_time,peak_time,end_time,aia_data,aia_times,lams,starts,peaks,ends,sharp_flare_data):
    # cross reference a peak across wavelengths
    peak_data = {}
    true_peak_inds = np.nan*np.zeros(len(lams))
        
    for l in range(len(lams)):
        times = np.array(aia_times[l])[peaks[l]]
        inds = np.where(np.logical_and(times>=start_time-timedelta(minutes=20), times <= end_time+timedelta(minutes=20)))[0]
        if len(inds)>0:
            ind1 = inds[np.argmin(np.abs(times[inds]-peak_time))]
            ind = inds[np.argmax(np.array(aia_data[l])[np.array(peaks[l])[inds]])]
            if ind1 != ind and np.abs(times[ind]-peak_time)>timedelta(minutes=30):
                ind = ind1
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

    if np.sum(np.isnan(true_peak_inds[2:])) == 4:
        return {}

    start_time = np.max([peak_data[lam+'_start_time'] for lam in lams[2:] if peak_data[lam+'_start_time'] is not np.nan])
    end_time = np.min([peak_data[lam+'_end_time'] for lam in lams[2:] if peak_data[lam+'_end_time'] is not np.nan])
    peak_data['aia_min_start_time'] = start_time
    peak_data['aia_mean_peak_time'] = start_time + np.mean([peak_data[lam+'_peak_time']-start_time for lam in lams if peak_data[lam+'_peak_time'] is not np.nan])
    peak_data['aia_max_end_time'] = end_time

    # find associated flare from GOES
    flare_data = sharp_flare_data[(sharp_flare_data['STARTTIME']<=end_time) & (sharp_flare_data['ENDTIME']>=start_time)]
    # choose closest flare in time
    if len(flare_data)>0:
        im = np.argmin(abs(flare_data['STARTTIME']-peak_data['aia_mean_peak_time']))
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

    if np.sum(np.isnan(true_peak_inds)) <=2: # corresponds to one of AIA 304,1600,131 not peaking   
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
                # now we've verified this is a peak and we want to add it to the true peak lists
                peaks_data.append(peak_data)

    # go through and trim 

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
        inds_starts,inds_peaks,inds_ends = detect_peaks(dt,lams[j],sum_intensity,thresh=0.05,positiveonly=True,returndiff=True)
        starts.append(inds_starts)
        peaks.append(inds_peaks)
        ends.append(inds_ends)

    # then verify and associate them with each other
    peaks_data = verify_peaks(aia_data,aia_times,lams,starts,peaks,ends,sharp_flare_data)
     
    df = pd.DataFrame(peaks_data)
    
    return df

