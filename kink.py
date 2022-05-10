import numpy as np
import scipy.signal as ss


def detect_kinks(x, signal, thresh=0.1, positiveonly=False, returndiff=False):
    """Detect kinks in the input signal, returning a list of all x locations.
       Assumes an even grid.  Note that this doesn't handle jumps- these 
       should probably be found first, and possibly removed, before kink detection.
       :param x: locations of the signal data points
       :param signal: data values to detect kinks
       :param thresh: relative scale factor for kink detection
       :param returninds: flag to return the kink indices, rather than x locations
    """
    # Data point spacing
    x = np.array(x)
    dx = np.diff(x)
    ds = np.diff(signal)

    # Compute the difference betwen the derivative
    # on the right and on the left of each point
    n = len(ds)
    frwd, bkwd = ds[1:]/dx[1:], ds[:n-1]/dx[:n-1]
    if positiveonly:
        diff = frwd-bkwd
    else:
        diff = abs(frwd-bkwd)
    diff /= max(diff)

    # Run a median filter on the difference signal
    diffmed = ss.medfilt(diff, 11)

    # If the curvature exceeds its local median by enough
    # then we have a kink
    if positiveonly:
        inds = np.where(np.logical_and(diff>thresh,frwd>thresh))[0]
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
        inds = inds+1

    if returndiff:
        return inds, diff
    else:
        return inds
