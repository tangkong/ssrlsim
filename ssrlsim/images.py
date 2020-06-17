'''
Contains functions used to generate images for various detectors. 
Much taken from bluesky tutorial materials  
'''
import numpy as np

# General functions
def gaussian(x, c=0, sig=1, amp=None):
    if amp is None:
        return (
            1.0
            / (np.sqrt(2.0 * np.pi) * sig)
            * np.exp(-np.power((x - c) / sig, 2.0) / 2)
        )
    else:
        return amp * np.exp(-np.power((x - c) / sig, 2.0) / 2)


def make_random_peaks(
    x, xmin=None, xmax=None, peak_chance=0.1, return_pristine_peaks=False
):
    """make_random_peaks randomly generates gaussian peaks and produces a 1D 
    diffraction pattern
    """
    # select boundaries for peaks
    if xmin is None:
        xmin = np.percentile(x, 10)
    if xmax is None:
        xmax = np.percentile(x, 90)

    y = np.zeros(len(x))

    # make peak positions
    peak_pos = np.array(np.random.random(len(x)) < peak_chance)
    peak_pos[x < xmin] = False
    peak_pos[x > xmax] = False

    for peak_idex in [i for i, x in enumerate(peak_pos) if x]:
        y += gaussian(x, c=x[peak_idex], sig=0.05, amp=(1 / x[peak_idex]) ** 0.5)

    # now for any diffuse low-Q component
    y += gaussian(x, c=0, sig=3, amp=0.5)

    return y

def generate_image(x, intensity, shape):
    """
    Given a 1D array of intensity, generate a 2D diffraction image.
    """
    xL, yL = shape[0] // 2, shape[1] // 2  # half-lengths of each dimension
    x_, y_ = np.mgrid[-xL:xL, -yL:yL]
    ordinal_r = np.hypot(x_, y_)
    unit_r = ordinal_r / ordinal_r.max()
    r = unit_r * x.max()
    return np.interp(r, x, intensity)