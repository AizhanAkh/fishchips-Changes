from matplotlib.patches import Ellipse
import numpy as np
import matplotlib.pyplot as plt
import fishchips.util

ALPHA1 = 1.52
ALPHA2 = 2.48
ALPHA3 = 3.44
PLOT_MULT = 4.
def get_ellipse(par1, par2, params, cov, scale1=1, scale2=1):
    """
    Extract ellipse parameters from covariance matrix.
    Parameters
    ----------
        par1 (string): name of parameter 1
        par2 (string): name of parameter 2
        params (list of strings): contains names of parameters to constrain
        cov (numpy array): covariance matrix
    Return
    ------
        tuple, ellipse a, b, angle in degrees, sigma_x, sigma_y, sigma_xy
    """
    # equations 1-4 Coe 2009. returns in degrees
    # first look up indices of parameters
    pind = dict(zip(params, list(range(len(params)))))
    i1 = pind[par1]
    i2 = pind[par2]
    sigma_x2 = cov[i1, i1] * scale1*scale1
    sigma_y2 = cov[i2, i2] * scale2*scale2
    sigma_xy = cov[i1, i2] * scale1*scale2

    if ((sigma_y2/sigma_x2) < 1e-10) or ((sigma_x2/sigma_y2) < 1e-10):
        a2 = max(sigma_x2, sigma_y2) + sigma_xy**2 / max(sigma_x2, sigma_y2)
        b2 = min(sigma_x2, sigma_y2) - sigma_xy**2 / max(sigma_x2, sigma_y2)
    else:
        a2 = (sigma_x2+sigma_y2)/2. + np.sqrt((sigma_x2 - sigma_y2)**2/4. +
                                              sigma_xy**2)
        b2 = (sigma_x2+sigma_y2)/2. - np.sqrt((sigma_x2 - sigma_y2)**2/4. +
                                              sigma_xy**2)
    angle = np.arctan(2.*sigma_xy/(sigma_x2-sigma_y2)) / 2.
    if (sigma_x2 < sigma_y2):
        a2, b2 = b2, a2

    return np.sqrt(a2), np.sqrt(b2), angle * 180.0 / np.pi, \
            np.sqrt(sigma_x2), np.sqrt(sigma_y2), sigma_xy

def plot_ellipse(ax, par1, par2, parameters_set, fiducial_set, cov, color='black',
                 resize_lims=True, positive_definite=[], one_sigma_only=False,
                 scale1=1, scale2=1, ls1='--', ls2='-'):
    """
    Plot 1 and 2-sigma ellipses, from Coe 2009.
    Parameters
    ----------
        ax (matpotlib axis): axis upon which the ellipses will be drawn
        par1 (string): parameter 1 name
        par2 (string): parameter 2 name
        obs (Observables object): contains names of parameters to constrain, etc
        cov (numpy array): covariance matrix
        color (string): color to plot ellipse with
        resize_lims (boolean): flag for changing the axis limits
        positive_definite (list of string): convenience input,
            parameter names passed in this list will be cut off at 0 in plots.
        scale1 and scale2 are for plotting scale
    Returns
    -------
        list of float : sigma_x, sigma_y, sigma_xy for judging the size of the
            plotting window
    """
    params = parameters_set
    pind = dict(zip(params, list(range(len(params)))))
    i1 = pind[par1]
    i2 = pind[par2]
    a, b, theta, sigma_x, sigma_y, sigma_xy = get_ellipse(
        par1, par2, params, cov, scale1, scale2)

    fid1 = fiducial_set[i1] * scale1
    fid2 = fiducial_set[i2] * scale2

    if not one_sigma_only:
        e1 = Ellipse(
            xy=(fid1, fid2),
            width=a * 2 * ALPHA2, height=b * 2 * ALPHA2,
            angle=theta, edgecolor=color, lw=2, facecolor='none', ls=ls2)
        ax.add_artist(e1)
        e1.set_clip_box(ax.bbox)

    # 1-sigma ellipse
    e2 = Ellipse(
        xy=(fid1, fid2),
        width=a*2*ALPHA1, height=b*2*ALPHA1,
        angle=theta, edgecolor=color, lw=2, facecolor='none', ls=ls1)
    ax.add_artist(e2)
    e2.set_alpha(1.0)
    e2.set_clip_box(ax.bbox)

    if resize_lims:
        if par1 in positive_definite:
            ax.set_xlim(max(0.0, -PLOT_MULT*sigma_x),
                        fid1+PLOT_MULT*sigma_x)
        else:
            ax.set_xlim(fid1 - PLOT_MULT * sigma_x,
                        fid1 + PLOT_MULT * sigma_x)
        if par2 in positive_definite:
            ax.set_ylim(max(0.0, fid2 - PLOT_MULT * sigma_y),
                        fid2 + PLOT_MULT * sigma_y)
        else:
            ax.set_ylim(fid2 - PLOT_MULT * sigma_y,
                        fid2 + PLOT_MULT * sigma_y)

    return sigma_x, sigma_y, sigma_xy


def plot_ell(parameters_set, fiducial_set, cov, positive_definite = [], color='black'):

    nparams = len(parameters_set)
    scales = np.ones(nparams)

    print('generating new axis')
    f, ax = plt.subplots(nparams, nparams, figsize=(12, 12))


    labels = [(r'$\mathrm{' + p.replace('_', r'\_') + r'}$')
              for p in parameters_set]
    print(labels)
    # stitch together axes to row=nparams-1 and col=0
    # and turn off non-edge
    for ii in range(nparams):
        for jj in range(nparams):
            if ii == jj:
                ax[jj, ii].get_yaxis().set_visible(False)
                if ii < nparams-1:
                    ax[jj, ii].get_xaxis().set_ticks([])

            if ax[jj, ii] is not None:
                if ii < jj:
                    if jj < nparams-1:
                        ax[jj, ii].set_xticklabels([])
                if ii > 0:
                        ax[jj, ii].set_yticklabels([])

                if jj > 0:
                    # stitch axis to the one above it
                    if ax[0, ii] is not None:
                        ax[jj, ii].get_shared_x_axes().join(ax[jj, ii], ax[0, ii])
                elif ii < nparams-1:
                    if ax[jj, nparams-1] is not None:
                        ax[jj, ii].get_shared_y_axes().join(ax[jj, ii], ax[jj, nparams-1])
                        
    # call plot_ellipse
    for ii in range(nparams):
        for jj in range(nparams):
            if ax[jj, ii] is not None:
                if ii < jj:
                    plot_ellipse(ax[jj, ii], parameters_set[ii],
                                    parameters_set[jj], parameters_set, fiducial_set, cov, color=color,
                                    positive_definite=positive_definite,
                                    scale1=scales[ii], scale2=scales[jj])
                    if jj == nparams-1:
                        ax[jj, ii].set_xlabel(labels[ii], labelpad=30)
                        for tick in ax[jj, ii].get_xticklabels():
                            tick.set_rotation(45)
                    if ii == 0:
                        ax[jj, ii].set_ylabel(labels[jj])
                elif ii == jj:
                    # plot a gaussian if we're on the diagonal
                    sig = np.sqrt(cov[ii, ii])
                    if parameters_set[ii] in positive_definite:
                        grid = np.linspace(
                            fiducial_set[ii],
                            fiducial_set[ii] + PLOT_MULT * sig, 100)
                    else:
                        grid = np.linspace(
                            fiducial_set[ii] - PLOT_MULT*sig,
                            fiducial_set[ii] + PLOT_MULT*sig, 100)
                    posmult = 1.0
                    if parameters_set[ii] in positive_definite:
                        posmult = 2.0
                    ax[jj, ii].plot(grid,
                                    posmult * np.exp(
                                        -(grid-fiducial_set[ii])**2 /
                                        (2 * sig**2)) / (sig * np.sqrt(2*np.pi)),
                                    '-', color=color)
                    if ii == nparams-1:
                        ax[jj, ii].set_xlabel(labels[ii], labelpad=30)
                else:
                    ax[jj, ii].axis('off')