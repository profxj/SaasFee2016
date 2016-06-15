#Module for the figures of COS-Halos Patchup

# Imports
from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
import glob, os, sys, json
import warnings
import h5py
import pickle

import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

from astropy import units as u
from astropy.units import Quantity
from astropy.table import Table

from linetools.abund import ions as lai
from linetools.abund.elements import ELEMENTS
from linetools.analysis import absline as ltaa
from linetools.spectra.plotting import get_flux_plotrange
from linetools.spectralline import AbsLine
from linetools.analysis import voigt as ltav

from pyigm.cgm.cos_halos import COSHalos
from pyigm.cgm import cos_halos as pch
from pyigm.metallicity.pdf import MetallicityPDF

#from xastropy.plotting import utils as xputils
from xastropy.xutils import xdebug as xdb
from xastropy.plotting import utils as xputils

# Local
#sys.path.append(os.path.abspath("../Analysis/py"))
#import coshalo_lls as chlls

#def fig_ionic_lls

def fig_lya_lines(outfil='Figures/fig_lya_lines.pdf'):
    """ Plot the LLS models

    Parameters
    ----------

    Returns
    -------

    """
    # 
    # Initialize
    ms = 7.
    
    # Start the plot
    if outfil is not None:
        pp = PdfPages(outfil)

    # Dummy line
    lya = AbsLine(1215.6700*u.AA)
    lya.attrib['N'] = 10.**(13.6)/u.cm**2
    b0 = 20
    lya.attrib['b'] = b0 * u.km/u.s
    lya.attrib['z'] = 0.

    # Wavelength array
    wave = np.linspace(1200., 1230., 10000) * u.AA

    fig = plt.figure(figsize=(8, 5))
    plt.clf()
    gs = gridspec.GridSpec(1, 2)

    lsz = 12.
    # NHI varies first
    ax = plt.subplot(gs[0])
    NHIs = [12., 13., 14., 15.]
    # Loop
    for jj, NHI in enumerate(NHIs):
        lya.attrib['N'] = 10.**(NHI)/u.cm**2
        f_obs = ltav.voigt_from_abslines(wave, [lya])

        # Plot
        ax.plot(f_obs.wavelength, f_obs.flux, '-', label=r'log $N_{\rm HI}$ '+'= {:g}'.format(NHI))

    legend = plt.legend(loc='upper right', scatterpoints=1, borderpad=0.3,
                        handletextpad=0.3, fontsize='small', numpoints=1, 
                        title=r'$b = $'+'{:g} km/s'.format(b0))
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    x_formatter = mpl.ticker.ScalarFormatter(useOffset=False)
    ax.xaxis.set_major_formatter(x_formatter)
    ax.minorticks_on()
    ax.set_xlabel('Wavelength (Ang)')
    ax.set_xlim(1214.8, 1216.6)
    ax.set_ylabel('Normalized Flux')

    # Now b
    ax = plt.subplot(gs[1])
    N0 = 14.
    lya.attrib['N'] = 10.**(N0)/u.cm**2
    bvals = [10., 30., 50.]
    # Loop
    for jj, bval in enumerate(bvals):
        lya.attrib['b'] = bval*u.km/u.s
        f_obs = ltav.voigt_from_abslines(wave, [lya])

        # Plot
        velo = f_obs.relative_vel(lya.wrest)
        ax.plot(velo, f_obs.flux, '-', label=r'log $N_{\rm HI}$ '+'= {:g}'.format(NHI))
        #ax.plot(f_obs.wavelength, f_obs.flux, '-', label=r'$b$ '+'= {:g} km/s'.format(bval))

    legend = plt.legend(loc='lower right', scatterpoints=1, borderpad=0.3,
                        handletextpad=0.3, fontsize='small', numpoints=1, 
                        title=r'log $N_{\rm HI} =$'+'{:g}'.format(N0))
    ax.set_xlabel('Relative Velocity (km/s)')
    ax.set_xlim(-150., 150.)
    ax.set_ylabel('Normalized Flux')

    # Layout and save
    print('Writing {:s}'.format(outfil))
    plt.tight_layout(pad=0.2,h_pad=0.1,w_pad=0.2)
    pp.savefig(bbox_inches='tight')
    plt.close()
    # Finish
    pp.close()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def fig_dla_compare(outfil='Figures/fig_dla_compare.pdf'):
    """ Plot the DLA

    Parameters
    ----------

    Returns
    -------

    """
    # 
    # Initialize
    ms = 7.
    
    # Start the plot
    if outfil is not None:
        pp = PdfPages(outfil)

    # Dummy line
    lya = AbsLine(1215.6700*u.AA)
    lya.attrib['N'] = 10.**(13.6)/u.cm**2
    b0 = 20
    lya.attrib['b'] = b0 * u.km/u.s
    lya.attrib['z'] = 0.


    # Wavelength array
    wave = np.linspace(1180., 1250., 20000) * u.AA

    fig = plt.figure(figsize=(5, 5))
    plt.clf()
    gs = gridspec.GridSpec(1, 1)

    lsz = 12.
    # NHI varies first
    ax = plt.subplot(gs[0])
    NHIs = [12., 14., 21]
    # Loop
    for jj, NHI in enumerate(NHIs):
        lya.attrib['N'] = 10.**(NHI)/u.cm**2
        f_obs = ltav.voigt_from_abslines(wave, [lya])

        # Plot
        ax.plot(f_obs.wavelength, f_obs.flux, '-', label=r'log $N_{\rm HI}$ '+'= {:g}'.format(NHI))

    legend = plt.legend(loc='lower right', scatterpoints=1, borderpad=0.3,
                        handletextpad=0.3, fontsize='small', numpoints=1, 
                        title=r'$b = $'+'{:g} km/s'.format(b0))
    #ax.xaxis.set_major_locator(plt.MultipleLocator(1.))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    x_formatter = mpl.ticker.ScalarFormatter(useOffset=False)
    ax.xaxis.set_major_formatter(x_formatter)
    ax.minorticks_on()
    ax.set_xlabel('Wavelength (Ang)')
    ax.set_xlim(1180., 1250.)
    ax.set_ylabel('Normalized Flux')

    # Layout and save
    print('Writing {:s}'.format(outfil))
    plt.tight_layout(pad=0.2,h_pad=0.1,w_pad=0.2)
    pp.savefig(bbox_inches='tight')
    plt.close()
    # Finish
    pp.close()


def fig_diff_NHI(cos_halos=None, outfil='fig_diff_NHI.pdf'):
    """ Compares original NHI with new
    """
    # Read LLS
    jfile = '../Analysis/COS_Halos_LLS.json'
    with open(jfile) as json_file:
        fdict = json.load(json_file)

    # Read COS-Halos
    if cos_halos is None:
        cos_halos = COSHalos()
        cos_halos.load_mega(skip_ions=True)
    orig_NHI = cos_halos.NHI
    orig_fNHI = cos_halos.flag_NHI

    # Init
    NHI_mnx = (14., 21.)

    # Start the plot
    if outfil is not None:
        pp = PdfPages(outfil)

    plt.figure(figsize=(5, 5))
    plt.clf()
    gs = gridspec.GridSpec(1, 1)

    ax = plt.subplot(gs[0])

    # Axes
    #wvmnx = awvmnx[iuvq['instr']]
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(500.))
    #ax.yaxis.set_major_locator(plt.MultipleLocator(1.))
    ax.minorticks_on()
    ax.set_xlim(NHI_mnx)
    ax.set_ylim(NHI_mnx)

    # Labels
    ax.set_xlabel(r'Original: $\log_{10} \, N_{\rm HI}$')
    ax.set_ylabel(r'New: $\log_{10} \, N_{\rm HI}$')

    # Loop on Systems
    for key in fdict.keys():
        # Original
        mt = np.where((cos_halos.field==fdict[key]['quasar']) &
                      (cos_halos.gal_id==fdict[key]['galaxy']))[0]
        if len(mt) != 1:
            raise ValueError("Not a proper match")
        # Plot
        if orig_fNHI[mt] == 1:
            warnings.warn("Assuming the value for {:s}_{:s} is a lower limit".format(
                fdict[key]['quasar'],fdict[key]['galaxy']))
            #ax.quiver(orig_NHI[mt], fdict[key]['fit_NHI'][0], 1, 0,
            #          units='x', width=0.03, headwidth=2., headlength=5.)
        elif orig_fNHI[mt] == 2:
            pass
            #ax.quiver(orig_NHI[mt], fdict[key]['fit_NHI'][0], 1, 0,
            #          units='x', width=0.03, headwidth=2., headlength=5.)
        elif orig_fNHI[mt] == 3:
            raise ValueError("Very Unlikely")
        else:
            raise ValueError("Nope")
        # New
        #xdb.set_trace()
        yerr=[[fdict[key]['fit_NHI'][0]-fdict[key]['fit_NHI'][1]],
              [fdict[key]['fit_NHI'][2]-fdict[key]['fit_NHI'][0]]]
        ax.errorbar([orig_NHI[mt]], [fdict[key]['fit_NHI'][0]], yerr=yerr,
                    capthick=2, fmt='o')

    # 1-1 line
    ax.plot(NHI_mnx, NHI_mnx, ':', color='gray')

    # End
    print('Writing {:s}'.format(outfil))
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    pp.savefig()
    pp.close()
    plt.close()


def fig_z_vs_dNHI(cos_halos=None, outfil='fig_z_vs_dNHI.pdf'):
    """ Shows change in NHI vs. z
    """
    # Read LLS
    jfile = '../Analysis/COS_Halos_LLS.json'
    with open(jfile) as json_file:
        fdict = json.load(json_file)
    keys = fdict.keys()
    nsys = len(keys)
    zabs = np.array([fdict[key]['z'] for key in keys])
    srt = np.argsort(zabs)


    # Read COS-Halos
    if cos_halos is None:
        cos_halos = load_ch(load_mtl=False)
    orig_NHI = cos_halos.NHI
    orig_fNHI = cos_halos.flag_NHI
    werk_NHI = cos_halos.werk14_NHI

    # Init
    NHI_mnx = (14.8, 19.1)

    # Start the plot
    if outfil is not None:
        pp = PdfPages(outfil)

    plt.figure(figsize=(5, 5))
    plt.clf()
    gs = gridspec.GridSpec(1, 1)

    ax = plt.subplot(gs[0])

    # Axes
    #wvmnx = awvmnx[iuvq['instr']]
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(500.))
    #ax.yaxis.set_major_locator(plt.MultipleLocator(1.))
    ax.minorticks_on()
    ax.set_xlim(NHI_mnx)
    ax.set_ylim(0, nsys+1)

    # Labels
    lsz = 16.
    ax.set_xlabel(r'$\log_{10} \, N_{\rm HI}$', size=lsz)
    ax.set_ylabel(r'System (Ordered by $z$)', size=lsz)
    new_lbl = False

    # Loop on Systems
    for jj, isrt in enumerate(srt):
        key = keys[isrt]
        # Original
        mt = np.where((cos_halos.field == fdict[key]['quasar']) &
                      (cos_halos.gal_id == fdict[key]['galaxy']))[0]
        if len(mt) != 1:
            raise ValueError("Not a proper match")
        # Plot original
        if orig_fNHI[mt] == 1:
            warnings.warn("Assuming the value for {:s}_{:s} is a lower limit".format(
                fdict[key]['quasar'],fdict[key]['galaxy']))
            #ax.quiver(orig_NHI[mt], fdict[key]['fit_NHI'][0], 1, 0,
            #          units='x', width=0.03, headwidth=2., headlength=5.)
        elif orig_fNHI[mt] == 2:
            pass
            #ax.quiver(orig_NHI[mt], fdict[key]['fit_NHI'][0], 1, 0,
            #          units='x', width=0.03, headwidth=2., headlength=5.)
        elif orig_fNHI[mt] == 3:
            xdb.set_trace()
            raise ValueError("Very Unlikely")
        else:
            xdb.set_trace()
            raise ValueError("Nope")

        # Tumlinson NHI
        if jj == 0:
            lbl = 'Tumlinson+13'
        else:
            lbl = None
        ax.scatter([orig_NHI[mt]], [jj+1], marker='o', edgecolor='gray',
                   facecolor='none', label=lbl)

        # Werk+14 NHI
        if jj == 0:
            lbl2 = 'Werk+14'
        else:
            lbl2 = None
        ax.scatter([werk_NHI[mt]], [jj+1], marker='s', edgecolor='red',
                   facecolor='none', label=lbl2)

        # New value
        if fdict[key]['flag_NHI'] == 1:
            xerr=[[fdict[key]['fit_NHI'][0]-fdict[key]['fit_NHI'][1]],
                  [fdict[key]['fit_NHI'][2]-fdict[key]['fit_NHI'][0]]]
            if new_lbl is False:
                lbl = 'New'
                new_lbl = True
            else:
                lbl = None
            ax.errorbar([fdict[key]['fit_NHI'][0]], [jj+1], xerr=xerr,
                        capthick=2, fmt='o', color='blue', label=lbl)
        elif fdict[key]['flag_NHI'] == 2:  # Lower limit
            ax.plot([fdict[key]['fit_NHI'][1]], [jj+1], '>', color='blue')
        elif fdict[key]['flag_NHI'] == 3:  # Upper limit
            ax.plot([fdict[key]['fit_NHI'][2]], [jj+1], '<', color='blue')

    legend = plt.legend(loc='upper right', scatterpoints=1, borderpad=0.3,
                        handletextpad=0.3, fontsize='small', numpoints=1)

    # End
    xputils.set_fontsize(ax,15.)
    print('Writing {:s}'.format(outfil))
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    pp.savefig()
    pp.close()
    plt.close()


def fig_sngl_metPDF(system):
    """ Shows a single metallicity PDF
    """
    from matplotlib import transforms

    # Read Metallicity input
    mfile = '../Analysis/COS_Halos_MTL.ascii'
    data = Table.read(mfile, format='ascii')
    mtl_inps = np.where(data['name'] == system[0]+'_'+system[1])[0]

    ion_lst = []
    clrs = []
    cdict = {0:'black', -1:'red', -2:'green'}
    for inp in mtl_inps:
        ion_lst.append(data[inp]['ion'])
        clrs.append(cdict[data[inp]['flag']])

    # Read COS-Halos
    cos_halos = COSHalos()
    cos_halos.load_single_fits(system, skip_ions=True)
    orig_NHI = cos_halos.NHI
    orig_fNHI = cos_halos.flag_NHI

    # Read PDF
    mtlfil = '../Analysis/COS_Halos_MTL.hdf5'
    fh5 = h5py.File(mtlfil,'r')
    mPDF = MetallicityPDF(fh5['met']['left_edge_bins']+
                                     fh5['met']['left_edge_bins'].attrs['BINSIZE']/2.,
                                     fh5['met'][system[0]+'_'+system[1]])

    # KLUDGE
    print("BIG KLUDGE HERE")
    bad = np.where(mPDF.ZH < -2)[0]
    mPDF.pdf_ZH[bad] = 0.
    mPDF.normalize()

    # Init
    NHI_mnx = (14.8, 19.1)

    # Start the plot
    outfil = 'fig_metPDF_'+system[0]+'_'+system[1]+'.pdf'
    pp = PdfPages(outfil)

    fig = plt.figure(figsize=(7, 4))
    plt.clf()
    ax = plt.gca()

    # Axes
    #wvmnx = awvmnx[iuvq['instr']]
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(500.))
    #ax.yaxis.set_major_locator(plt.MultipleLocator(1.))
    #plt.ylim(0, nsys+1)
    ax.set_xlim(-2,1)

    # Labels
    lsz = 19.

    # Giddy up
    ax.bar(mPDF.ZH-mPDF.dZH/2., mPDF.pdf_ZH, width=mPDF.dZH)
    ax.set_xlabel("[Z/H]", size=lsz)
    ax.set_ylabel("PDF", size=lsz)

    # Label system
    csz = 18.
    lbl = '{:s}_{:s}'.format(system[0], system[1])
    ax.text(0.1, 0.87, lbl, transform=ax.transAxes, color='black',
            size=csz, ha='left')

    # Ions
    x = -1.8
    ymnx = ax.get_ylim()
    y = 0.75 * ymnx[1]
    isz = 14.
    t = plt.gca().transData
    for s,c in zip(ion_lst,clrs):
        text = plt.text(x,y," "+s+" ",color=c, transform=t, size=isz)
        text.draw(fig.canvas.get_renderer())
        ex = text.get_window_extent()
        t = transforms.offset_copy(text._transform, x=ex.width*0.9, units='dots')

    # End
    xputils.set_fontsize(ax, 16.)
    print('Writing {:s}'.format(outfil))
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    pp.savefig()
    pp.close()
    plt.close()
"""                         """



def fig_full_nH_PDF(cos_halos=None, outfil = 'fig_full_nH_PDF.pdf'):
    """ Shows the combined PDF
    """
    # Read COS-Halos
    if cos_halos is None:
        cos_halos = load_ch()

    # Init
    nH_mnx = (-5., 1)

    # Start the plot
    pp = PdfPages(outfil)

    plt.figure(figsize=(5, 5))
    plt.clf()
    ax = plt.gca()

    # Axes
    #wvmnx = awvmnx[iuvq['instr']]
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(500.))
    #ax.yaxis.set_major_locator(plt.MultipleLocator(1.))
    #plt.ylim(0, nsys+1)
    ax.set_xlim(nH_mnx)


    sumPDF = None
    # Loop on the systems
    nsys = 0
    for cgm_abs in cos_halos.cgm_abs:
        # Cut on quality
        qual = chmtl.mtl_quality(cgm_abs)
        if qual < 1:
            continue
        # Sum the PDF
        nsys += 1
        if sumPDF is None:
            sumPDF = cgm_abs.igm_sys.density
        else:
            sumPDF = sumPDF + cgm_abs.igm_sys.density

    print("We are using {:d} systems in COS-Halos".format(nsys))
    sumPDF.normalize()

    # Giddy up
    lsz = 19.
    ax.bar(sumPDF.nH-sumPDF.dnH/2., sumPDF.pdf_nH, width=sumPDF.dnH, color='green')
    ax.set_xlabel(r"$\log \, n_{\rm H}$", size=lsz)
    ax.set_ylabel("Normalized PDF", size=lsz)

    # Label
    csz = 18.
    lbl = 'COS-Halos'
    ax.text(0.05, 0.87, lbl, transform=ax.transAxes, color='black', size=csz, ha='left')
    lbl2 = '({:d} systems)'.format(nsys)
    ax.text(0.05, 0.75, lbl2, transform=ax.transAxes, color='black', size=csz, ha='left')

    # End
    xputils.set_fontsize(ax, 16.)
    print('Writing {:s}'.format(outfil))
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    pp.savefig()
    pp.close()
    plt.close()


def fig_x_vs_ZH(x, xmnx, xlbl, cos_halos=None, msk=None,
                outfil = None, lsz=16., ax=None, log=None):
    """ Plot ZH vs. a input quantity
    """
    # Read COS-Halos
    if cos_halos is None:
        cos_halos = load_ch()

    # Init
    ZH_mnx = (-2., 1)

    # Start the plot
    if ax is None:
        pp = PdfPages(outfil)
        plt.figure(figsize=(5, 5))
        plt.clf()
        ax = plt.gca()

    # Axes
    #wvmnx = awvmnx[iuvq['instr']]
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(500.))
    #ax.yaxis.set_major_locator(plt.MultipleLocator(1.))
    #plt.ylim(0, nsys+1)
    ax.set_xlim(xmnx)
    ax.set_ylim(ZH_mnx)


    # Loop on the systems
    for ii, cgm_abs in enumerate(cos_halos.cgm_abs):
        if msk is None:
            # Cut on quality
            try:
                flgs = cgm_abs.igm_sys.metallicity.inputs['data'][:, 3]
            except IndexError:
                print('No constraint for {:s}'.format(cgm_abs.name))
                print('Skipping')
                continue
            indices = np.where(flgs != "-1")[0]
            if len(indices) == 0:
                print('No constraint for {:s}'.format(cgm_abs.name))
                print('Skipping')
                continue
        else:
            if msk[ii] is False:
                continue
        #
        ZH = cgm_abs.igm_sys.metallicity.medianZH
        sig = cgm_abs.igm_sys.metallicity.confidence_limits(0.68)
        yerr = [[ZH-sig[0]], [sig[1]-ZH]]
        if np.sum(np.array(yerr)) > 1.:
            pcolor='lightgray'
            eclr = 'none'
        else:
            pcolor='blue'
            fclr ='blue'
            eclr = 'black'

        if x == 'NHI':
            xval = cgm_abs.igm_sys.NHIPDF.median
        elif x == 'nH':
            xval = cgm_abs.igm_sys.density.median
        else:
            xval = getattr(cgm_abs, x)
            if isinstance(xval, Quantity):
                xval = xval.value
        if log:
            xval = np.log10(xval)
        #if (x == 'NHI') and (xval < 16) and (ZH < -1):
        #    xdb.set_trace()
        ax.errorbar([xval], [ZH], yerr=yerr, capthick=2, fmt='o', color=pcolor)
                    #edgecolor=eclr)

    # Labels
    ax.set_xlabel(xlbl)
    ax.set_ylabel("[Z/H]")
    ax.plot(xmnx, [0., 0.], '--', color='green')

    # End
    xputils.set_fontsize(ax, lsz)
    if outfil is not None:
        print('Writing {:s}'.format(outfil))
        plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
        pp.savefig()
        pp.close()
        plt.close()


def fig_NH_hist(cos_halos=None, outfil = 'fig_NH_hist.pdf', lsz=17.):
    """ Shows a histogram of NH values
    """
    # Read COS-Halos
    if cos_halos is None:
        cos_halos = load_ch()

    # Read the grid
    model = os.getenv('DROPBOX_DIR')+'/cosout/grid_minextended.pkl'
    fil=open(model)
    modl=pickle.load(fil)
    fil.close()

    # Init
    NH_mnx = (18., 23)
    ymnx = (0., 9)
    binsz = 0.25
    nbins = (NH_mnx[1]-NH_mnx[0])/binsz

    # Start the plot
    pp = PdfPages(outfil)

    plt.figure(figsize=(5, 5))
    plt.clf()
    ax = plt.gca()

    # Axes
    #wvmnx = awvmnx[iuvq['instr']]
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(500.))
    #ax.yaxis.set_major_locator(plt.MultipleLocator(1.))
    #plt.ylim(0, nsys+1)
    ax.set_xlim(NH_mnx)
    ax.set_ylim(ymnx)


    sumPDF = None
    # Loop on the systems
    nsys = 0
    NH_val = []
    for cgm_abs in cos_halos.cgm_abs:
        # Cut on quality
        try:
            flgs = cgm_abs.igm_sys.metallicity.inputs['data'][:, 3]
        except IndexError:
            print('No constraint for {:s}'.format(cgm_abs.name))
            print('Skipping')
            continue
        indices = np.where(flgs != "-1")[0]
        if len(indices) == 0:
            print('No constraint for {:s}'.format(cgm_abs.name))
            print('Skipping')
            continue
        # Find best indices
        mtl = cgm_abs.igm_sys.metallicity.medianZH
        imet = np.argmin(np.abs(modl[1]['met']-mtl))
        dens = cgm_abs.igm_sys.density.mediannH
        idens = np.argmin(np.abs(modl[1]['dens']-dens))
        ired = np.argmin(np.abs(modl[1]['red']-cgm_abs.zabs))
        icol = np.argmin(np.abs(modl[1]['col']-cgm_abs.igm_sys.NHI))
        # xHI, NH
        xHI = modl[2]['HI'][icol][idens][imet][ired]
        NH = cgm_abs.igm_sys.NHI - xHI
        NH_val.append(NH)

    print("We are using {:d} systems in COS-Halos".format(nsys))

    # Histogram
    lsz = 19.
    hist, edges = np.histogram(np.array(NH_val), range=NH_mnx, bins=nbins)
    ax.bar(edges[:-1], hist, width=binsz, color='green')
    ax.set_xlabel(r'$\log \, N_{\rm H}$')
    ax.set_ylabel(r"$N_{\rm sys}$")

    # Label
    prev_medNH = 19.1  # Eyeball
    ax.plot([prev_medNH]*2, ymnx, '--', color='red')
    '''
    csz = 18.
    lbl = 'COS-Halos'
    ax.text(0.05, 0.87, lbl, transform=ax.transAxes, color='black', size=csz, ha='left')
    lbl2 = '({:d} systems)'.format(nsys)
    ax.text(0.05, 0.75, lbl2, transform=ax.transAxes, color='black', size=csz, ha='left')
    '''

    # End
    xputils.set_fontsize(ax, lsz)
    print('Writing {:s}'.format(outfil))
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    pp.savefig()
    pp.close()
    plt.close()
"""                                         """

def fig_mtlPDF_cuts(cos_halos=None, outfil = 'fig_mtlPDF_cuts.pdf',
                    qcut=1):
    """ Shows the metallicity PDF for various cuts
    Avoids 0 detection systems
    """
    # Read COS-Halos
    if cos_halos is None:
        cos_halos = load_ch()

    # Quality cut
    msk = [False]*len(cos_halos.cgm_abs)
    for ii,cgm_abs in enumerate(cos_halos.cgm_abs):
        # Cut on quality
        if chmtl.mtl_quality(cgm_abs) <= qcut:
            print('No constraint for {:s}'.format(cgm_abs.name))
            print('Skipping')
            continue
        msk[ii] = True
    mska = np.array(msk)

    # Init
    ZH_mnx = (-2., 1)

    # Start the plot
    pp = PdfPages(outfil)

    plt.figure(figsize=(5, 5))
    plt.clf()
    gs = gridspec.GridSpec(2, 2)

    for qq in range(4):
        ax = plt.subplot(gs[qq])
        if qq == 0:  # Mstar
            lbl = r'$\log \, M_*$'
            xval = getattr(cos_halos, 'stellar_mass')
            medx = 0
        elif qq == 1:  # sSFR
            lbl = 'sSFR'
            xval = getattr(cos_halos, 'ssfr')
            medx = 1e-11
        elif qq == 2:  # Rperp
            lbl = r'$R_\perp$'
            xval = getattr(cos_halos, 'rho').value
            medx = 0
        elif qq == 3:  # NHI
            lbl = r'$N_{\rm HI}$'
            #xval = getattr(cos_halos, 'NHI')
            xarr = getattr(cos_halos, 'NHIPDF')  # USE THE PDF
            xval = np.array([ixa.median for ixa in xarr])
            medx = 0

        # Stats (as needed)
        if medx == 0:
            medx = np.median(xval[mska])

        # Build PDFs
        lowPDF, hiPDF = None, None
        for ii, cgm_abs in enumerate(cos_halos.cgm_abs):
            if msk[ii] is False:
                continue
            # Sum the PDF
            if xval[ii] > medx:
                if hiPDF is None:
                    hiPDF = cgm_abs.igm_sys.metallicity
                else:
                    hiPDF = hiPDF + cgm_abs.igm_sys.metallicity
            else:
                if lowPDF is None:
                    lowPDF = cgm_abs.igm_sys.metallicity
                else:
                    lowPDF = lowPDF + cgm_abs.igm_sys.metallicity

        ax.set_xlim(ZH_mnx)
        lowPDF.normalize()
        hiPDF.normalize()

        # Giddy up
        lsz = 19.
        ax.bar(lowPDF.ZH-lowPDF.dZH/2., lowPDF.pdf_ZH, width=lowPDF.dZH, color='red', alpha=0.5)
        ax.bar(hiPDF.ZH-hiPDF.dZH/2., hiPDF.pdf_ZH, width=hiPDF.dZH, color='blue', alpha=0.5)
        ax.set_xlabel("[Z/H]", size=lsz)
        ax.set_ylabel("Normalized PDFs", size=lsz)

        # Label
        csz = 14.
        ax.text(0.05, 0.87, lbl, transform=ax.transAxes, color='black', size=csz, ha='left')
        #lbl2 = '({:d} systems)'.format(nsys)
        #ax.text(0.05, 0.75, lbl2, transform=ax.transAxes, color='black', size=csz, ha='left')
        # Axes
        ax.xaxis.set_major_locator(plt.MultipleLocator(1.))
        xputils.set_fontsize(ax, 13.)

    # End
    print('Writing {:s}'.format(outfil))
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    pp.savefig()
    pp.close()
    plt.close()


def fig_mtlPDF_vs_extrinsic(cos_halos=None, outfil = 'fig_mtlPDF_vs_ex.pdf',
                    qcut=1):
    """ Shows [Z/H] vs x for several x
    Avoids 0 detection systems
    """
    # Read COS-Halos
    if cos_halos is None:
        cos_halos = load_ch()

    # Quality cut
    msk = [False]*len(cos_halos.cgm_abs)
    for ii,cgm_abs in enumerate(cos_halos.cgm_abs):
        # Cut on quality
        if chmtl.mtl_quality(cgm_abs) <= qcut:
            print('No constraint for {:s}'.format(cgm_abs.name))
            print('Skipping')
            continue
        msk[ii] = True
    mska = np.array(msk)

    # Init
    ZH_mnx = (-2., 1)

    # Start the plot
    pp = PdfPages(outfil)

    plt.figure(figsize=(5, 5))
    plt.clf()
    gs = gridspec.GridSpec(2, 2)

    for qq in range(4):
        ax = plt.subplot(gs[qq])
        log = False
        if qq == 0:  # Mstar
            lbl = r'$\log \, M_*$'
            xval = 'stellar_mass'
            xmnx = (9.5, 11.6)
            xstep = 0.5
        elif qq == 1:  # sSFR
            lbl = 'log sSFR'
            xval = 'ssfr'
            xmnx = (-13, -9)
            log = True
            xstep = 1.
        elif qq == 2:  # Rperp
            lbl = r'$R_\perp$'
            xval = 'rho'
            xmnx = (10., 165)
            xstep = 50.
        elif qq == 3:  # NHI
            lbl = r'$N_{\rm HI}$'
            xval = 'NHI'
            xmnx = (14., 20)
            xstep = 2.
        # Call fig_x
        fig_x_vs_ZH(xval, xmnx, lbl, cos_halos=cos_halos, msk=msk, ax=ax,
                    log=log, lsz=13.)
        ax.xaxis.set_major_locator(plt.MultipleLocator(xstep))

    # End
    print('Writing {:s}'.format(outfil))
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    pp.savefig()
    pp.close()
    plt.close()

def fig_mtlPDF_vs_intrinsic(cos_halos=None, outfil = 'fig_mtlPDF_vs_in.pdf', qcut=1):
    """ Shows [Z/H] vs x for several x
    qcut : int, optional

    Avoids 0 detection systems
    """
    # Read COS-Halos
    if cos_halos is None:
        cos_halos = load_ch()

    # Quality cut
    msk = [False]*len(cos_halos.cgm_abs)
    for ii,cgm_abs in enumerate(cos_halos.cgm_abs):
        # Cut on quality
        if chmtl.mtl_quality(cgm_abs) <= qcut:
            print('No constraint for {:s}'.format(cgm_abs.name))
            print('Skipping')
            continue
        msk[ii] = True
    mska = np.array(msk)

    # Init
    ZH_mnx = (-2., 1)

    # Start the plot
    pp = PdfPages(outfil)

    plt.figure(figsize=(8, 4.5))
    plt.clf()
    gs = gridspec.GridSpec(1, 2)

    for qq in range(2):
        ax = plt.subplot(gs[qq])
        log = False
        if qq == 0:  # NHI
            lbl = r'$N_{\rm HI}$'
            xval = 'NHI'
            xmnx = (14., 20)
            xstep = 2.
        elif qq == 1:  # nH
            lbl = r'$n_{\rm H}$'
            xval = 'nH'
            xmnx = (-4.5, -2)
            xstep = 1.
        # Call fig_x
        fig_x_vs_ZH(xval, xmnx, lbl, cos_halos=cos_halos, msk=msk, ax=ax,
                    log=log, lsz=13.)
        ax.xaxis.set_major_locator(plt.MultipleLocator(xstep))

    # End
    print('Writing {:s}'.format(outfil))
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    pp.savefig()
    pp.close()
    plt.close()


def fig_mtl_chains(cos_halos=None, outfil = 'fig_mtl_chains.pdf',
                    qcut=1):
    """ Shows [Z/H] vs x from the chains
    Avoids 0 detection systems
    """
    # Read COS-Halos
    if cos_halos is None:
        cos_halos = load_ch()

    # Load chains
    msk = [False]*len(cos_halos.cgm_abs)
    chains = [[],[],[]]  # ZH, NHI, nH
    for ii,cgm_abs in enumerate(cos_halos.cgm_abs):
        # Cut on quality
        if chmtl.mtl_quality(cgm_abs) <= qcut:
            print('No constraint for {:s}'.format(cgm_abs.name))
            print('Skipping')
            continue
        msk[ii] = True
        # Load emcee
        emcee_fil = '../Analysis/MCMC_FULL/{:s}_emcee.hd5'.format(cgm_abs.name)
        fh5 = h5py.File(emcee_fil,'r')
        tags = list(fh5['outputs']['tags'].value)
        for jj,tag in enumerate(['col', 'dens', 'met']):
            idx = tags.index(tag)
            chains[jj] += list(fh5['outputs']['pdfs'][:,idx])
        fh5.close()
    # Repack
    samples = np.zeros((len(chains[0]), 3))
    for ii in range(3):
        samples[:,ii] = chains[ii]
    #ZH_chains = np.array(chains[0])
    #NHI_chains = np.array(chains[1])
    #nH_chains = np.array(chains[2])
    #mska = np.array(msk)
    #xdb.set_trace()

    # Init
    ZH_mnx = (-2., 2)
    Zbins = np.linspace(ZH_mnx[0],ZH_mnx[1],25)

    # Start the plot
    '''
    pp = PdfPages(outfil)

    fig = plt.figure(figsize=(5, 5))
    plt.clf()
    gs = gridspec.GridSpec(2, 2)
    cm = plt.get_cmap('Blues')
    '''

    lbl_kwargs = dict(size=16)
    contour_kwargs = {'colors':'k'}
    ftags = [r'$\log \; N_{\rm HI}$',
             r'$\log \; n_{\rm H}$', '[M/H]']
    cfig = xcorner.corner(samples, labels=ftags, quantiles=[0.05,0.5,0.95],
                          verbose=False, color='b', contour_kwargs=contour_kwargs,
                          label_kwargs=lbl_kwargs)
    cfig.savefig('fig_mtl_chains.pdf')

    '''
    for qq in range(1):
        ax = plt.subplot(gs[qq])
        log = False
        yval = ZH_chains
        ybins = Zbins
        if qq == 0:  # NHI
            lbl = r'$N_{\rm HI}$'
            xmnx = (14.,20.)
            xbins = np.linspace(xmnx[0],xmnx[1],25)
            xstep = 0.5
            xval = NHI_chains
        elif qq == 1:  # sSFR
            lbl = 'log sSFR'
            xval = 'ssfr'
            xmnx = (-13, -9)
            log = True
            xstep = 1.
        elif qq == 2:  # Rperp
            lbl = r'$R_\perp$'
            xval = 'rho'
            xmnx = (10., 165)
            xstep = 50.
        elif qq == 3:  # NHI
            lbl = r'$N_{\rm HI}$'
            xval = 'NHI'
            xmnx = (14., 20)
            xstep = 2.
        # 2D Histogram
        #xdb.set_trace()
        counts, xedges, yedges = np.histogram2d(xval, yval, bins=(xbins,ybins))
        mplt = ax.pcolormesh(xedges,yedges,counts.transpose(), cmap=cm)
        #cb = fig.colorbar(mplt)
        #ax.scatter(xval, yval, marker='.')

    # End
    print('Writing {:s}'.format(outfil))
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    pp.savefig()
    pp.close()
    plt.close()
    '''

def fig_si3_c3_vs_hi(cos_halos=None, outfil='fig_si3_c3_vs_hi.pdf'):
    """ Plots N(SiIII),N(CIII) vs. NHI
    import patchup_figs as pfigs
    chalos = pfigs.load_ch(skip_comp=False)   # Takes ~1min
    pfigs.fig_si3_c3_vs_hi(cos_halos=chalos)
    """
    # Read COS-Halos
    if cos_halos is None:
        cos_halos = load_ch(skip_comp=False)

    # Init
    NHI_mnx = (12., 18.)
    ions = [(6,3), (14,3)]
    lbls = ['CIII', 'SiIII']
    clrs = ['blue', 'green']
    nion = len(ions)

    SiIII_tbl = cos_halos.ion_tbl((14,3))
    CIII_tbl = cos_halos.ion_tbl((6,3), fill_ion=False)
    ion_tbls = [CIII_tbl, SiIII_tbl]

    NHI = cos_halos.NHI

    # Start the plot
    if outfil is not None:
        pp = PdfPages(outfil)

    plt.figure(figsize=(5, 5))
    plt.clf()
    gs = gridspec.GridSpec(1, 1)

    ax = plt.subplot(gs[0])

    # Axes
    #wvmnx = awvmnx[iuvq['instr']]
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(500.))
    #ax.yaxis.set_major_locator(plt.MultipleLocator(1.))
    ax.minorticks_on()
    ax.set_xlim(NHI_mnx)
    ax.set_ylim(12., 14.5)

    # Labels
    ax.set_xlabel(r'$\log_{10} \, N_{\rm HI}$')
    ax.set_ylabel(r'$\log N_{\rm ion}$')

    # Loop on ions
    for jj,ion in enumerate(ions):
        itbl = ion_tbls[jj]
        # Plot
        # Values
        gdv = np.where(itbl['flag_N'] == 1)[0]
        ax.errorbar(NHI[gdv], itbl['logN'][gdv], xerr=0, color=clrs[jj], linestyle='None',
                    yerr=itbl['sig_logN'][gdv], marker='o', label=lbls[jj])
        # Limits
        limsz = 50.
        ulim = np.where(itbl['flag_N'] == 3)[0]
        ax.scatter(NHI[ulim], itbl['logN'][ulim], color=clrs[jj], marker='v', s=limsz, label=None, facecolor='none')
        llim = np.where(itbl['flag_N'] == 2)[0]
        ax.scatter(NHI[llim], itbl['logN'][llim], color=clrs[jj], marker='^', s=limsz, label=None)

    legend = plt.legend(loc='upper left', scatterpoints=1, borderpad=0.3,
                        handletextpad=0.3, fontsize='small', numpoints=1)

    # End
    print('Writing {:s}'.format(outfil))
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    pp.savefig()
    pp.close()
    plt.close()

def fig_ew_si3_c3_vs_hi(cos_halos=None, outfil='fig_ew_si3_c3_vs_hi.pdf'):
    """ Plots W(SiIII),W(CIII) vs. NHI
    import patchup_figs as pfigs
    chalos = pfigs.load_ch(skip_comp=False)   # Takes ~1min
    pfigs.fig_ew_si3_c3_vs_hi(cos_halos=chalos)
    """
    # Read COS-Halos
    if cos_halos is None:
        cos_halos = load_ch(skip_comp=False)

    # Init
    NHI_mnx = (12., 18.)
    trans = ['CIII 977', 'SiIII 1206']
    clrs = ['blue', 'green']

    CIII_tbl = cos_halos.trans_tbl(trans[0])
    SiIII_tbl = cos_halos.trans_tbl(trans[1])
    trans_tbls = [CIII_tbl, SiIII_tbl]

    NHI = cos_halos.NHI

    # Start the plot
    if outfil is not None:
        pp = PdfPages(outfil)

    plt.figure(figsize=(5, 5))
    plt.clf()
    gs = gridspec.GridSpec(1, 1)

    ax = plt.subplot(gs[0])

    # Axes
    #wvmnx = awvmnx[iuvq['instr']]
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(500.))
    #ax.yaxis.set_major_locator(plt.MultipleLocator(1.))
    ax.minorticks_on()
    ax.set_xlim(NHI_mnx)
    ax.set_ylim(-0.2, 1.5)

    # Labels
    ax.set_xlabel(r'$\log_{10} \, N_{\rm HI}$')
    ax.set_ylabel(r'$W (\AA)$')

    # Loop on ions
    for jj,ion in enumerate(trans):
        itbl = trans_tbls[jj]
        # Plot
        # Values
        gdv = np.where(itbl['flag_EW'] == 1)[0]
        ax.errorbar(NHI[gdv], itbl['EW'][gdv], xerr=0, color=clrs[jj], linestyle='None',
                    yerr=itbl['sig_EW'][gdv], marker='o', label=trans[jj])
        # Limits
        limsz = 50.
        ulim = np.where(itbl['flag_EW'] == 3)[0]
        ax.scatter(NHI[ulim], 2*itbl['sig_EW'][ulim], color=clrs[jj], marker='v', s=limsz, label=None, facecolor='none')
        # Check the low values
        lowv = np.where((2*itbl['sig_EW'][ulim]<0.1) & (np.abs(NHI[ulim] - 14.5) < 0.7))
        print(itbl[ulim][lowv][['cgm_name','EW', 'sig_EW']])
        print(NHI[ulim][lowv])

    legend = plt.legend(loc='upper left', scatterpoints=1, borderpad=0.3,
                        handletextpad=0.3, fontsize='small', numpoints=1)

    # End
    print('Writing {:s}'.format(outfil))
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    pp.savefig()
    pp.close()
    plt.close()


def fig_sngl_cldy_model(sys, dpath=os.getenv('COSHALOS_DATA')+'/Cloudy/', ax=None, outfil=None,
                        lsz=15., show_sys=False):
    """ Compare columns between model and data
    """
    # Read
    fh5 = h5py.File(dpath+sys+'_emcee.hd5', 'r')
    ions = fh5['inputs']['ions'].value
    data = fh5['inputs']['data'].value
    fit = fh5['outputs']['best_fit'].value
    fh5.close()

    # IP values
    IPs = np.zeros(len(ions))
    for ii,ion in enumerate(ions):
        Zion = lai.name_ion(ion)
        elm = ELEMENTS[Zion[0]]
        IPs[ii] = elm.ionenergy[Zion[1]-1]
    #xdb.set_trace()
    resid = data[:,1].astype(float) - fit

    # Start the plot
    if outfil is not None:
        pp = PdfPages(outfil)
        plt.figure(figsize=(7, 7))
        plt.clf()
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0])

    # Axes
    #wvmnx = awvmnx[iuvq['instr']]
    #ax.yaxis.set_major_locator(plt.MultipleLocator(1.))
    ax.minorticks_on()
    xlim = (5., np.max(IPs)+4)
    ax.set_xlim(xlim)
    ylim = (-1., 1.)
    ax.set_ylim(ylim)

    # Labels
    ax.set_xlabel('IP (eV)', size=lsz)
    ax.set_ylabel(r'$\Delta \, \log \, N$', size=lsz)
    ax.xaxis.set_major_locator(plt.MultipleLocator(10.))

    # Values
    gdv = np.where(data[:,3].astype(int) == 0)[0]
    if len(gdv) > 0:
        ax.errorbar(IPs[gdv],resid[gdv], xerr=0, linestyle='None',
                    yerr=data[:,2].astype(float)[gdv], color='blue', marker='o')
    # Limits
    limsz = 50.
    ulim = np.where(data[:,3].astype(int) == -1)[0]
    ax.scatter(IPs[ulim],resid[ulim], color='red', marker='v', s=limsz)
    llim = np.where(data[:,3].astype(int) == -2)[0]
    ax.scatter(IPs[llim],resid[llim], color='green', marker='^', s=limsz)

    # Label
    for kk,ion in enumerate(ions):
        if (resid[kk] < ylim[1]) & (resid[kk] > ylim[0]):
            ax.text(IPs[kk]+1, resid[kk], ion, color='k', size=lsz)  # Only if on page
    if show_sys:
        ax.text(0.04, 0.04, sys, transform=ax.transAxes, size=lsz-1, ha='left')
                #bbox={'facecolor':'white'})

    #legend = plt.legend(loc='upper right', scatterpoints=1, borderpad=0.3,
    #                    handletextpad=0.3, fontsize='small', numpoints=1)
    ax.plot(xlim, [0.,0.], 'g:')

    # End
    xputils.set_fontsize(ax,lsz)
    if outfil is not None:
        print('Writing {:s}'.format(outfil))
        plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
        pp.savefig()
        pp.close()
        plt.close()

def fig_high_ZH(cos_halos=None, outfil='fig_high_ZH.pdf'):
    """ Plots Models for high ZH systems
    """
    # Read COS-Halos
    if cos_halos is None:
        cos_halos = load_ch()

    # Grab high ZH systems
    gdsys = []
    for cgm_abs in cos_halos.cgm_abs:
        low_sig = cgm_abs.igm_sys.metallicity.confidence_limits(0.68)[0]
        if low_sig > -0.3:  # 1/2 solar
            gdsys.append(cgm_abs.name)

    nhigh = len(gdsys)
    if nhigh > 8:
        raise ValueError("Not ready for this many systems")

    # Start the plot
    if outfil is not None:
        pp = PdfPages(outfil)

    plt.figure(figsize=(8, 4))
    plt.clf()
    nrow = 2
    ncol = 4
    gs = gridspec.GridSpec(nrow, ncol)

    for qq,isys in enumerate(gdsys):
        ax = plt.subplot(gs[qq//ncol, qq%ncol])
        fig_sngl_cldy_model(isys, ax=ax, lsz=11, show_sys=True)

    # End
    print('Writing {:s}'.format(outfil))
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    pp.savefig()
    pp.close()
    plt.close()

#### ########################## #########################
#### ########################## #########################
#### ########################## #########################

def main(flg_fig):

    if flg_fig == 'all':
        flg_fig = np.sum( np.array( [2**ii for ii in range(2)] ))
    else:
        flg_fig = int(flg_fig)

    # Lya profiles
    if (flg_fig % 2**1) >= 2**0:
        fig_lya_lines()

    # diff NHI
    if (flg_fig % 2**2) >= 2**1:
        fig_dla_compare()


# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1:
        flg_fig = 0
        #flg_fig += 2**0   # Lya lines
        flg_fig += 2**1   # diff NHI
        #flg_fig += 2**2   # Individual PDFs
        #flg_fig += 2**3   # Full PDFs
        #flg_fig += 2**4   # R vs ZH
        #flg_fig += 2**5   # M* vs ZH
        #flg_fig += 2**6   # NH histogram
        #flg_fig += 2**7   # nH PDF
        #flg_fig += 2**8   # mtlPDF cuts
        #flg_fig += 2**9   # x vs ZH
        #flg_fig += 2**10   # mtl chains
        #flg_fig += 2**11   # SiIII, CIII vs NHI
        #flg_fig += 2**12   # High metallicity systems
        #flg_fig += 2**13   # [Z/H] vs. intrinsic
    else:
        flg_fig = sys.argv[1]

    main(flg_fig)
