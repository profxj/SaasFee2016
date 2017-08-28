#Module for the talk figures of COS-Halos Patchup

# Imports
from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
import glob, os, sys, json, imp
import warnings, copy
import pdb

from scipy.interpolate import interp1d

import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from pkg_resources import resource_filename

from astropy import units as u
from astropy import constants as const
from astropy.table import Table
from astropy.io import fits
from astropy.cosmology import Planck15

from linetools.analysis import absline as ltaa
from linetools.spectralline import AbsLine
from linetools.analysis import voigt as ltav
from linetools.spectra import io as lsio

from specdb.specdb import IgmSpec

from pyigm import utils as pyiu
from pyigm.continuum import quasar as pyicq
from pyigm.fN.fnmodel import FNModel
from pyigm.fN.tau_eff import lyman_ew, lyman_limit
from pyigm.fN import mockforest as pyimock

#from xastropy.sdss import quasars
#from xastropy.xutils import xdebug as xdb
#from xastropy.plotting import utils as xputils


def example_ew(outfil='Figures/example_ew.pdf'):
    """ A simple example of EW
    """

    # Generate a simple Lya line
    lya = AbsLine(1215.6700*u.AA)
    NHI = 13.6
    bval = 30.
    lya.attrib['N'] = 10.**(NHI)/u.cm**2
    lya.attrib['b'] = bval * u.km/u.s
    lya.attrib['z'] = 0.

    # Spectrum
    wave = np.linspace(1180., 1250., 20000) * u.AA
    f_obs = ltav.voigt_from_abslines(wave, [lya])
    f_obs.sig = 0.1*np.ones(f_obs.npix)

    # Measure EW
    lya.analy['spec'] = f_obs
    lya.analy['wvlim'] = [1210.,1220]*u.AA
    lya.measure_ew()

    # Initialize
    xmnx = (1214.2, 1217.2)
    ymnx = (-0.05, 1.08)
    ms = 7.

    # Start the plot
    pp = PdfPages(outfil)
    fig = plt.figure(figsize=(8.5, 3.7))

    plt.clf()
    gs = gridspec.GridSpec(1,2)

    # Lya line
    ax = plt.subplot(gs[0,0])
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
    #ax.xaxis.set_major_locator(plt.MultipleLocator(20.))
    #ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    #ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.set_xlim(xmnx)
    ax.set_ylim(ymnx) 
    ax.set_ylabel('Normalized Flux')
    ax.set_xlabel('Wavelength (Angstroms)')

    lw = 2
    ax.plot(f_obs.wavelength, f_obs.flux, 'k', linewidth=lw)
    ax.fill_between(f_obs.wavelength.value, f_obs.flux.value, np.ones(f_obs.npix),
        color='blue', alpha=0.7)

    # Label
    csz = 12.
    cNHI = '{:0.1f}'.format(NHI)
    ax.text(0.05, 0.2, r'$N_{\rm HI} = 10^{'+cNHI+r'} \rm cm^{-2}$',
        transform=ax.transAxes, size=csz, ha='left')#, bbox={'facecolor':'white'})
    ax.text(0.05, 0.12, r'$b = $'+'{:d} km/s'.format(int(bval)),
        transform=ax.transAxes, size=csz, ha='left')#, bbox={'facecolor':'white'})
    ax.text(0.05, 0.04, r'$W_\lambda = $'+'{:0.2f} Ang'.format(lya.attrib['EW'].value),
        transform=ax.transAxes, size=csz, ha='left')#, bbox={'facecolor':'white'})

    # EW panel
    ax = plt.subplot(gs[0,1])
    ax.set_xlim(xmnx)
    ax.set_ylim(ymnx) 
    ax.set_ylabel('Normalized Flux')
    ax.set_xlabel('Wavelength (Angstroms)')

    xval = [0]+[1215.67-lya.attrib['EW'].value/2]*2 + [1215.67+lya.attrib['EW'].value/2]*2 + [1500]
    ax.plot(xval, [1,1,0,0,1,1], 'k', linewidth=lw)
    ax.fill_between(np.array([lya.attrib['EW'].value/2]*2)*np.array([-1,1])+1215.67,
        [0,0], [1,1], color='red', alpha=0.7)

    # Layout and save
    print('Writing {:s}'.format(outfil))
    plt.tight_layout(pad=0.2,h_pad=0.0,w_pad=0.4)
    plt.subplots_adjust(hspace=0)
    pp.savefig(bbox_inches='tight')
    plt.close()
    # Finish
    pp.close()

def lsf(outfil='Figures/lsf.pdf'):
    """ Compares sinc^2 and Gaussian
    """
    # Start the plot
    xmnx = (-4.,4.)
    pp = PdfPages(outfil)
    fig = plt.figure(figsize=(8.0, 5.0))

    plt.clf()
    gs = gridspec.GridSpec(1,1)

    # Lya line
    ax = plt.subplot(gs[0])
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
    #ax.xaxis.set_major_locator(plt.MultipleLocator(20.))
    #ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    #ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.set_xlim(xmnx)
    #ax.set_ylim(ymnx) 
    ax.set_ylabel('Profile')
    ax.set_xlabel('Pixels (Arbitrary)')

    xval = np.linspace(xmnx[0],xmnx[1],101)
    lw = 2
    ax.plot(xval, np.sinc(xval)**2, 'k', linewidth=lw, 
        label=r'sinc$(x)^2 = [\sin(\pi x)/\pi x$]^2')

    sig = 0.425
    gauss = 1. * np.exp(-1*(xval**2)/(2*sig**2))
    ax.plot(xval, gauss, color='blue', linewidth=lw, label=r'Gaussian (FWHM=1)')


    # Legend
    legend = plt.legend(loc='upper right', scatterpoints=1, borderpad=0.3, 
        handletextpad=0.3, fontsize='large', numpoints=1)

    # Layout and save
    print('Writing {:s}'.format(outfil))
    plt.tight_layout(pad=0.2,h_pad=0.0,w_pad=0.4)
    plt.subplots_adjust(hspace=0)
    pp.savefig(bbox_inches='tight')
    plt.close()
    # Finish
    pp.close()

def fj0812(outfil='Figures/fj0812.pdf'):
    """ Plots of FJ0812 in several spectrometers
    """
    # Load spectra
    sdss_dr7 = quasars.SdssQuasars()
    qso = sdss_dr7[(861,333)]
    qso.load_spec()
    qso.spec.normed = True

    esi = lsio.readspec('/Users/xavier/Keck/ESI/RedData/FJ0812+32/FJ0812+32_f.fits')
    hires = lsio.readspec('/Users/xavier/Keck/HIRES/RedData/FJ0812+32/FJ0812+32B_f.fits')

    # Initialize
    xmnx = (4100., 4450)
    ymnx = (-0.05, 1.28)
    lw = 1.0
    # Start the plot
    pp = PdfPages(outfil)
    fig = plt.figure(figsize=(8.5, 5.0))

    plt.clf()
    gs = gridspec.GridSpec(2,2)
    lbls = ['SDSS: R=2000', 'ESI: R=8000', 'HIRES: R=30000']
    clrs = ['blue', 'red', 'green']

    # Final plot
    ax2 = plt.subplot(gs[1,1])
    ax2.set_xlim(4270, 4295)
    ax2.set_ylim(ymnx) 
    ax2.set_xlabel('Wavelength (Angstroms)')

    for qq in range(3):
        scl = 1.
        if qq == 0:
            spec = qso.spec
            scl = 0.8
        elif qq == 1:
            spec = esi
        elif qq == 2:
            spec = hires

        # SDSS
        ax = plt.subplot(gs[qq % 2,qq // 2])
        #ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
        #ax.xaxis.set_major_locator(plt.MultipleLocator(20.))
        #ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
        #ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax.set_xlim(xmnx)
        ax.set_ylim(ymnx) 
        ax.set_ylabel('Normalized Flux')
       # if qq == 0:
       #     ax.get_xaxis().set_ticks([])
       # else:
        ax.set_xlabel('Wavelength (Angstroms)')

        ax.plot(spec.wavelength, spec.flux/scl, 'k', linewidth=lw)
        ax2.plot(spec.wavelength, spec.flux/scl, color=clrs[qq], linewidth=lw,
            drawstyle='steps-mid')

        # Label
        csz = 12.
        ax.text(0.95, 0.9, lbls[qq], transform=ax.transAxes, color=clrs[qq], 
            size=csz, ha='right', bbox={'facecolor':'white'})

    # Layout and save
    print('Writing {:s}'.format(outfil))
    plt.tight_layout(pad=0.2,h_pad=0.3,w_pad=0.4)
    #plt.subplots_adjust(hspace=0)
    pp.savefig(bbox_inches='tight')
    plt.close()
    # Finish
    pp.close()

def qso_sed(outfil='Figures/qso_sed.pdf'):
    """ Plots a few QSO examples
    """
    igmsp = IgmSpec()
    #sdss_dr7 = quasars.SdssQuasars()
    #bal = sdss_dr7[(367, 506)]
    #bal = sdss_dr7[(726, 60)]
    weakbal, wb_meta = igmsp.get_sdss(668, 193, groups=['SDSS_DR7'])
    normal, n_meta = igmsp.get_sdss(393,250, groups=['SDSS_DR7'])
    sbal, sb_meta = igmsp.get_sdss(690,131, groups=['SDSS_DR7'])
    # Start the plot
    xmnx = (1180., 1580)
    pp = PdfPages(outfil)
    fig = plt.figure(figsize=(8.0, 5.0))

    plt.clf()
    gs = gridspec.GridSpec(3,1)

    # Lya line
    for qq in range(3):
        ylim = None
        if qq == 0:
            spec = normal
            meta = n_meta
        elif qq == 1:
            spec = weakbal
            meta = wb_meta
            ylim = (-10,99)
        elif qq == 2:
            spec = sbal
            meta = sb_meta
            ylim = (-1,7)
        ax = plt.subplot(gs[qq])
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
    #ax.xaxis.set_major_locator(plt.MultipleLocator(20.))
    #ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    #ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax.set_xlim(xmnx)
        if ylim is not None:
            ax.set_ylim(ylim)
    #ax.set_ylim(ymnx)
        ax.set_ylabel('Relative Flux')
        if qq < 1:
            ax.get_xaxis().set_ticks([])
        else:
            ax.set_xlabel('Rest Wavelength (Angstroms)')

        lw = 1.
        ax.plot(spec.wavelength/(meta['zem_GROUP']+1), spec.flux, 'k', linewidth=lw)
        set_fontsize(ax, 17.)

    # Layout and save
    print('Writing {:s}'.format(outfil))
    plt.tight_layout(pad=0.2,h_pad=0.0,w_pad=0.4)
    plt.subplots_adjust(hspace=0)
    pp.savefig(bbox_inches='tight')
    plt.close()
    # Finish
    pp.close()


def qso_template(outfil='Figures/qso_template.pdf'):
    """ van den berk
    """
    # Load
    telfer = pyicq.get_telfer_spec()

    clight = const.c.cgs

    # Beta spliced to vanden Berk template with host galaxy  removed
    van_file = resource_filename('pyigm', '/data/quasar/VanDmeetBeta_fullResolution.txt')
    van_tbl = Table.read(van_file,format='ascii')
    isort = np.argsort(van_tbl['nu'])
    nu_van = van_tbl['nu'][isort]
    fnu_van = van_tbl['f_nu'][isort]
    lam_van = (clight/(nu_van*u.Hz)).to('AA')
    flam_van = fnu_van * clight / lam_van**2
    nrm_pix = np.abs(lam_van-1450*u.AA) < 10*u.AA
    nrm_van = np.median(flam_van[nrm_pix])
    flam_van = flam_van / nrm_van

    # Start the plot
    xmnx = (1050., 2300)
    pp = PdfPages(outfil)
    fig = plt.figure(figsize=(8.0, 5.0))

    plt.clf()
    gs = gridspec.GridSpec(1,1)

    # Lya line
    ax = plt.subplot(gs[0])
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
    #ax.xaxis.set_major_locator(plt.MultipleLocator(20.))
    #ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    #ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.set_xlim(xmnx)
    #ax.set_ylim(ymnx) 
    ax.set_ylabel('Relative Flux')
    ax.set_xlabel('Rest Wavelength (Angstroms)')

    lw = 1.
    ax.plot(telfer.wavelength, telfer.flux, 'k', linewidth=lw, 
        label='Telfer (z~1)') 
    ax.plot(lam_van, flam_van, 'b', linewidth=lw, label='SDSS (z~2)')

    # Legend
    legend = plt.legend(loc='upper right', scatterpoints=1, borderpad=0.3, 
        handletextpad=0.3, fontsize='large', numpoints=1)
    # Layout and save
    set_fontsize(ax, 17.)
    print('Writing {:s}'.format(outfil))
    plt.tight_layout(pad=0.2,h_pad=0.0,w_pad=0.4)
    plt.subplots_adjust(hspace=0)
    pp.savefig(bbox_inches='tight')
    plt.close()
    # Finish
    pp.close()

def qso_fuv(outfil='Figures/qso_fuv.pdf'):
    """ FUV flux from QSOs
    """
    pyigm_path = imp.find_module('pyigm')[1]

    # Load
    telfer = pyicq.get_telfer_spec()


    # Start the plot
    xmnx = (900., 1220)
    pp = PdfPages(outfil)
    fig = plt.figure(figsize=(8.0, 5.0))

    plt.clf()
    gs = gridspec.GridSpec(1,1)

    # Lya line
    ax = plt.subplot(gs[0])
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
    #ax.xaxis.set_major_locator(plt.MultipleLocator(20.))
    #ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    #ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.set_xlim(xmnx)
    #ax.set_ylim(ymnx) 
    ax.set_ylabel('Relative Flux')
    ax.set_xlabel('Rest Wavelength (Angstroms)')

    lw = 1.
    ax.plot(telfer.wavelength, telfer.flux, 'k', linewidth=lw, 
        label='Telfer Average QSO SED (z~1)') 

    # Legend
    legend = plt.legend(loc='upper left', scatterpoints=1, borderpad=0.3, 
        handletextpad=0.3, fontsize='x-large', numpoints=1)
    # Layout and save
    xputils.set_fontsize(ax, 17.)
    print('Writing {:s}'.format(outfil))
    plt.tight_layout(pad=0.2,h_pad=0.0,w_pad=0.4)
    plt.subplots_adjust(hspace=0)
    pp.savefig(bbox_inches='tight')
    plt.close()
    # Finish
    pp.close()

def redshift(outfil='Figures/redshift.pdf'):
    """ Series of plots illustrating redshift in the Lya forest
    """
    lrest = np.array([900., 1250])  # Ang
    zem = 3.
    toff = 0.15
    yqso = 0.7
    sqso = 35

    # QSO lines
    qsolya = AbsLine(1215.6700*u.AA)
    qsolya.attrib['N'] = 10.**(16.0)/u.cm**2
    qsolya.attrib['b'] = 40 * u.km/u.s
    qsolya.attrib['z'] = zem
    qsolyb = AbsLine('HI 1025')
    qsolyb.attrib['N'] = qsolya.attrib['N'] 
    qsolyb.attrib['b'] = qsolya.attrib['b'] 
    qsolyb.attrib['z'] = qsolya.attrib['z'] 

    def tick_function(z, X):
        V = X*(1+z)
        return ["{:d}".format(int(round(x))) for x in V]

    def add_lines(axi,z):
        wvtwo = (1+z)*1215.67/(1+zem)
        axi.scatter([wvtwo], [yqso], marker='o', facecolor='none', 
            edgecolor='green', s=sqso*5)
        axi.text(wvtwo, yqso-1.7*toff, 'HI Gas (z={:0.1f})'.format(z), 
            color='green', ha='center')
        #
        twolya = copy.deepcopy(qsolya)
        twolya.attrib['z'] = z
        twolyb = copy.deepcopy(qsolyb)
        twolyb.attrib['z'] = z
        return [twolya, twolyb]

    # Telfer
    telfer = pyicq.get_telfer_spec()

    # Start the plot
    pp = PdfPages(outfil)
    scl = 1.0
    fig = plt.figure(figsize=(8.0*scl, 5.0*scl))

    plt.clf()
    gs = gridspec.GridSpec(5,1)
    jet = cm = plt.get_cmap('jet') 

    # Loop
    for qq in range(9):
        # Cartoon
        ax0 = plt.subplot(gs[0,0])
        ax0.set_xlim(lrest)
        ax0.set_ylim(0.,1.)
        ax0.set_frame_on(False)
        ax0.axes.get_yaxis().set_visible(False)
        ax0.axes.get_xaxis().set_visible(False)


        # QSO
        ax0.scatter([1215.67], [yqso], marker='o', facecolor='blue', s=sqso)
        ax0.text(1215.67, yqso+toff, 'Quasar (z=3)', color='blue', ha='center')

        # Redshifted light
        if qq > 0:
            light = np.linspace(1215.67, lrest[0],20)
            ax0.scatter(light, [yqso]*len(light), marker='_', s=40, cmap=cm, c=1./light)

        # Gas at QSO
        if qq > 1:
            ax0.scatter([1215.67], [yqso], marker='o', facecolor='none', 
                edgecolor='green', s=sqso*5)
            ax0.text(1215.67, yqso-1.7*toff, 'HI Gas (z=3)', color='green', ha='center')

        # Spectrum
        ax = plt.subplot(gs[1:,0])
        ax.set_xlim(lrest)
        #ax.set_ylim(ymnx) 
        ax.set_ylabel('Relative Flux')
        ax.set_xlabel("Rest Wavelength")
        if qq < 3:
            tsty = 'k'
        else: 
            tsty = 'b:'
        ax.plot(telfer.wavelength, telfer.flux, tsty)

        # Observer frame axis
        if qq > 0:
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            xtcks = ax.get_xticks()
            ax2.set_xticks(xtcks)
            ax2.set_xticklabels(tick_function(zem, xtcks))
            ax2.set_xlabel('Observed Wavelength (Angstroms)')

        # Absorption lines
        abslines = []
        if (qq > 2) and (qq != 8):  
            # Lya at zem

            abslines.append(qsolya)
            if qq > 3: # Lyb at zem
                abslines.append(qsolyb)
            # Gas at z=2.8
            if qq > 4:
                zadd = 2.8
                abslines += add_lines(ax0, zadd)
            # Gas at z=2.5
            if qq > 5:
                zadd = 2.5
                abslines += add_lines(ax0, zadd)
            if qq > 6:
                zadd = 2.2
                abslines += add_lines(ax0, zadd)
            #
            abs_model = ltav.voigt_from_abslines(telfer.wavelength*(1+zem), abslines)
            #ax.plot(telfer.wavelength, telfer.flux*abs_model.flux, 'k')
            ax.plot(telfer.wavelength, telfer.flux.value*abs_model.flux, 'k')

        # Final plot
        if qq == 8:
            nlin = 100
            dotwv = np.linspace(900,1215.,nlin)
            ax0.scatter(dotwv, [yqso]*nlin, marker='o', facecolor='none', 
                edgecolor='green', s=sqso*5)
            # Mock spectrum
            fN_model = FNModel.default_model()
            gdp = np.where(telfer.wavelength > 900.*u.AA)[0]
            mock_spec, HI_comps, misc = pyimock.mk_mock(
                telfer.wavelength[gdp]*(1+zem), 
                zem, fN_model, s2n=100., fwhm=3, add_conti=False)
            ax.plot(telfer.wavelength[gdp], 
                telfer.flux[gdp].value*mock_spec.flux, 'k')


        # Layout and save
        plt.tight_layout(pad=0.2,h_pad=0.0,w_pad=0.4)
        plt.subplots_adjust(hspace=0)
        pp.savefig(bbox_inches='tight', transparent=True)
        plt.close()
    # Finish
    print('Writing {:s}'.format(outfil))
    pp.close()
    return mock_spec


def q1422(outfil='Figures/q1422.pdf'):
    """ Series of plots on Q1422
    """
    q1422 = lsio.readspec('/Users/xavier/Keck/HIRES/RedData/Q1422+2309/Q1422+2309.fits')

    axmnx = [ [4000, 6000],
              [4900, 5500],
              [5100, 5300],
              [5150, 5250],
              [5190, 5220],
    ]
    aymnx = [ [0., 7300],
              [0., 2000],
              [0., 1700],
              [0., 1100],
              [0., 1100],
    ]
    lw = 1.
    csz = 19.

    # Start the plot
    pp = PdfPages(outfil)
    fig = plt.figure(figsize=(8.0, 5.0))

    plt.clf()
    gs = gridspec.GridSpec(1,1)

    # Loop
    for qq, xmnx in enumerate(axmnx):

        # Spectrum
        ax = plt.subplot(gs[0])
        ax.set_xlim(xmnx)
        ax.set_ylim(aymnx[qq])
        #ax.set_ylim(ymnx) 
        ax.set_ylabel('Relative Flux')
        ax.set_xlabel("Wavelength (Ang)")

        ax.plot(q1422.wavelength, q1422.flux, 'k', linewidth=lw)

        # 
        ax.text(0.05, 0.9, 'Keck/HIRES: Q1422+2309', color='blue',
            transform=ax.transAxes, size=csz, ha='left')#, bbox={'facecolor':'white'})
        #
        xputils.set_fontsize(ax, 17.)
        # Layout and save
        plt.tight_layout(pad=0.2,h_pad=0.0,w_pad=0.4)
        plt.subplots_adjust(hspace=0)
        pp.savefig(bbox_inches='tight', transparent=True)
        plt.close()
    # Finish
    print('Writing {:s}'.format(outfil))
    pp.close()


def evolving_forest(outfil='Figures/evolving_forest.pdf'):
    """ Show varying IGM transmission
    """
    #hdlls_path = '/u/xavier/paper/LLS/Optical/Data/DR1/Spectra/'
    esi_path = '/u/xavier/Keck/ESI/RedData/'
    hst_path = '/u/xavier/HST/Cycle23/z1IGM/Archive/PG1206+459/'
    #
    igmsp = IgmSpec()
    idicts = [dict(coord='J212329.50-005052.9', group=['HD-LLS_DR1']),
              dict(coord='J020950.7-000506.4', group=['HD-LLS_DR1'], INSTR='HIRES'),
              dict(coord='J113621.00+005021.0', group=['HD-LLS_DR1']),
              dict(coord='J094932.27+033531.7', group=['HD-LLS_DR1']),
              dict(coord='J013421.63+330756.5', group=['HD-LLS_DR1']),
              dict(coord='J083122.57+404623.4', group=['ESI_DLA']),
              dict(coord='J113246.5+120901.6', group=['ESI_DLA']),
              dict(filename=esi_path+'J1148+5251/SDSSJ1148+5251_stack.fits'),  # z=6
              dict(coord='J212329.50-005052.9', group=['HD-LLS_DR1']),
              dict(filename=hst_path+'PG1206+459_E230M_f.fits'),
              ]
    '''
    dat_files = [
                hdlls_path+'HD-LLS_J212329.50-005052.9_HIRES.fits',
                hdlls_path+'HD-LLS_J020951.10-000513.0_HIRES.fits',
                hdlls_path+'HD-LLS_J113621.00+005021.0_MIKE.fits',
                hdlls_path+'HD-LLS_J094932.27+033531.7_MIKE.fits',
                esi_path+'PSS0134+3307/PSS0134+3307_f.fits',
                esi_path+'J0831+4046/J0831+4046a_f.fits',
                esi_path+'J1132+1209/J1132+1209a_f.fits',
                esi_path+'J1148+5251/SDSSJ1148+5251_stack.fits',
                hdlls_path+'HD-LLS_J212329.50-005052.9_HIRES.fits',
                hst_path+'PG1206+459_E230M_f.fits',
                '/Users/xavier/MLSS/data/3C273/STIS/E140M/3C273_STIS_E140M_F.fits',
                ]
    conti_fil = '/Users/xavier/MLSS/data/3C273/STIS/E140M/3C273_STIS_E140M_c.fits'
    conti_3c273 = fits.open(conti_fil)[0].data
    '''
    lbls = [
            'Keck/HIRES: J2123-0050',
            'Keck/HIRES: J0209-0005',
            'Magellan/MIKE: J1136+0050', # 3.43
            'Magellan/MIKE: J0949+0335',
            'Keck/ESI: PSS0134+3307',
            'Keck/ESI: J0831+4046',
            'Keck/ESI: J1132+1209', # 5.17
            'Keck/ESI: J1148+5251',
            'Keck/HIRES: J2123-0050',
            'HST/STIS: PG1206+459',
            'HST/STIS: 3C273',
            ]
    zems = [2.26, 2.86, 3.43, 4.05, 4.52, 4.89, 5.17, 6.4, 2.26, 1.16, 0.17]
    xrest = np.array([1080, 1200.])
    ymnx = [-0.1, 1.1]

    lw = 1.
    csz = 19.

    # Start the plot
    pp = PdfPages(outfil)
    fig = plt.figure(figsize=(8.0, 5.0))

    plt.clf()
    gs = gridspec.GridSpec(1,1)

    # Loop
    for qq, lbl in enumerate(lbls):
        if qq > 9:
            break

        # Grab data
        idict = idicts[qq]
        if 'coord' in idict.keys():
            qdict = {}
            for key in idict.keys():
                if key not in ['coord','group']:
                    qdict[key] = idict[key]
            spec, meta = igmsp.spectra_from_coord(idict['coord'], tol=5.*u.arcsec, groups=idict['group'], query_dict=qdict)
            if meta is None:
                print("Bad coord?")
                pdb.set_trace()
            elif len(meta) > 1:
                pdb.set_trace()
        else:
            spec = lsio.readspec(idict['filename'])
        if lbl == 'HST/STIS: 3C273':
            spec.co = conti_3c273
            spec.normed = True

        # Spectrum
        ax = plt.subplot(gs[0])
        ax.set_xlim(xrest*(1+zems[qq])/1215.67 - 1)
        ax.set_ylim(ymnx) 
        ax.set_ylabel('Normalized Flux')
        ax.set_xlabel(r'Redshift of Ly$\alpha$')


        ax.plot(spec.wavelength.value/1215.6701 - 1, spec.flux, 'k', linewidth=lw)

        # 
        ax.text(0.05, 0.95, lbl+' zem={:0.1f}'.format(zems[qq]), color='blue',
            transform=ax.transAxes, size=csz, ha='left', bbox={'facecolor':'white'})
        #
        set_fontsize(ax, 17.)
        # Layout and save
        plt.tight_layout(pad=0.2,h_pad=0.0,w_pad=0.4)
        plt.subplots_adjust(hspace=0)
        pp.savefig(bbox_inches='tight', transparent=True)
        plt.close()
    # Finish
    print('Writing {:s}'.format(outfil))
    pp.close()


def evolving_forest_in_chapter(outfil='Figures/evolving_forest_in_chapter.pdf'):
    """ Show varying IGM transmission
    """
    #hdlls_path = '/u/xavier/paper/LLS/Optical/Data/DR1/Spectra/'
    esi_path = '/u/xavier/Keck/ESI/RedData/'
    hst_path = '/u/xavier/HST/Cycle23/z1IGM/Archive/PG1206+459/'
    #
    igmsp = IgmSpec()
    idicts = [
        dict(filename='Data/3C273_STIS_E140M_F.fits'),
        dict(filename=hst_path+'PG1206+459_E230M_f.fits'),
        dict(coord='J212329.50-005052.9', group=['HD-LLS_DR1']),
        dict(coord='J113621.00+005021.0', group=['HD-LLS_DR1']),
        dict(coord='J113246.5+120901.6', group=['ESI_DLA']),
        dict(filename=esi_path+'J1148+5251/SDSSJ1148+5251_stack.fits'),  # z=6
        ]
    lbls = [
        'HST/STIS: 3C273',
        'HST/STIS: PG1206+459',
        'Keck/HIRES: J2123-0050',  # 2.26
        'Magellan/MIKE: J1136+0050', # 3.43
        'Keck/ESI: J1132+1209', # 5.17
        'Keck/ESI: J1148+5251', # 6.4
        ]
    zems = [0.17, 1.16, 2.26, 3.43, 5.17, 6.4]
    xrest = np.array([1080, 1200.])
    ymnx = [-0.1, 1.1]

    lw = 1.
    csz = 19.

    # Start the plot
    fig = plt.figure(figsize=(5.0, 8.0))

    plt.clf()
    gs = gridspec.GridSpec(6,1)

    # Loop
    for qq, lbl in enumerate(lbls):

        # Grab data
        idict = idicts[qq]
        if 'coord' in idict.keys():
            qdict = {}
            for key in idict.keys():
                if key not in ['coord','group']:
                    qdict[key] = idict[key]
            spec, meta = igmsp.spectra_from_coord(idict['coord'], tol=5.*u.arcsec, groups=idict['group'], query_dict=qdict)
            if meta is None:
                print("Bad coord?")
                pdb.set_trace()
            elif len(meta) > 1:
                pdb.set_trace()
        else:
            spec = lsio.readspec(idict['filename'])

        if lbl == 'HST/STIS: 3C273':
            #spec.co = conti_3c273
            spec.normed = True

        # Spectrum
        ax = plt.subplot(gs[qq])
        ax.set_xlim(xrest*(1+zems[qq])/1215.67 - 1)
        ax.set_ylim(ymnx)
        if qq == 3:
            ax.set_ylabel('Normalized Flux')
        if qq == len(lbls)-1:
            ax.set_xlabel(r'Redshift of Ly$\alpha$')


        ax.plot(spec.wavelength.value/1215.6701 - 1, spec.flux, 'k', linewidth=lw)

        # Label
        #ax.text(0.05, 0.95, lbl+' zem={:0.1f}'.format(zems[qq]), color='blue',
        #    transform=ax.transAxes, size=csz, ha='left', bbox={'facecolor':'white'})
        #
        set_fontsize(ax, 12.)

    # Layout and save
    #plt.subplots_adjust(hspace=0)
    plt.tight_layout(pad=0.2,h_pad=0.0,w_pad=0.4)
    plt.savefig(outfil)
    plt.close()
    # Finish
    print('Writing {:s}'.format(outfil))


def dteff(outfil='Figures/dteff.pdf'):
    """ Differential teff (Lya)
    """
    # Load fN
    fN_model = FNModel.default_model()

    # teff
    cumul = []
    iwave = 1215.67 * (1+2.5)
    zem = 2.6
    teff_alpha = lyman_ew(iwave, zem, fN_model, cumul=cumul)
    print('teff = {:g}'.format(teff_alpha))

    dteff = cumul[1] - np.roll(cumul[1],1)
    dteff[0] = dteff[1] # Fix first value

    # Start the plot
    xmnx = (12, 22)
    pp = PdfPages(outfil)
    fig = plt.figure(figsize=(8.0, 5.0))

    plt.clf()
    gs = gridspec.GridSpec(1,1)

    # Lya line
    ax = plt.subplot(gs[0])
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
    #ax.xaxis.set_major_locator(plt.MultipleLocator(20.))
    #ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    #ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.set_xlim(xmnx)
    #ax.set_ylim(ymnx) 
    ax.set_ylabel(r'$d\tau_{\rm eff, \alpha}/d\log N$')
    ax.set_xlabel(r'$\log \, N_{\rm HI}$')

    lw = 2.
    ax.plot(cumul[0], dteff, 'k', linewidth=lw)

    # Label
    csz = 17
    ax.text(0.60, 0.90, 'f(N) from Prochaska+14', color='blue',
            transform=ax.transAxes, size=csz, ha='left') 
    ax.text(0.60, 0.80, 'z=2.5', color='blue',
            transform=ax.transAxes, size=csz, ha='left') 
    xputils.set_fontsize(ax, 17.)
    # Layout and save
    print('Writing {:s}'.format(outfil))
    plt.tight_layout(pad=0.2,h_pad=0.0,w_pad=0.4)
    plt.subplots_adjust(hspace=0)
    pp.savefig(bbox_inches='tight')
    plt.close()
    # Finish
    pp.close()

def sawtooth(outfil='Figures/sawtooth.pdf', all_tau=None):
    """ Sawtooth opacity
    """
    # Load fN
    fN_model = FNModel.default_model()
    fN_model.zmnx = (2.,4.1) # extrapolate a bit
    # teff
    zem = 4
    wave = np.arange(4500., 6200., 10)
    # Calculate
    if all_tau is None:
        all_tau = np.zeros_like(wave)
        for qq,iwave in enumerate(wave):
            all_tau[qq] = lyman_ew(iwave, zem, fN_model)
    # Flux attenuation
    flux = np.exp(-all_tau)

    # Start the plot
    xmnx = (4500, 6200)
    pp = PdfPages(outfil)
    fig = plt.figure(figsize=(8.0, 5.0))

    plt.clf()
    gs = gridspec.GridSpec(1,1)

    # Lya line
    ax = plt.subplot(gs[0])
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
    #ax.xaxis.set_major_locator(plt.MultipleLocator(20.))
    #ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    #ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.set_xlim(xmnx)
    ax.set_ylim(0., 1.1)
    ax.set_ylabel('IGM Transmission')
    ax.set_xlabel('Observed wavelength (z=4 source)')

    lw = 2.
    ax.plot(wave, flux, 'b', linewidth=lw)

    # Label
    csz = 17
    ax.text(0.10, 0.90, 'f(N) from Prochaska+14', color='blue',
            transform=ax.transAxes, size=csz, ha='left') 
    ax.text(0.10, 0.80, 'Ignores Lyman continuum opacity', color='blue',
            transform=ax.transAxes, size=csz, ha='left') 
    xputils.set_fontsize(ax, 17.)
    # Layout and save
    print('Writing {:s}'.format(outfil))
    plt.tight_layout(pad=0.2,h_pad=0.0,w_pad=0.4)
    plt.subplots_adjust(hspace=0)
    pp.savefig(bbox_inches='tight')
    plt.close()
    # Finish
    pp.close()
    return all_tau

def obs_sawtooth(outfil='Figures/obs_sawtooth.pdf', all_tau=None, scl=1.):
    """ Sawtooth opacity
    """
    # SDSS
    hdu = fits.open('/Users/xavier/paper/LLS/taueff/Analysis/stack_DR7_z3.92_z4.02.fits')
    sdss_fx = hdu[0].data
    sdss_wave = hdu[2].data
    # Telfer
    telfer = pyicq.get_telfer_spec()
    i1450 = np.argmin(np.abs(telfer.wavelength.value - 1450.))
    nrm = np.median(telfer.flux[i1450-5:i1450+5])
    telfer.flux = telfer.flux / nrm
    trebin = telfer.rebin(sdss_wave*u.AA)
    # Load fN
    fN_model = FNModel.default_model()
    fN_model.zmnx = (2.,4.1) # extrapolate a bit
    # teff
    zem = 4.
    wave = np.arange(4500., 6200., 10)
    # Calculate
    if all_tau is None:
        all_tau = np.zeros_like(wave)
        for qq,iwave in enumerate(wave):
            all_tau[qq] = lyman_ew(iwave, zem, fN_model)
    # Flux attenuation
    trans = np.exp(-all_tau)

    ftrans = interp1d(wave, trans, fill_value=1.,
        bounds_error=False)

    # Start the plot
    xmnx = (4500, 6200)
    pp = PdfPages(outfil)
    fig = plt.figure(figsize=(8.0, 5.0))

    plt.clf()
    gs = gridspec.GridSpec(1,1)

    # Lya line
    ax = plt.subplot(gs[0])
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
    #ax.xaxis.set_major_locator(plt.MultipleLocator(20.))
    #ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    #ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.set_xlim(xmnx)
    ax.set_ylim(0., 1.5)
    ax.set_ylabel('Relative Flux')
    ax.set_xlabel('Observed wavelength')

    lw = 2.
    # Data
    ax.plot(sdss_wave*(1+zem), sdss_fx, 'k', linewidth=lw, label='SDSS QSOs (z=4)')
    # Model
    model = trebin.flux * ftrans(sdss_wave*(1+zem)) * scl
    ax.plot(sdss_wave*(1+zem), model, 'r', linewidth=lw, label='IGM+Telfer model')

    # Label
    csz = 17
    #ax.text(0.10, 0.10, 'SDSS quasar stack at z=4', color='black',
    #        transform=ax.transAxes, size=csz, ha='left') 
    # Legend
    legend = plt.legend(loc='upper left', scatterpoints=1, borderpad=0.3, 
        handletextpad=0.3, fontsize='large', numpoints=1)
    xputils.set_fontsize(ax, 17.)
    # Layout and save
    print('Writing {:s}'.format(outfil))
    plt.tight_layout(pad=0.2,h_pad=0.0,w_pad=0.4)
    plt.subplots_adjust(hspace=0)
    pp.savefig(bbox_inches='tight')
    plt.close()
    # Finish
    pp.close()
    return all_tau

def wfc3_qso(outfil='Figures/wfc3_qso.pdf'):
    """ Show a QSO that is transmissive at the Lyman limit
    """
    # WFC3
    wfc3, _ = pyicq.wfc3_continuum(0)
    #zem = 4.


    # Start the plot
    xmnx = (650, 1300)
    pp = PdfPages(outfil)
    fig = plt.figure(figsize=(8.0, 5.0))

    plt.clf()
    gs = gridspec.GridSpec(1,1)

    # Lya line
    ax = plt.subplot(gs[0])
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
    #ax.xaxis.set_major_locator(plt.MultipleLocator(20.))
    #ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    #ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.set_xlim(xmnx)
    ax.set_ylim(0., 70)
    ax.set_ylabel('Relative Flux')
    ax.set_xlabel('Rest wavelength')

    lw = 2.
    # Data
    ax.plot(wfc3.wavelength, wfc3.flux, 'k', linewidth=lw, label='SDSS QSOs (z=4)')

    # Label
    csz = 17
    ax.text(0.10, 0.80, 'HST/WFC3: QSO spectrum (z~2)', color='black',
            transform=ax.transAxes, size=csz, ha='left') 
    xputils.set_fontsize(ax, 17.)
    # Layout and save
    print('Writing {:s}'.format(outfil))
    plt.tight_layout(pad=0.2,h_pad=0.0,w_pad=0.4)
    plt.subplots_adjust(hspace=0)
    pp.savefig(bbox_inches='tight')
    plt.close()
    # Finish
    pp.close()

def dXdz(outfil='Figures/dXdz.pdf'):
    """ Plot dXdz vs. z
    """
    # z
    zval = np.linspace(1., 5, 100)

    # dX/dz
    dXdz = pyiu.cosm_xz(zval, cosmo=Planck15, flg_return=1)

    # Start the plot
    xmnx = (1., 5)
    pp = PdfPages(outfil)
    fig = plt.figure(figsize=(8.0, 5.0))

    plt.clf()
    gs = gridspec.GridSpec(1,1)

    # Lya line
    ax = plt.subplot(gs[0])
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
    #ax.xaxis.set_major_locator(plt.MultipleLocator(20.))
    #ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    #ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.set_xlim(xmnx)
    ax.set_ylim(0., 5)
    ax.set_ylabel('dX/dz')
    ax.set_xlabel('z')

    lw = 2.
    # Data
    ax.plot(zval, dXdz, 'k', linewidth=lw)#, label='SDSS QSOs (z=4)')

    # Label
    csz = 17
    #ax.text(0.10, 0.80, 'HST/WFC3: QSO spectrum (z~2)', color='black',
    #        transform=ax.transAxes, size=csz, ha='left') 
    xputils.set_fontsize(ax, 17.)
    # Layout and save
    print('Writing {:s}'.format(outfil))
    plt.tight_layout(pad=0.2,h_pad=0.0,w_pad=0.4)
    plt.subplots_adjust(hspace=0)
    pp.savefig(bbox_inches='tight')
    plt.close()
    # Finish
    pp.close()

def teff_LL(outfil='Figures/teff_LL.pdf'):
    """ Plot teff_LL from z=3.5 down
    """
    # z
    zem = 3.5
    z912 = 3.

    # f(N)
    fnmodel = FNModel.default_model()
    fnmodel.zmnx = (0.5,4) # extend default range

    # Calculate
    zval, teff_LL = lyman_limit(fnmodel, z912, zem)

    # Start the plot
    xmnx = (3.5, 3)
    pp = PdfPages(outfil)
    fig = plt.figure(figsize=(8.0, 5.0))

    plt.clf()
    gs = gridspec.GridSpec(1,1)

    # Lya line
    ax = plt.subplot(gs[0])
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
    #ax.xaxis.set_major_locator(plt.MultipleLocator(20.))
    #ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    #ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.set_xlim(xmnx)
    ax.set_ylim(0., 2)
    ax.set_ylabel(r'$\tau_{\rm eff}^{\rm LL}$')
    ax.set_xlabel('z')

    lw = 2.
    # Data
    ax.plot(zval, teff_LL, 'b', linewidth=lw)#, label='SDSS QSOs (z=4)')

    # Label
    csz = 17
    ax.text(0.10, 0.80, 'Source at z=3.5',
        color='black', transform=ax.transAxes, size=csz, ha='left')
    xputils.set_fontsize(ax, 17.)
    # Layout and save
    print('Writing {:s}'.format(outfil))
    plt.tight_layout(pad=0.2,h_pad=0.0,w_pad=0.4)
    plt.subplots_adjust(hspace=0)
    pp.savefig(bbox_inches='tight')
    plt.close()
    # Finish
    pp.close()

def mfp_spec(outfil='Figures/mfp_spec.pdf', all_tau=None, scl=1.):
    """ Stacked spectrum for MFP
    """
    # SDSS
    zq = 4.
    hdu = fits.open('/Users/xavier/paper/LLS/taueff/Analysis/stack_DR7_z3.92_z4.02.fits')
    sdss_fx = hdu[0].data
    sdss_wave = hdu[2].data
    zem = 4.

    # Start the plot
    xmnx = (800, 1000)
    ymnx = (0., 1.0)
    pp = PdfPages(outfil)
    fig = plt.figure(figsize=(8.0, 5.0))

    plt.clf()
    gs = gridspec.GridSpec(1,1)

    # Lya line
    ax = plt.subplot(gs[0])
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
    #ax.xaxis.set_major_locator(plt.MultipleLocator(20.))
    #ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    #ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.set_xlim(xmnx)
    ax.set_ylim(ymnx)
    ax.set_ylabel('Relative Flux')
    ax.set_xlabel('Rest Wavelength (Ang)')

    lw = 1.5
    # Data
    ax.plot(sdss_wave, sdss_fx, 'k', linewidth=lw, label='SDSS QSOs (z=4)')

    # Label
    xexp = 860.
    headl = 6.
    ax.plot([911.7]*2, ymnx, '--', color='red', linewidth=1.5)
    ax.arrow(911.7, 0.6, xexp-911.7+headl, 0., linewidth=2,
        head_width=0.03, head_length=headl, fc='blue', ec='blue')
    csz = 19.
    ax.text(890., 0.63, r'r = $\lambda_{\rm mfp}$', color='blue', size=csz,
        ha='center')

    # exp(-1)
    y1 = 0.43
    y2 = np.exp(-1)*y1
    yplt = (y1+y2)/2.
    yerr = (y2-y1)/2.
    ax.errorbar([xexp], [yplt], yerr=yerr, color='green', linewidth=2,
        capthick=2)
    ax.plot([xexp,911.7], [y1]*2, ':', color='green', linewidth=2)
    ax.text(855., yplt, 'exp(-1)', color='green', size=csz, ha='right')

    #
    ax.text(0.05, 0.90, 'SDSS QSO stack (z=4)', color='black',
            transform=ax.transAxes, size=csz, ha='left') 
    xputils.set_fontsize(ax, 17.)
    # Layout and save
    print('Writing {:s}'.format(outfil))
    plt.tight_layout(pad=0.2,h_pad=0.0,w_pad=0.4)
    plt.subplots_adjust(hspace=0)
    pp.savefig(bbox_inches='tight')
    plt.close()
    # Finish
    pp.close()
    return all_tau

def dla_vary_NHI(outfil='Figures/dla_vary_NHI.pdf'):
    """ DLA profiles with NHI varying
    """
    # Wavelength array for my 'perfect' instrument
    wave = np.linspace(1160., 1270., 20000) * u.AA
    vel = (wave-1215.67*u.AA)/(1215.67*u.AA) * const.c.to('km/s')

    # Lya line
    lya = AbsLine(1215.6700*u.AA)
    #lya.attrib['N'] = 10.**(13.6)/u.cm**2
    lya.attrib['b'] = 30 * u.km/u.s
    lya.attrib['z'] = 0.

    aNHI = [20.3, 21., 21.5, 22.]

    # Start the plot
    xmnx = (-10000, 10000)
    ymnx = (0., 1.0)
    pp = PdfPages(outfil)
    fig = plt.figure(figsize=(8.0, 5.0))

    plt.clf()
    gs = gridspec.GridSpec(1,1)

    # Lya line
    ax = plt.subplot(gs[0])
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
    #ax.xaxis.set_major_locator(plt.MultipleLocator(20.))
    #ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    #ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.set_xlim(xmnx)
    ax.set_ylim(ymnx)
    ax.set_ylabel('Normalized Flux')
    ax.set_xlabel('Relative Velocity (km/s)') 

    lw = 1.5
    # Data
    for NHI in aNHI:
        lyai = copy.deepcopy(lya)
        lyai.attrib['N'] = 10**NHI / u.cm**2
        f_obsi = ltav.voigt_from_abslines(wave, [lyai])
        ax.plot(vel, f_obsi.flux, linewidth=lw, 
            label=r'$\log N_{\rm HI} = $'+'{:0.2f}'.format(NHI))

    # Legend
    legend = plt.legend(loc='lower left', scatterpoints=1, borderpad=0.3, 
        handletextpad=0.3, fontsize='large', numpoints=1)
    xputils.set_fontsize(ax, 17.)
    # Layout and save
    print('Writing {:s}'.format(outfil))
    plt.tight_layout(pad=0.2,h_pad=0.0,w_pad=0.4)
    plt.subplots_adjust(hspace=0)
    pp.savefig(bbox_inches='tight')
    plt.close()
    # Finish
    pp.close()

def real_dla_vary_NHI(outfil='Figures/real_dla_vary_NHI.pdf'):
    """ DLA profiles with NHI varying
    """
    # Wavelength array for my 'perfect' instrument
    files = ['/Users/xavier/GRB/data/080607/LRIS/GRB080607_B600N.fits']
    lbls  = [r'GRB080607: $\log N_{\rm HI} = 22.7$']
    zdlas = [3.03626]

    # Start the plot
    xmnx = (-20000, 20000)
    ymnx = (0., 1.0)
    pp = PdfPages(outfil)
    fig = plt.figure(figsize=(8.0, 5.0))

    plt.clf()
    gs = gridspec.GridSpec(1,1)

    # Lya line
    ax = plt.subplot(gs[0])
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
    #ax.xaxis.set_major_locator(plt.MultipleLocator(20.))
    #ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    #ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.set_xlim(xmnx)
    ax.set_ylim(ymnx)
    ax.set_ylabel('Normalized Flux')
    ax.set_xlabel('Relative Velocity (km/s)') 

    lw = 1.5
    # Data
    for zdla,lbl,fil in zip(zdlas,lbls,files):
        spec = lsio.readspec(fil)
        vel = spec.relative_vel(1215.6701*u.AA*(1+zdla))
        ax.plot(vel, spec.flux, linewidth=lw, 
            label=lbl)

    # Legend
    legend = plt.legend(loc='lower left', scatterpoints=1, borderpad=0.3, 
        handletextpad=0.3, fontsize='large', numpoints=1)
    xputils.set_fontsize(ax, 17.)
    # Layout and save
    print('Writing {:s}'.format(outfil))
    plt.tight_layout(pad=0.2,h_pad=0.0,w_pad=0.4)
    plt.subplots_adjust(hspace=0)
    pp.savefig(bbox_inches='tight')
    plt.close()
    # Finish
    pp.close()

def dla_deviation(outfil='Figures/dla_deviation.pdf'):
    """ Deviations from Voigt (Lee 2003)
    """
    # Wavelength array for my 'perfect' instrument
    wave = np.linspace(1140., 1290., 20000) * u.AA
    vel = (wave-1215.67*u.AA)/(1215.67*u.AA) * const.c.to('km/s')
    nu = (const.c/wave).to('Hz')

    # Lya line
    lya = AbsLine(1215.6700*u.AA)
    lya.attrib['N'] = 10.**(22)/u.cm**2
    lya.attrib['b'] = 30 * u.km/u.s
    lya.attrib['z'] = 0.

    # Lee
    sigmaT = (6.65e-29 * u.m**2).to('cm**2')  # Approximate!
    f_jk = 0.4162
    nu_jk = (const.c/lya.wrest).to('Hz')
    def eval_cross_Lee(nu):
        return sigmaT * (f_jk/2)**2 * (nu_jk/(nu-nu_jk))**2 * (1-1.792*(nu-nu_jk)/nu_jk)
    tau_lee = lya.attrib['N']*eval_cross_Lee(nu)
    flux_lee = np.exp(-1*tau_lee.value)
    # Start the plot
    xmnx = (-15000, 15000)
    ymnx = (0., 1.0)
    pp = PdfPages(outfil)
    fig = plt.figure(figsize=(8.0, 5.0))

    plt.clf()
    gs = gridspec.GridSpec(1,1)

    # Lya line
    ax = plt.subplot(gs[0])
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
    #ax.xaxis.set_major_locator(plt.MultipleLocator(20.))
    #ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    #ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.set_xlim(xmnx)
    ax.set_ylim(ymnx)
    ax.set_ylabel('Normalized Flux')
    ax.set_xlabel('Relative Velocity (km/s)') 

    f_voigt = ltav.voigt_from_abslines(wave, [lya])
    lw = 2
    ax.plot(vel, f_voigt.flux, 'k', linewidth=lw, 
        label=r'Voigt: $\log N_{\rm HI} = 22$')
    ax.plot(vel, flux_lee, 'r', linewidth=lw, 
        label=r'Lee2003: $\log N_{\rm HI} = 22$')

    # Legend
    legend = plt.legend(loc='lower left', scatterpoints=1, borderpad=0.3, 
        handletextpad=0.3, fontsize='large', numpoints=1)
    xputils.set_fontsize(ax, 17.)
    # Layout and save
    print('Writing {:s}'.format(outfil))
    plt.tight_layout(pad=0.2,h_pad=0.0,w_pad=0.4)
    plt.subplots_adjust(hspace=0)
    pp.savefig(bbox_inches='tight')
    plt.close()
    # Finish
    pp.close()

def drho_dNHI(outfil='Figures/drho_dNHI.pdf'):
    """ Differential contribution to rho_HI
    """
    # Wavelength array for my 'perfect' instrument
    fnmodel = FNModel.default_model()

    rhoHI, cumul, lgNHI = fnmodel.calculate_rhoHI(2.5, (12., 22.5), cumul=True)
    cumul = cumul/cumul[-1] / (lgNHI[1]-lgNHI[0]) # dlogN
    diff = cumul - np.roll(cumul,1)
    diff[0] = diff[1]

    # Start the plot
    xmnx = (16., 22.5)
    ymnx = (0., 1.0)
    pp = PdfPages(outfil)
    fig = plt.figure(figsize=(8.0, 5.0))

    plt.clf()
    gs = gridspec.GridSpec(1,1)

    # Lya line
    ax = plt.subplot(gs[0])
    #ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
    #ax.xaxis.set_major_locator(plt.MultipleLocator(20.))
    #ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    #ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.set_xlim(xmnx)
    #ax.set_ylim(ymnx)
    ax.set_ylabel(r'Normalized $d\rho_{\rm HI} / d\log N_{\rm HI}$')
    ax.set_xlabel(r'$\log N_{\rm HI}$')

    ax.plot(lgNHI, diff, 'b')

    # Legend
    #legend = plt.legend(loc='lower left', scatterpoints=1, borderpad=0.3, 
    #    handletextpad=0.3, fontsize='large', numpoints=1)
    xputils.set_fontsize(ax, 17.)
    # Layout and save
    print('Writing {:s}'.format(outfil))
    plt.tight_layout(pad=0.2,h_pad=0.0,w_pad=0.4)
    plt.subplots_adjust(hspace=0)
    pp.savefig(bbox_inches='tight')
    plt.close()
    # Finish
    pp.close()

def set_fontsize(ax,fsz):
    '''
    Generate a Table of columns and so on
    Restrict to those systems where flg_clm > 0

    Parameters
    ----------
    ax : Matplotlib ax class
    fsz : float
      Font size
    '''
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fsz)


#### ########################## #########################
#### ########################## #########################
#### ########################## #########################

def main(flg_fig):

    if flg_fig == 'all':
        flg_fig = np.sum( np.array( [2**ii for ii in range(15)] ))
    else:
        flg_fig = int(flg_fig)

    # EW
    if (flg_fig % 2**1) >= 2**0:
        example_ew()

    # LSF
    if (flg_fig % 2**2) >= 2**1:
        lsf()

    # FJ0812
    if (flg_fig % 2**3) >= 2**2:
        fj0812()

    # QSO SED
    if (flg_fig % 2**4) >= 2**3:
        qso_sed()

    # QSO template
    if (flg_fig % 2**5) >= 2**4:
        qso_template()

    # Redshift
    if (flg_fig % 2**6) >= 2**5:
        redshift()

    # J1422
    if (flg_fig % 2**7) >= 2**6:
        q1422()

    # IGM with Redshift
    if (flg_fig % 2**8) >= 2**7:
        #evolving_forest()
        evolving_forest_in_chapter()

    # dteff
    if (flg_fig % 2**9) >= 2**8:
        dteff()

    # IGM transmission
    if (flg_fig % 2**10) >= 2**9:
        sawtooth()

    # IGM transmission observed!
    if (flg_fig % 2**11) >= 2**10:
        obs_sawtooth()

    # WFC3 QSO
    if (flg_fig % 2**12) >= 2**11:
        wfc3_qso()

    # dX/dz
    if (flg_fig % 2**13) >= 2**12:
        dXdz()

    # teff_LL
    if (flg_fig % 2**14) >= 2**13:
        teff_LL()

    # mfp_spec
    if (flg_fig % 2**15) >= 2**14:
        mfp_spec()

    # Idealized DLA with NHI
    if (flg_fig % 2**16) >= 2**15:
        dla_vary_NHI()

    # DLA with NHI
    if (flg_fig % 2**17) >= 2**16:
        real_dla_vary_NHI()

    # DLA deviation
    if (flg_fig % 2**18) >= 2**17:
        dla_deviation()

    # QSO FUV
    if (flg_fig % 2**19) >= 2**18:
        qso_fuv()

    # QSO FUV
    if (flg_fig % 2**20) >= 2**19:
        drho_dNHI()



# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1:
        flg_fig = 0
        #flg_fig += 2**0   # Example of EW
        #flg_fig += 2**1   # LSF
        #flg_fig += 2**2   # FJ0812
        #flg_fig += 2**3   # QSO SED
        #flg_fig += 2**4   # QSO Template
        #flg_fig += 2**5   # Redshift
        #flg_fig += 2**6   # Q1422
        flg_fig += 2**7   # Evolving IGM
        #flg_fig += 2**8   # dteff
        #flg_fig += 2**9   # IGM transmission
        #flg_fig += 2**10   # IGM transmission
        #flg_fig += 2**11   # WFC3 QSO
        #flg_fig += 2**12   # dXdz
        #flg_fig += 2**13   # teff_LL
        #flg_fig += 2**14   # MFP stacked spectrum
        #flg_fig += 2**15   # DLA with NHI
        #flg_fig += 2**16   # real DLA 
        #flg_fig += 2**17   # DLA deviation 
        #flg_fig += 2**18   # QSO FUV
        #flg_fig += 2**19   # QSO FUV
    else:
        flg_fig = sys.argv[1]

    main(flg_fig)
