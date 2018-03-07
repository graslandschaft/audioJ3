# -*- coding: utf-8 -*-

import sys
import os
import glob
import platform
import argparse
from collections import OrderedDict

try:
    from PyQt4 import QtGui, QtCore, Qt
except Exception, details:
    print 'Unfortunately, your system misses the PyQt4 packages.'
    quit()

#############

# from audioio import AudioLoader, PlayAudio, fade
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar

import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
import matplotlib.mlab as ml
import matplotlib.colors as mc
import matplotlib.widgets as widgets
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.dates import date2num, num2date
from matplotlib.widgets import SpanSelector

import soundfile as sf

# ######################################################

try:
    from audioio import PlayAudio
    playback = True
except ImportError:
    playback = False
    print('No playback available')

# ######################################################

debug = 0

class DataViewer(QtGui.QMainWindow):
    ti = 0  # counter for iterating through timeframes
    playback = None

    def __init__(self, app, filename, start=0, stop=1., startx=0, stopx=-1., 
                 timeframes=None, timeframesx=None, events=None,
                 outpath='./'):
        QtCore.QObject.__init__(self, None)
        self.app = app

        self.filename = filename
        self.start = start
        self.startx = startx
        self.stop = stop
        self.stopx = stopx
        self.timeframes = timeframes
        self.timeframesx = timeframesx
        self.events = np.array(events) if events is not None else events
        self.outpath = outpath

        self.name = 'Audian'

        self.width = 1800
        self.height = 900
        self.offset_left = 30
        self.offset_top = 30
        self.max_tab_width = 1000
        self.min_tab_width = 400

        self.setGeometry(self.offset_left, self.offset_top, self.width, self.height)
        self.setSizePolicy(Qt.QSizePolicy.Maximum, Qt.QSizePolicy.Maximum)
        self.setMinimumSize(self.width, self.height)
        self.setWindowTitle(self.name)

        # #################

        if os.name == 'posix':
            self.label_font_size = 18
        else:
            self.label_font_size = 12

        # #################

        # LAYOUTS
        self.main = QtGui.QWidget()
        self.setCentralWidget(self.main)

        self.main_layout = QtGui.QVBoxLayout()
        self.main.setLayout(self.main_layout)

        # #################

        if os.name == 'posix':
            self.label_font_size = 18
        else:
            self.label_font_size = 12

        # #################
        # DATA
        self.data = Data(self, self.filename)

        # PLUGINS
        self.audiotrace = AudioTrace(self)
        self.spec = AudioSpectogramm(self)

        if self.timeframes is not None and self.timeframesx is not None:
            self.tfedit = TimeFrameEditor(self)

        if playback:
            self.playback = Playback(self)

        # #################

        # #################
        # CONFIG
        self.create_actions()
        self.handle_options()
        self.update()

    def create_actions(self):
        # Goto somewhere
        self.actionGoTo = QtGui.QAction('Goto',self)
        self.actionGoTo.setShortcut(QtGui.QKeySequence("Ctrl+G"))
        self.actionGoTo.triggered.connect(self.goto_somewhere)
        self.addAction(self.actionGoTo)

        # Change Y-scale
        self.actionMoveRight = QtGui.QAction('MoveRight',self)
        self.actionMoveRight.setShortcut(Qt.Qt.Key_Right)
        self.actionMoveRight.triggered.connect(self.moveRight)
        self.addAction(self.actionMoveRight)

        # Change Y-scale
        self.actionMoveLeft = QtGui.QAction('MoveLeft',self)
        self.actionMoveLeft.setShortcut(Qt.Qt.Key_Left)
        self.actionMoveLeft.triggered.connect(self.moveLeft)
        self.addAction(self.actionMoveLeft)

        # Change X-scale
        self.actionZoomIn = QtGui.QAction('ZoomIn',self)
        self.actionZoomIn.setShortcut(Qt.Qt.Key_X)
        self.actionZoomIn.triggered.connect(self.zoomIn)
        self.addAction(self.actionZoomIn)

        # Change X-scale
        self.actionZoomOut = QtGui.QAction('ZoomOut',self)
        self.actionZoomOut.setShortcut(Qt.Qt.SHIFT+Qt.Qt.Key_X)
        self.actionZoomOut.triggered.connect(self.zoomOut)
        self.addAction(self.actionZoomOut)

        # Previous timeframe
        self.actionPrevFrame = QtGui.QAction('Previous Timeframe',self)
        self.actionPrevFrame.setShortcut(Qt.Qt.Key_PageUp)
        self.actionPrevFrame.triggered.connect(self.previousTimeframe)
        self.addAction(self.actionPrevFrame)

        # Next timeframe
        self.actionNextFrame = QtGui.QAction('Next Timeframe',self)
        self.actionNextFrame.setShortcut(Qt.Qt.Key_PageDown)
        self.actionNextFrame.triggered.connect(self.nextTimeframe)
        self.addAction(self.actionNextFrame)

        # Save Data cutout
        self.actionNextFrame = QtGui.QAction('Save Audio Cutout',self)
        self.actionNextFrame.setShortcut(QtGui.QKeySequence("Ctrl+S"))
        self.actionNextFrame.triggered.connect(self.data.save_cutout)
        self.addAction(self.actionNextFrame)

        # Switch between raw and filtered data
        self.actionNextFrame = QtGui.QAction('Switch Dataset',self)
        self.actionNextFrame.setShortcut(QtGui.QKeySequence("Tab"))
        self.actionNextFrame.triggered.connect(self.switch)
        self.addAction(self.actionNextFrame)

    def switch(self):
        self.data.switch()
        self.update()

    def goto_somewhere(self):

        # asks the user to what point in the recording he wants to go
        dlg = QtGui.QInputDialog()
        dlg.setInputMode(QtGui.QInputDialog.TextInput)
        dlg.setWindowTitle('Go-to-time')
        dlg.resize(500,200)
        ok = dlg.exec_()  # shows the dialog
        if not ok: 
            # announce that something is not good
            return

        t = str(dlg.textValue()).split(' ')
        if len(t) == 1:
            start = float(t[0])
            stop = start + 1.
        elif len(t) == 2:
            start = float(t[0])
            stop = float(t[1])
        else:
            return

        self.startx = int(float(start)*self.data.audio.samplerate)
        self.stopx = int(float(stop)*self.data.audio.samplerate)
        if self.startx > self.data.audio.frames:
            return

        # check and set new position
        self.setWindowTitle(self.name)
        self.start = 1.*self.startx / self.data.audio.samplerate
        self.stop = 1.*self.stopx / self.data.audio.samplerate
        self.update()

    def moveRight(self):
        # increase time by half of the window
        # print('right')
        moven = (self.stopx - self.startx)/4
        moven = min((moven, self.data.audio.frames-self.stopx))
        self.startx += moven
        self.stopx += moven
        self.update_startstop()
        self.update()

    def moveLeft(self):
        # decrease time by half of the window
        # print('left')
        moven = (self.stopx - self.startx)/4
        moven = min((moven, self.startx))
        self.startx -= moven
        self.stopx -= moven
        self.update_startstop()
        self.update()

    def zoomIn(self):
        quart = (self.stopx-self.startx)/4
        self.startx = self.startx + quart
        self.stopx = self.stopx - quart
        self.update_startstop()
        self.update()

    def zoomOut(self):
        quart = (self.stopx-self.startx)/4
        right = min((quart, self.data.audio.frames-self.stopx))
        left = min((quart, self.startx))
        self.stopx += right
        self.startx -= left
        self.update_startstop()
        self.update()

    def update_startstop(self):
        self.start = 1.*self.startx/self.data.audio.samplerate
        self.stop = 1.*self.stopx/self.data.audio.samplerate

    def previousTimeframe(self):
        # print('previous timeframe')
        if self.ti > 0:
            self.ti -= 1
        self.updateTimeframe()

    def nextTimeframe(self):
        # print('next timeframe')
        t = len(self.timeframes) if self.timeframes is not None else 0
        tx = len(self.timeframesx) if self.timeframesx is not None else 0
        length = max((t, tx))
        if self.ti+1 < length:
            self.ti += 1
        self.updateTimeframe()

    def updateTimeframe(self):
        if self.timeframesx is None and self.timeframes is None: return
        if self.timeframesx is not None:
            startx, stopx = self.timeframesx[self.ti]
            self.startx = startx
            self.stopx = stopx
            length = len(self.timeframesx)
        else:
            start, stop = self.timeframes[self.ti]
            self.startx = int(float(start)*self.data.audio.samplerate)
            self.stopx = int(float(stop)*self.data.audio.samplerate)
            length = len(self.timeframes)
        perc = 100.*(self.ti+1)/length
        self.setWindowTitle(self.name + ' - {}/{} ({:.1f}%)'.format(self.ti+1, length, perc))
        self.startx -= max((0,int(0.2*self.data.audio.samplerate)))
        self.stopx += min((int(0.2*self.data.audio.samplerate), self.data.audio.frames))
        self.start = 1.*self.startx / self.data.audio.samplerate
        self.stop = 1.*self.stopx / self.data.audio.samplerate
        self.update()

    def update(self):
        self.data.load_data(self.startx, self.stopx)
        self.audiotrace.update()
        self.spec.update()

    def handle_options(self):
        # parser = argparse.ArgumentParser()
        # parser.add_argument('path', nargs='+', help='Path of a file or a folder of files.')
        # parser.add_argument('--start', default=-1., help='start time in seconds')
        # parser.add_argument('--startx', default=0, help='start index')
        # parser.add_argument('--stop', default=-1., help='stop time in seconds')
        # parser.add_argument('--stopx', default=-1, help='stop index')
        # args = parser.parse_args()
        # self.args = args

        # if args.start < 0.:
        #     self.startx = int(args.startx)
        # else:
        #     self.startx = int(float(args.start)*self.data.audio.samplerate)

        # if args.stopx > 0:
        #     self.stopx = int(args.stopx)
        # else:
        #     self.stopx = int(float(args.start)*self.data.audio.samplerate)

        # self.file = args.path
        # if not os.path.exists(self.file):
        #     print('File does not exists')
        #     self.close()

        if self.timeframes is None and self.timeframesx is None:
            if self.startx > 0:
                self.startx = int(self.startx)
            else:
                self.startx = int(float(self.start)*self.data.audio.samplerate)

            if self.stopx > 0:
                self.stopx = int(self.stopx)
            else:
                self.stopx = int(float(self.stop)*self.data.audio.samplerate)
        elif self.timeframes is not None:
            self.updateTimeframe()
        elif self.timeframesx is not None:
            self.updateTimeframe()
        else:
            print('whats happening?')
            self.close()

        if not os.path.exists(self.filename):
            print('File does not exists')
            self.close()

    def closeEvent(self, event):
        if self.playback is not None:
            self.playback.close()
        if len(self.audiotrace.tfedit.bad_ranges):
            self.audiotrace.tfedit.save()

class AudioSpectogramm(QtCore.QObject):
    ax = None
    defaults = dict(nfft=128,
                    fmax=1.)
    fmax=defaults['fmax']  # multiplication factor
    nfft = defaults['nfft']
    vmin = 60

    def __init__(self, main, parent=None):
        QtCore.QObject.__init__(self, parent)

        self.main = main
        self.create_actions()

        # THE PLOT
        self.fig = plt.figure()
        params = {'axes.labelsize': 22,
                  'font.size': 16,
                  'ytick.labelsize': 16,
                  'xtick.labelsize': 16}
        plt.rcParams.update(params)
        self.canvas = Canvas(self.fig, parent=self)
        # self.canvas.setMaximumHeight(400)
        # self.toolbar = NavigationToolbar(self.canvas)
        self.main.main_layout.addWidget(self.canvas)

    def create_actions(self):
        # Change Y-scale
        self.actionIncrease = QtGui.QAction('Increase Y',self)
        self.actionIncrease.setShortcut(Qt.Qt.SHIFT+Qt.Qt.Key_R)
        self.actionIncrease.triggered.connect(self.increaseNfft)
        self.main.addAction(self.actionIncrease)

        # Change Y-scale
        self.actionDecrease = QtGui.QAction('Decrease R',self)
        self.actionDecrease.setShortcut(Qt.Qt.Key_R)
        self.actionDecrease.triggered.connect(self.decreaseNfft)
        self.main.addAction(self.actionDecrease)

        # Change V-scale
        self.actionIncreaseV = QtGui.QAction('Increase V',self)
        self.actionIncreaseV.setShortcut(Qt.Qt.SHIFT+Qt.Qt.Key_V)
        self.actionIncreaseV.triggered.connect(self.increaseVmin)
        self.main.addAction(self.actionIncreaseV)

        # Change V-scale
        self.actionDecreaseV = QtGui.QAction('Decrease V',self)
        self.actionDecreaseV.setShortcut(Qt.Qt.Key_V)
        self.actionDecreaseV.triggered.connect(self.decreaseVmin)
        self.main.addAction(self.actionDecreaseV)

        # Change Fmax
        self.actionIncreaseF = QtGui.QAction('Increase F',self)
        self.actionIncreaseF.setShortcut(Qt.Qt.SHIFT+Qt.Qt.Key_F)
        self.actionIncreaseF.triggered.connect(self.increaseFmax)
        self.main.addAction(self.actionIncreaseF)

        # Change Fmax
        self.actionDecreaseF = QtGui.QAction('Decrease F',self)
        self.actionDecreaseF.setShortcut(Qt.Qt.Key_F)
        self.actionDecreaseF.triggered.connect(self.decreaseFmax)
        self.main.addAction(self.actionDecreaseF)

    def increaseNfft(self):
        # print('increase')
        self.nfft *= 2
        self.update()

    def decreaseNfft(self):
        # print('decrease')
        self.nfft = np.max((self.nfft/2, 16))
        self.update()

    def increaseVmin(self):
        if self.vmin < 90:
            self.vmin += 10
            self.update()

    def decreaseVmin(self):
        if self.vmin > 0:
            self.vmin -= 10
            self.update()

    def increaseFmax(self):
        if self.fmax*2 < 1.:
            self.fmax *= 2
        else:
            self.fmax = 1.
        self.update()

    def decreaseFmax(self):
        if self.fmax/2 > 0.005:
            self.fmax /= 2
            self.update()

    def set_data(self, data, samplerate):
        specpower, freqs, bins = ml.specgram(data, NFFT=self.nfft, Fs=samplerate, noverlap=self.nfft/2,
            detrend=ml.detrend_mean)
        # specpower[specpower<=0.0] = np.min(specpower[specpower>0.0]) # remove zeros
        z = specpower
        z = 10.*np.log10(specpower)
        z = np.flipud(z)
        toffset = 1.*self.main.startx / samplerate
        bins += toffset
        extent = toffset, np.amax(bins), freqs[0], freqs[-1]
        vmin = np.percentile(z, self.vmin)
        vmax = np.percentile(z, 100)

        if self.ax is None:          
            gs = gridspec.GridSpec(1, 1)
            gs.update(left=0.1, right=0.98, top=0.94, bottom=0.12, hspace=0.2, wspace=0.2)
            self.ax = self.fig.add_subplot(gs[0,0])

            cm = plt.get_cmap('jet')
            self.spectrogram_artist = self.ax.imshow(z, aspect='auto',
                                                     extent=extent, vmin=vmin, vmax=vmax,
                                                     cmap=cm, zorder=1, interpolation='bicubic')
            self.ax.set_xlabel('Time [s]', labelpad=0, fontsize=14)
            self.ax.set_ylabel('Frequency [Hz]', labelpad=10, fontsize=14)
        else :
            self.spectrogram_artist.set_data(z)
            self.spectrogram_artist.set_extent(extent)
            self.spectrogram_artist.set_clim(vmin, vmax)

        self.ax.set_title('nfft={}'.format(self.nfft), fontsize=14)
        self.ax.set_xlim(toffset, np.amax(bins))
        self.ax.set_ylim(0., freqs[-1]*self.fmax)
        self.fig.canvas.draw()

    def update(self):
        self.set_data(self.main.data.get_data(), self.main.data.audio.samplerate)

    def cutout(self, inx, out='./'):

        fig = plt.figure(figsize=(18./2.54, 8./2.54))
        params = {'font.size': 16,
                  'axes.labelsize': 16,
                  'axes.linewidth': .4,
                  'ytick.labelsize': 14,
                  'xtick.labelsize': 14}
        plt.rcParams.update(params)
        gs = gridspec.GridSpec(1, 1)
        gs.update(left=0.18, right=0.98, top=0.9, bottom=0.15, hspace=0.2, wspace=0.2)
        ax = fig.add_subplot(gs[0, 0])

        specpower, freqs, bins = ml.specgram(self.main.data.get_data(), NFFT=self.nfft, Fs=self.main.data.audio.samplerate, noverlap=self.nfft/2, detrend=ml.detrend_mean)
        # specpower[specpower<=0.0] = np.min(specpower[specpower>0.0]) # remove zeros
        z = specpower
        z = 10.*np.log10(specpower)
        z = np.flipud(z)
        toffset = 1.*self.main.startx / self.main.data.audio.samplerate
        bins += toffset
        extent = toffset, np.amax(bins), freqs[0], freqs[-1]
        vmin = np.percentile(z, self.vmin)
        vmax = np.percentile(z, 100)

        cm = plt.get_cmap('jet')
        ax.imshow(z, aspect='auto',
            extent=extent, vmin=vmin, vmax=vmax,
            cmap=cm, zorder=1, interpolation='bicubic')
        ax.set_xlabel('Time [s]', labelpad=0, fontsize=14)
        ax.set_ylabel('Frequency [Hz]', labelpad=10, fontsize=14)

        ax.set_title('nfft={}'.format(self.nfft), fontsize=14)
        ax.set_xlim(toffset, np.amax(bins))
        ax.set_ylim(0., freqs[-1]*self.fmax)

        fn = os.path.join(out, 'cutout_{:04d}_spec.png'.format(inx))
        fig.savefig(fn, dpi=300)
        plt.close()

class AudioTrace(QtCore.QObject):
    ax = None
    defaults = dict(ymax=1.,
                    maxn=100000)
    ymax = defaults['ymax']
    maxn = defaults['maxn']
    timeframe = None
    events = None
    envelope = None
    bads = None

    def __init__(self, main, debug=0, parent=None):
        QtCore.QObject.__init__(self, parent)

        self.main = main
        self.create_actions()

        # THE PLOT
        self.fig = plt.figure()
        params = {'axes.labelsize': 22,
                  'font.size': 16,
                  'ytick.labelsize': 16,
                  'xtick.labelsize': 16}
        plt.rcParams.update(params)
        self.canvas = Canvas(self.fig, parent=self)
        # self.canvas.setMaximumHeight(400)
        # self.toolbar = NavigationToolbar(self.canvas)
        self.main.main_layout.addWidget(self.canvas)
        
        self.label = QtGui.QLabel('')
        self.label_layout = QtGui.QHBoxLayout()
        self.label_layout.addStretch(0)
        self.label_layout.addWidget(self.label)
        self.label_layout.addStretch(0)
        self.main.main_layout.addLayout(self.label_layout)

        self.tfedit = TimeFrameEditor(self)

    def set_data(self, tdata, data, env):

        # init
        if self.ax is None:
            gs = gridspec.GridSpec(1, 1)
            gs.update(left=0.1, right=0.98, top=0.94, bottom=0.12, hspace=0.2, wspace=0.2)
            self.ax = self.fig.add_subplot(gs[0,0])
            self.trace_artist, = self.ax.plot(tdata, data, '-k', lw=1.5)
            self.envelope_artist, = self.ax.plot(tdata, env, '-', c='orangered', lw=1.5)
            self.ax.set_xlabel('Time [s]', labelpad=0, fontsize=14)

            # selector for timeframes
            self.span = SpanSelector(self.ax, self.onselect, 'horizontal', useblit=False,
                    rectprops=dict(alpha=0.5, facecolor='red'))
            self.span.set_active(False)

        else:
            self.trace_artist.set_data(tdata, data)
            self.envelope_artist.set_data(tdata, env)

        # clean up
        if self.timeframe is not None:
            self.timeframe.remove()
            self.timeframe = None
        if self.events is not None:
            for e in self.events:
                e.remove()
            self.events = None
        if self.bads is not None:
            for b in self.bads:
                b.remove()
            self.bads = None

        # indicate timeframe
        if self.main.timeframes is not None:
            start, stop = self.main.timeframes[self.main.ti]
            self.timeframe = self.ax.axvspan(start, stop, facecolor='lightgray',zorder=-10)
        # indicate events
        if self.main.events is not None:
            self.events = list()
            eve = self.main.events
            events = eve[(eve>=tdata[0])&(eve<=tdata[-1])]
            for e in events:
                self.events.append(self.ax.axvline(e, color='dodgerblue',zorder=10, lw=1.5))
        # indicate bad ranges
        if len(self.tfedit.bad_ranges):
            self.bads = list()
            for a,b in self.tfedit.bad_ranges:
                self.bads.append(self.ax.axvspan(a, b, facecolor='lightgreen',zorder=-8, alpha=0.3))

        self.ax.set_title('Timeframe: {:.3f} -- {:.3f} s'.format(tdata[0], tdata[-1]), fontsize=14)
        self.ax.set_ylim(-self.ymax, self.ymax)
        self.ax.set_xlim(tdata[0], tdata[-1])

        self.fig.canvas.draw()

    def update(self, draw=False):
        data = self.main.data.get_data()
        env = self.main.data.env
        tdata = self.main.data.tdata
        if data.size > self.maxn:
            # print('interpolation !!')
            step = data.size / self.maxn
            data = data[::step]
            env = env[::step]
            tdata = tdata[::step]
            # tmax = 1.*data.size/self.main.data.audio.samplerate
            # tdata_new = np.linspace(0., tmax, self.maxn)
            # data = np.interp(tdata_new, tdata, data)
            # tdata = tdata_new
        self.set_data(tdata, data, env)
        if draw:
            self.fig.canvas.draw()

    def create_actions(self):
        # Change Y-scale
        self.actionIncrease = QtGui.QAction('Increase Y',self)
        self.actionIncrease.setShortcut(Qt.Qt.SHIFT+Qt.Qt.Key_Y)
        self.actionIncrease.triggered.connect(self.increaseY)
        self.main.addAction(self.actionIncrease)

        # Change Y-scale
        self.actionDecrease = QtGui.QAction('Decrease Y',self)
        self.actionDecrease.setShortcut(Qt.Qt.Key_Y)
        self.actionDecrease.triggered.connect(self.decreaseY)
        self.main.addAction(self.actionDecrease)

    def increaseY(self):
        # print('increase')
        self.ymax *= 2.
        self.ymax = min((self.ymax, 1.))
        self.ax.set_ylim(-self.ymax, self.ymax)
        self.fig.canvas.draw()

    def decreaseY(self):
        # print('decrease')
        self.ymax /= 2.
        self.ax.set_ylim(-self.ymax, self.ymax)
        self.fig.canvas.draw()

    def onselect(self, xmin, xmax):
        self.tfedit.bad_ranges.append((xmin, xmax))
        self.update(draw=True)
        self.spanselector_toggle()
        
    def spanselector_toggle(self):
        if self.span.active:
            self.span.set_active(False)
            self.label.setText('')
        else:
            self.span.set_active(True)
            self.label.setText('Select bad range!')

    def cutout(self, inx, out='./'):

        data = self.main.data.get_data()
        env = self.main.data.env
        tdata = self.main.data.tdata
        if data.size > self.maxn:
            step = data.size / self.maxn
            data = data[::step]
            env = env[::step]
            tdata = tdata[::step]

        fig = plt.figure(figsize=(18./2.54, 8./2.54))
        params = {'font.size': 16,
                  'axes.labelsize': 16,
                  'axes.linewidth': .4,
                  'ytick.labelsize': 14,
                  'xtick.labelsize': 14}
        plt.rcParams.update(params)
        gs = gridspec.GridSpec(1, 1)
        gs.update(left=0.18, right=0.98, top=0.9, bottom=0.15, hspace=0.2, wspace=0.2)
        ax = fig.add_subplot(gs[0, 0])

        ax.plot(tdata, data, '-k', lw=.5)
        ax.plot(tdata, env, '-', c='orangered', lw=.5)
        ax.set_xlabel('Time [s]', labelpad=0, fontsize=14)

        # indicate timeframe
        if self.main.timeframes is not None:
            start, stop = self.main.timeframes[self.main.ti]
            ax.axvspan(start, stop, facecolor='lightgray',zorder=-10)
        # indicate events
        if self.main.events is not None:
            self.events = list()
            eve = self.main.events
            events = eve[(eve>=tdata[0])&(eve<=tdata[-1])]
            for e in events:
                self.events.append(ax.axvline(e, color='dodgerblue',zorder=10, lw=1.5))
        # indicate bad ranges
        if len(self.tfedit.bad_ranges):
            self.bads = list()
            for a,b in self.tfedit.bad_ranges:
                self.bads.append(ax.axvspan(a, b, facecolor='lightgreen',zorder=-8, alpha=0.3))

        ax.set_title('Timeframe: {:.3f} -- {:.3f} s'.format(tdata[0], tdata[-1]), fontsize=14)
        ax.set_ylim(-self.ymax, self.ymax)
        ax.set_xlim(tdata[0], tdata[-1])

        fn = os.path.join(out, 'cutout_{:04d}_trace.png'.format(inx))
        fig.savefig(fn, dpi=300)
        plt.close()

class TimeFrameEditor():
    bad_ranges = list()

    def __init__(self, trace):
        self.trace = trace
        self.create_actions()
        self.bad_ranges = self.load()

    def create_actions(self):

        # Toggle Span selector
        self.trace.main.actionSpan = QtGui.QAction('Toggle span selector',self.trace.main)
        self.trace.main.actionSpan.setShortcut(Qt.Qt.ALT+Qt.Qt.Key_S)
        self.trace.main.actionSpan.triggered.connect(self.trace.spanselector_toggle)
        self.trace.main.addAction(self.trace.main.actionSpan)

        # Add full view as bad range
        self.trace.main.actionBadCurrent = QtGui.QAction('Bad view',self.trace.main)
        self.trace.main.actionBadCurrent.setShortcut(Qt.Qt.Key_B)
        self.trace.main.actionBadCurrent.triggered.connect(self.bad_current_range)
        self.trace.main.addAction(self.trace.main.actionBadCurrent)

        # Add full view as bad range
        self.trace.main.actionAllBad = QtGui.QAction('All bad',self.trace.main)
        self.trace.main.actionAllBad.setShortcut(Qt.Qt.ALT+Qt.Qt.Key_A)
        self.trace.main.actionAllBad.triggered.connect(self.all_bad)
        self.trace.main.addAction(self.trace.main.actionAllBad)

        # Remove ranges in view
        self.trace.main.actionRemove = QtGui.QAction('Remove ranges',self.trace.main)
        self.trace.main.actionRemove.setShortcut(Qt.Qt.ALT+Qt.Qt.Key_R)
        self.trace.main.actionRemove.triggered.connect(self.remove_bad)
        self.trace.main.addAction(self.trace.main.actionRemove)

    def bad_current_range(self):
        # Add current range bad range
        if self.trace.main.timeframes is not None:
            start, stop = self.trace.main.timeframes[self.trace.main.ti]
        elif self.trace.main.timeframesx is not None:
            rate = self.trace.main.data.audio.samplerate
            startx, stopx = self.trace.main.timeframesx[self.trace.main.ti]
            start, stop = 1.*startx/rate, 1.*stopx/rate

        print('bad range added: {:.4f}--{:.4f}'.format(start, stop))
        self.bad_ranges.append([start, stop])
        self.trace.update(draw=True)

    def all_bad(self):
        start = 0.
        stop = 1.*self.trace.main.data.audio.frames / self.trace.main.data.audio.samplerate
        print('bad range added: {:.4f}--{:.4f}'.format(start, stop))
        self.bad_ranges.append([start, stop])
        self.trace.update(draw=True)

    def remove_bad(self):
        # remove all ranges in view
        ## only those fully in view!
        good = list()
        start = 1.*self.trace.main.startx / self.trace.main.data.audio.samplerate
        stop = 1.*self.trace.main.stopx / self.trace.main.data.audio.samplerate
        for i, (a,b) in enumerate(self.bad_ranges):
            if a < stop and b > start:
                good.append(i)
        for i in reversed(good):
            self.bad_ranges.pop(i)
        self.trace.update(draw=True)

    def load(self):
        file = self.trace.main.data.file
        inx = os.path.basename(file).split('_')[0]
        fn = '{}_badrange.txt'.format(inx)
        fn = os.path.join(self.trace.main.outpath, fn)
        if not os.path.exists(fn):
            return list()
        print('loading range-file...')
        with open(fn, 'r') as f:
            data = [line.split() for line in f.readlines()]
            return [(float(a), float(b)) for a,b in data]

    def save(self):
        file = self.trace.main.data.file
        inx = os.path.basename(file).split('_')[0]
        fn = '{}_badrange.txt'.format(inx)
        fn = os.path.join(self.trace.main.outpath, fn)
        with open(fn, 'w') as f:
            for a,b in self.bad_ranges:
                f.write('{:.5f} {:.5f}\n'.format(a,b))
                f.write('{:.5f} {:.5f}\n'.format(a,b))
        print('ranges saved.')

class Data():
    raw = None
    processed = None
    env = None
    audio = None
    startx = 0
    stopx = 0

    current_dataset = 'processed'

    # def __init__(self, main, file, highpass_on=True, highpass_cf=3000.,
    def __init__(self, main, file, highpass_on=True, highpass_cf=500.,
                 lowpass_on=True, lowpass_cf=9000.):
    # def __init__(self, main, file, highpass_on=True, highpass_cf=70.,
    #              lowpass_on=True, lowpass_cf=9000.):
        self.main = main
        self.highpass_on = highpass_on
        self.highpass_cf = highpass_cf
        self.lowpass_on = lowpass_on
        self.lowpass_cf = lowpass_cf

        self.file = file
        self.open_datafile(file)

    def open_datafile(self, file):
        if not os.path.exists(file):
            print('File does not exist')
            self.main.app.quit()
        self.audio = AudioLoader(file)

    def load_data(self, startx, stopx):
        self.raw = self.audio.readn(startx, stopx-startx)

        ## extract with audioio ...
        if self.audio.channels > 1:
            self.rawsingle = np.mean(self.raw, axis=1)
        else:
            self.rawsingle = self.raw

        self.process()

        # debug
        # sf.write('cutout.wav', self.raw/np.max(self.raw), self.audio.samplerate)

    def switch(self):
        if self.current_dataset == 'raw':
            self.current_dataset = 'processed'
        else:
            self.current_dataset = 'raw'

    def get_data(self):
        if self.current_dataset == 'raw':
            return self.rawsingle
        else:
            return self.processed

    def process(self):
        data = self.rawsingle
        rate = self.audio.samplerate
        if self.highpass_on:
            data = highpass_filter(rate, data, self.highpass_cf)
        if self.lowpass_on:
            data = lowpass_filter(rate, data, self.lowpass_cf)
        self.processed = data

        # time array
        self.tdata = np.arange(data.size, dtype=float)/self.audio.samplerate
        self.tdata += 1.*self.main.startx/self.audio.samplerate

        # calculate envelope
        self.env = envelope(rate, self.get_data())

    def save_cutout(self):
        i = 1  # index for save files
        while len(glob.glob(os.path.join(self.main.outpath, 'cutout_{:04d}*'.format(i)))):
            i += 1

        # save audio
        afn = os.path.join(self.main.outpath, 'cutout_{:04d}_raw.wav'.format(i))
        sf.write(afn, self.raw, self.audio.samplerate)
        afn = os.path.join(self.main.outpath, 'cutout_{:04d}_filtered.wav'.format(i))
        sf.write(afn, self.processed/np.max(self.processed), self.audio.samplerate)

        # save info
        ifn = os.path.join(self.main.outpath, 'cutout_{:04d}_info.txt'.format(i))
        with open(ifn, 'w') as f:
            f.write('data filename: {}\n'.format(self.main.filename))
            f.write('# start in seconds\n')
            f.write('start: {:.4f}\n'.format(1.*self.main.startx/self.audio.samplerate))
            f.write('# stop in seconds\n')
            f.write('stop: {:.4f}\n'.format(1.*self.main.stopx/self.audio.samplerate))
        
        self.main.audiotrace.cutout(i, self.main.outpath)
        self.main.spec.cutout(i, self.main.outpath)

        print('Cutout saved: {}'.format(i))

    def close_datafile(self):
        self.audio.close()
        self.audio = None


class AudioLoader(object):
    frames = 0
    samplerate = 0
    channels = 0

    def __init__(self, file):
        self.open(file)

    def open(self, file):
        self.sf = sf.SoundFile(file, 'r')
        self.samplerate = self.sf.samplerate
        self.frames = len(self.sf)
        self.channels = self.sf.channels
        # self.frames = self.sf.seek(0, soundfile.SEEK_END)

    def timeframe(self, a, b):
        # get data for timeframe
        ax = int(a*self.sf.samplerate)
        bx = int(b*self.sf.samplerate)
        return self.readn(ax, bx-ax)

    def readn(self, ax, n):
        # go to ax and read n
        ax = max((ax, 0))  # handle negative values
        if ax >= len(self.sf): return np.zeros((0,self.sf.channels))
        nn = min((n, len(self.sf)-ax))
        # nn = max((nn, 0))  # handle negative values

        if nn == 0: return np.zeros((0,self.sf.channels))
        self.sf.seek(ax)
        return self.sf.read(nn)

    def readt(self, a, t):
        # go to time a and read t seconds
        ax = int(a*self.sf.samplerate)
        n = int(t*self.sf.samplerate) - ax
        return self.readn(ax, n)

    # def readframes(self, n):
    #     # a generator working with while
    #     # NOT TESTED
    #     while self.sf.tell() < len(self.sf):
    #         nn = min((n, len(self.sf)-self.sf.tell()))
    #         nn = max((nn, 0))  # handle negative values
    #         if nn == 0: break
    #         yield self.sf.read(nn)

def highpass_filter(rate, data, cutoff, order=10):
    nyq = 0.5*rate
    high = cutoff/nyq
    b, a = sig.butter( 4, high, btype='highpass')
    # PLOT FILTERCURVE
    ## w, h = sig.freqz(b, a)
    ## plt.semilogx(w, 20 * np.log10(abs(h)))
    ## plt.show()
    for o in xrange(order):
        # print('highpass run {0}/{1}'.format(o+1, order))
        fdata = sig.filtfilt(b, a, data)
    return fdata

def lowpass_filter(rate, data, cutoff, order=10):
    nyq = 0.5*rate
    low = cutoff/nyq
    b, a = sig.butter( 4, low, btype='lowpass')
    # PLOT FILTERCURVE
    ## w, h = sig.freqz(b, a)
    ## plt.semilogx(w, 20 * np.log10(abs(h)))
    ## plt.show()
    for o in xrange(order):
        # print('lowpass run {0}/{1}'.format(o+1, order))
        fdata = sig.filtfilt(b, a, data)
        # fdata = sig.lfilter(b, a, data)
    return fdata

def envelope(rate, data, rstd_window_size_time = 0.001):
    """rstd_window_size_time in seconds """
    # width
    from scipy.signal import gaussian

    rstd_window_size = int(rstd_window_size_time * rate)
    # w = 1.0 * gaussian(rstd_window_size, std=rstd_window_size/7)
    w = 1.0 * np.ones(rstd_window_size)
    w /= np.sum(w)
    rstd = (np.sqrt((np.correlate(data**2, w, mode='same')-np.correlate(data,w, mode='same')**2)).ravel())*np.sqrt(2.)
    return rstd


class Canvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__( self, fig, parent=None ):

        FigureCanvas.__init__(self, fig)
        FigureCanvas.setSizePolicy(self,
            QtGui.QSizePolicy.Expanding,
            QtGui.QSizePolicy.Expanding)
        FigureCanvas.setMinimumSize(self, 400, 150)
        FigureCanvas.updateGeometry(self)

class Playback():
    def __init__(self, main):
        self.main = main
        self.audio = PlayAudio()
        self.create_actions()

    def create_actions(self):

        # Playback
        self.main.actionPlay = QtGui.QAction('Play data chunk',self.main)
        self.main.actionPlay.setShortcut(Qt.Qt.Key_P)
        self.main.actionPlay.triggered.connect(self.play)
        self.main.addAction(self.main.actionPlay)

    def play(self):
        os.close(sys.stderr.fileno())  # block error messages
        data = self.main.data.get_data() / self.main.data.get_data().max()
        rate = self.main.data.audio.samplerate
        self.audio.play(data, rate, blocking=False)
        os.open(sys.stderr.fileno())    # unblock error messages

    def close(self):
        self.audio.close()

def viewData(*args, **kwargs):
    '''
    # example for how to call this program from another python script
    from subprocess import call
    tf = 'timeframes:' + ';'.join(['{:.4f},{:.4f}'.format(a,b) for a,b in timeframes])
    events = 'events:' + ';'.join(['{:.4f}'.format(a) for a in pulsetimes])
    out = 'out:{}'.format(path)
    audian_path = 'audianJ.py'
    cmd = ('python', audian_path, fn, tf, events, out)
    call(cmd)
    '''
    qApp = QtGui.QApplication(sys.argv)  # create the main application
    # qApp.lastWindowClosed.connect(qApp.quit)
    main = DataViewer(qApp, *args, **kwargs)  # create the mainwindow instance
    main.show()  # show the mainwindow instance
    qApp.exec_()  # start the event-loop: no signals are sent or received without this.

# ######################################################
# ######################################################

if __name__=="__main__":

    # test-mode
    if len(sys.argv) > 1:
        fn = sys.argv[1]
        args = sys.argv[2:]
        kwargs = dict()
        for arg in args:
            key, data = arg.split(':')
            if key == 'timeframes':
                framelist = data.split(';')
                framelist = [frame.split(',') for frame in framelist]
                framelist = [[float(a), float(b)] for a,b in framelist]
                kwargs['timeframes'] = framelist
            if key == 'events':
                eventlist = data.split(';')
                eventlist = [float(a) for a in eventlist]
                kwargs['events'] = eventlist
            if key == 'out':
                kwargs['outpath'] = data

        viewData(fn, **kwargs)
    else:
        framelist = list([[0., 10.], [2., 5.], [6., 10.]])
        fn = 'test_data_midi.wav'
        viewData(fn, timeframes=framelist)
    # # remove all saved items
    # for item in glob.glob('cutout*'):
    #     os.remove(item)

