import FlowCal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import math

from scipy.stats import gaussian_kde


def plot_channel(s, channel_name):
    FlowCal.plot.hist1d(s, channel=channel_name)
    plt.show()
    
    
def plot_all_channels(s, nb_line=4):  
    nb_channel = len(s.channels)
    channel_per_line = nb_channel/nb_line
    fig,axs = plt.subplots(nb_line, math.ceil(channel_per_line))
    for i in range(nb_channel):
        x = math.floor(i/channel_per_line)
        y = math.floor(i%channel_per_line)
        if math.ceil(channel_per_line) != 1 : 
            plt.sca(axs[x,y])
        else:
            plt.sca(axs[x])
                    
        FlowCal.plot.hist1d(s, channel=s.channels[i])

    fig.set_size_inches(3*nb_line,2*nb_line)
    plt.tight_layout()
    plt.show()
    
    
def dot_plot(s, channel_1, channel_2):
    FlowCal.plot.density2d(s, channels=[channel_1, channel_2], mode='scatter')
    plt.show()


def dot_plot_personnalized(data, channel1, channel2, xscale=None, yscale=None, markers=[], colors=[], size=[], axesScale="linear", plot=plt):
    x = data.iloc[:, channel1]
    y = data.iloc[:, channel2]

    if len(size) != 0:
        plt.figure(figsize=(size[0], size[1]))

    if len(colors) == 0:
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)

        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        plot.scatter(x, y, c=z, s=0.2, alpha=0.9, cmap=plt.cm.jet)

    else:
        #one of the characters {'b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'}, 
        #which are short-hand notations for shades of blue, green, red, cyan, magenta, yellow, black, and white.
        #OR
        #{'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
        #'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'} (case-insensitive);
        LABEL_COLOR_MAP = {-1: 'tab:red', 0 : 'tab:cyan', 1 : 'tab:gray', 2:'tab:blue', 3:'tab:pink', 4:'tab:green', 5:'tab:olive',6:'tab:purple',7:'tab:brown'}
        label_color = [LABEL_COLOR_MAP[l] for l in colors]
        plot.scatter(x, y, c=label_color, s=0.2, alpha=0.9, cmap="Set1")

    if xscale != None :
        plot.gca().set_xlim(xscale[0], xscale[1])

    if yscale != None :
        plot.gca().set_ylim(yscale[0], yscale[1])

    if len(markers) != 0 :
        for marker in markers:
            plot.scatter(marker[0], marker[1], marker="+", linestyle='None', c='#0000ff')

    if plot == plt:
        plot.yscale(axesScale)
        plot.xscale(axesScale)
        plot.gca().set_xlabel(channel1)
        plot.gca().set_ylabel(channel2)

def dot_plot_all(data, channels, xscale=None, yscale=None):
    for i in range(len(channels)):
        for j in range(i+1, len(channels)):
            dot_plot_personnalized(data, channels[i], channels[j], xscale, yscale)

    

def dot_plot_ind(s, channel_1, channel_2):
    FlowCal.plot.density2d(s, channels=[s.channels[channel_1], s.channels[channel_2]], mode='scatter')
    plt.show()