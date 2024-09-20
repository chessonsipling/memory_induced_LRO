import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import torch
from scipy.stats import linregress

import plt_config
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') # for running on server without GUI
matplotlib.rcParams['agg.path.chunksize'] = 10000

color = plt.rcParams['axes.prop_cycle'].by_key()['color']

os.makedirs('fits', exist_ok=True)
os.makedirs('histograms', exist_ok=True)



# tds = np.linspace(50, 100.5, 102)
tds = [15, 17, 20, 25, 34, 51, 68, 77, 82, 85, 88]
# tds = [34, 51, 68, 77]
Ls = [64, 128, 256, 512]
n_ensembles = [100, 400, 2000, 4800]
# tds = [51]
# Ls = [64]
# n_ensembles = [100]
td_strings = [f'{td}'for td in tds]
window_length = 30

names = ['cluster_sizes', 'duration']
xlabels = ['Avalanche size s', 'Avalanche duration T']
var_names = ['s', 'T']


tds_corr = np.zeros((len(Ls), len(tds)))
correlation_lengths = np.zeros((len(Ls), len(tds)))
correlation_times = np.zeros((len(Ls), len(tds)))
for td_idx, td in enumerate(tds):
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    figs = [fig1, fig2]
    axs = [ax1, ax2]
    for L_idx, L in enumerate(Ls):
        folder = f'../correlation_length_no_limit/td_{td}'

        td_string = td_strings[td_idx]
        size_vs_duration = []
        for name_idx, name in enumerate(names):
            xs_PDF = []
            ys_PDF = []
            slopes_PDF = []
            intercepts_PDF = []
            try:
                name_string = f'{L}_{td}_30'
                n_ensembles_i = n_ensembles[L_idx]
                stats = []
                cluster_sizes = []
                for i in range(n_ensembles_i):
                    try:
                        cluster_sizes_i = torch.load(f'{folder}/{name}_{name_string}_{i}.pt')
                    except FileNotFoundError:
                        continue
                    except Exception as e:
                        print(e)
                        raise e
                    cluster_sizes_i = cluster_sizes_i.cpu().numpy()
                    if len(cluster_sizes_i) > 0:
                        cluster_sizes.append(cluster_sizes_i)

                if len(cluster_sizes) == 0:
                    continue
                cluster_sizes = np.concatenate(cluster_sizes)
                stats.append(cluster_sizes)
                size_vs_duration.append(cluster_sizes)

                log_cluster_sizes = np.log10(cluster_sizes[cluster_sizes > 0])
                std = np.std(log_cluster_sizes)
                bin_width = 3.5 * std / (len(log_cluster_sizes) ** (1 / 3))
                bin_width = max(bin_width, 0.02)
                hist_min = 0
                hist_max = 6
                n_bins = int((hist_max - hist_min) / bin_width)
                n_bins = max(n_bins, 1)
                bins = bin_width * np.arange(n_bins + 1)
                bins = (10 ** bins).astype(int)
                bins = np.unique(bins)

                hist, bin_edges = np.histogram(cluster_sizes, bins=bins)
                bin_sizes = np.diff(bins)
                hist = hist / bin_sizes
                hist = hist / hist.sum()
                bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2).astype(int)
                bin_centers = bin_centers[hist > 0]
                hist = hist[hist > 0]
                bin_centers = np.log10(bin_centers)
                xs_PDF.append(bin_centers)
                ys_PDF.append(hist)

                np.save(f'histograms/{name}_{L}_{td}.npy', [bin_centers, hist])

                try:
                    # fit = np.polyfit(np.log(bin_centers[fit_mask]), np.log(hist[fit_mask]), 1)
                    slope, intercept, r, p, se = linregress(bin_centers[:int(0.6*len(bin_centers))],
                                                            np.log10(hist[:int(0.6*len(bin_centers))]))
                    n_bins = len(bin_centers)
                    max_bin = bin_centers[-1]
                except:
                    slope, intercept, r, p, se = [np.nan, np.nan, np.nan, np.nan, np.nan]
                    n_bins = 0
                    max_bin = 0
                slopes_PDF.append(slope)
                intercepts_PDF.append(intercept)


                try:
                    # deviations = np.abs(np.log10(hist) - (slope * bin_centers + intercept))
                    # distance = np.diff(bin_centers)
                    # distance = np.concatenate([distance, [0]])
                    # deviations = deviations[bin_centers > 0.7]
                    # distance = distance[bin_centers > 0.7]
                    # max_dist_idx = np.argsort(distance)[::-1]
                    # distance = distance[max_dist_idx]
                    # power_law_mask = (deviations < 4)[max_dist_idx]
                    # max_dist_idx = max_dist_idx[power_law_mask]
                    # distance = distance[power_law_mask]

                    unique_cluster_sizes = np.unique(cluster_sizes)
                    unique_cluster_sizes = unique_cluster_sizes[unique_cluster_sizes > 0]
                    unique_cluster_sizes = np.sort(unique_cluster_sizes)
                    log_cluster_sizes = np.log10(unique_cluster_sizes)
                    distances = np.diff(log_cluster_sizes)
                    distances = np.concatenate([distances, [0]])
                    distance_idx = np.argsort(distances)[::-1]
                    distances = distances[distance_idx]

                    if name_idx == 0:
                        if (distances[0] < 0.5 and L == 64) or (distances[0] < 1 and L > 64):
                            corr_len = np.sqrt(np.max(unique_cluster_sizes))
                        else:
                            corr_len = np.sqrt(unique_cluster_sizes[distance_idx[0]])
                        tds_corr[L_idx, td_idx] = td
                        correlation_lengths[L_idx, td_idx] = corr_len
                    else:
                        if (distances[0] < 0.5 and L == 64) or (distances[0] < 1 and L > 64):
                            corr_time = np.max(unique_cluster_sizes)
                        else:
                            corr_time = unique_cluster_sizes[distance_idx[0]]
                        correlation_times[L_idx, td_idx] = corr_time

                    print(f'{name_string}\t{td}')

                except:
                    continue

                try:
                    axs[name_idx].scatter(10 ** bin_centers, hist, s=15, alpha=0.5, color=color[L_idx])
                    axs[name_idx].plot(10 ** bin_centers, 10 ** (slope * bin_centers + intercept), '--',
                                       color=color[L_idx], label=f'{L}x{L} ~${var_names[name_idx]}^{{{slope:.2f}}}$')
                    # axs[name_idx].text(5, 10 ** -1, f'$\sim s^{{{slope:.2f}}}$', fontsize=18)
                    axs[name_idx].set_xscale('log')
                    axs[name_idx].set_yscale('log')
                    axs[name_idx].set_xlabel(xlabels[name_idx])
                    axs[name_idx].set_ylabel(f'Probability P({var_names[name_idx]})')
                    axs[name_idx].set_title(f'$\\tau_D$={td}')
                except:
                    pass
            except:
                continue

    ax1.legend()
    fig1.savefig(f'fits/cluster_sizes_{td}.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    ax2.legend()
    fig2.savefig(f'fits/duration_{td}.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)

np.save(f'histograms/corr_length.npy', [tds_corr, correlation_lengths, correlation_times])