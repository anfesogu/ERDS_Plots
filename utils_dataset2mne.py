from scipy.io import loadmat 
import mne
import numpy as np
import pandas as pd


def data2mne(fname_mat, scale=1e-6):

    Data = loadmat(fname_mat)

    # Motor Execution
    EEG_ME_Left = Data['eeg']['movement_left'][0][0] 
    EEG_ME_Right = Data['eeg']['movement_right'][0][0]
    ME_Event = Data['eeg']['movement_event'][0][0]

    # Motor Imagery
    EEG_MI_Left = Data['eeg']['imagery_left'][0][0]
    EEG_MI_Right = Data['eeg']['imagery_right'][0][0]
    MI_Event = Data['eeg']['imagery_event'][0][0]

    # Resting State
    RS = Data['eeg']['rest'][0][0]

    # Blinking - Noise1
    Noise1 = Data['eeg']['noise'][0][0][0][0]

    # eyeball up-down - Noise2
    Noise2 = Data['eeg']['noise'][0][0][1][0]

    # eyeball left-right - Noise3
    Noise3 = Data['eeg']['noise'][0][0][2][0]

    # jaw clenching - Noise4
    Noise4 = Data['eeg']['noise'][0][0][3][0]

    # head movement - Noise5
    Noise5 = Data['eeg']['noise'][0][0][4][0]

    #General Info
    sfreq = Data['eeg']['srate'][0][0][0][0] #Sampling Frequency
    psenloc = Data['eeg']['psenloc'][0][0] #Sensor location projected to unit sphere
    senloc = Data['eeg']['senloc'][0][0] #Sensor locations 3D
    bad_trial_indices_voltage = Data['eeg']['bad_trial_indices'][0][0][0][0][0][0][0] #Bad trials indices voltage
    bad_trial_indices_mi = Data['eeg']['bad_trial_indices'][0][0][0][0][0][0][1] #Bad trials indices MI
    names_types = pd.read_excel(r'ch_names&types.xlsx') #Read file with ch names and types
    ch_types = names_types['ch_types'].tolist() #ch_types
    ch_names = names_types['ch_names'].tolist() #ch_names

    # Create stimuli channel
    Noise1_Event = np.zeros((1,Noise1.shape[1]))
    Noise1_Event[0,0] = 1
    Noise2_Event = np.zeros((1,Noise2.shape[1]))
    Noise2_Event[0,0] = 2
    Noise3_Event = np.zeros((1,Noise3.shape[1]))
    Noise3_Event[0,0] = 3
    Noise4_Event = np.zeros((1,Noise4.shape[1]))
    Noise4_Event[0,0] = 4
    Noise5_Event = np.zeros((1,Noise5.shape[1]))
    Noise5_Event[0,0] = 5
    RS_Event = np.zeros((1,RS.shape[1]))
    RS_Event[0,0] = 6
    ME_Event_Rigth = ME_Event * 7
    ME_Event_Left = ME_Event * 8
    MI_Event_Rigth = MI_Event * 9
    MI_Event_Left = MI_Event * 10

    # Join All Events and create stimuli channels
    Event_Stimuli = np.concatenate((Noise1_Event, Noise2_Event, Noise3_Event, Noise4_Event, Noise5_Event, RS_Event, ME_Event_Rigth, ME_Event_Left, MI_Event_Rigth, MI_Event_Left), axis=1)

    # Join all data
    Chs_Data = np.concatenate((Noise1, Noise2, Noise3, Noise4, Noise5, RS, EEG_ME_Right, EEG_ME_Left, EEG_MI_Right, EEG_MI_Left), axis=1) * scale #Convert from uV to V

    # Create Info
    info = mne.create_info(ch_names, sfreq, ch_types)

    # Create a montage based on the electrode positions
    SensorLoc = dict(zip(ch_names[0:64], psenloc.tolist()))
    montage = mne.channels.make_dig_montage(ch_pos=SensorLoc, coord_frame='head')

    # Include the montage in the info
    info.set_montage(montage)

    # Create raw object
    Raw_Data = np.concatenate((Chs_Data,Event_Stimuli), axis=0) 
    raw = mne.io.RawArray(Raw_Data, info)

    return raw

import asrpy
def apply_ASR(raw,tmin=51,tmax=115):

    # Creating ASR object
    asr = asrpy.ASR(sfreq=raw.info["sfreq"])

    # Filter the data - 0.5Hz Highpass
    filtered = raw.load_data().copy().filter(0.5,None)

    # Fit the ASR method with RS dataca
    asr.fit(filtered.copy().crop(tmin=tmin,tmax=tmax))

    # Applying the method to filtered data
    return asr.transform(filtered)

def create_name(subject, folder, type='read_mat'):
    if type == 'read_mat':
        fname = folder + '\s' + '{:02d}'.format(subject) + '.mat'
    else:
        fname = folder + '\s' + '{:02d}'.format(subject) + '_' + type + '.fif'
    return fname

import matplotlib.pyplot as plt
def plot_TF_M(tfr, event_ids,mode='mean',baseline=None,vmin=None,vmax=None,fmin=None,fmax=None,cmap='RdBu',comment='', reporting=False):

    if reporting:
        figures = []

    for event in event_ids:
        # select desired epochs for visualization
        tfr_ev = tfr[event]
        fig, axes = plt.subplots(
            1, 3, figsize=(12, 4), gridspec_kw={"width_ratios": [10, 10, 1]}
        )
        fig.canvas.manager.set_window_title(f"Time-Frequency Map ({event})")
        for ch, ax in enumerate(axes[:-1]):  # for each channel
            tfr_ev.average().plot([ch], baseline=baseline, mode=mode ,vmin=vmin, vmax=vmax, fmin=fmin, fmax=fmax, cmap=cmap,show=False,axes=ax,colorbar=False)

            ax.set_title(tfr.ch_names[ch], fontsize=10)
            ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
            if ch != 0:
                ax.set_ylabel("")
                ax.set_yticklabels("")
        fig.colorbar(axes[0].images[-1], cax=axes[-1]).ax.set_yscale("linear")
        fig.suptitle(f"Time-Frequency Map ({event}) " + comment)

        if reporting:
            figures.append(fig)
        plt.show()

    if reporting:
        return figures


from mne.stats import permutation_cluster_1samp_test as pcluster_test
from matplotlib.colors import TwoSlopeNorm
def plot_TF_M_stat(tfr, event_ids, mode='percent', cmap='RdBu', baseline=(-1, 0), vmin=-1, vmax=1.5, comment='', reporting=False):
    
    tfr.apply_baseline(baseline, mode=mode)
    
    cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center & max ERDS
    kwargs = dict(n_permutations=100, step_down_p=0.05, seed=1, buffer_size=None, out_type="mask")  # for cluster test

    if reporting:
        figures = []

    for event in event_ids:
        # select desired epochs for visualization
        tfr_ev = tfr[event]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={"width_ratios": [10, 10, 1]})
        fig.canvas.manager.set_window_title(f"Time-Frequency Map p<0.05 ({event})")
        for ch, ax in enumerate(axes[:-1]):  # for each channel
            # positive clusters
            _, c1, p1, _ = pcluster_test(tfr_ev.data[:, ch], tail=1, **kwargs)
            # negative clusters
            _, c2, p2, _ = pcluster_test(tfr_ev.data[:, ch], tail=-1, **kwargs)

            # note that we keep clusters with p <= 0.05 from the combined clusters
            # of two independent tests; in this example, we do not correct for
            # these two comparisons
            c = np.stack(c1 + c2, axis=2)  # combined clusters
            p = np.concatenate((p1, p2))  # combined p-values
            mask = c[..., p <= 0.05].any(axis=-1)

            # plot TFR (ERDS map with masking)
            tfr_ev.average().plot(
                [ch],
                cmap=cmap,
                cnorm=cnorm,
                axes=ax,
                colorbar=False,
                show=False,
                mask=mask,
                mask_style="mask"
            )

            ax.set_title(tfr.ch_names[ch], fontsize=10)
            ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
            if ch != 0:
                ax.set_ylabel("")
                ax.set_yticklabels("")
        fig.colorbar(axes[0].images[-1], cax=axes[-1]).ax.set_yscale("linear")
        fig.suptitle(f"Time-Frequency Map p<0.05 ({event})" + comment)

        if reporting:
            figures.append(fig)
        plt.show()
    
    if reporting:
        return figures

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
def plot_ERDs(tfr,channels,tmin=-1.5,tmax=3.5,baseline=(None,0),comment='', reporting=False, boundaries=[0,4,8,13,30,140]):

    # ERDS Plots 
    power_tfr = tfr.copy().crop(tmin,tmax).pick_channels(channels).apply_baseline(baseline=baseline, mode="percent") # apply baseline
    df = power_tfr.to_data_frame(time_format=None, long_format=True)

    # Map to frequency bands:
    freq_bounds = {"_": boundaries[0], "delta": boundaries[1], "theta": boundaries[2], "mu/alpha": boundaries[3], "beta": boundaries[4], "gamma": boundaries[5]}
    df["band"] = pd.cut(df["freq"], list(freq_bounds.values()), labels=list(freq_bounds)[1:])

    # Filter to retain only relevant frequency bands:
    freq_bands_of_interest = ["mu/alpha", "beta"]
    df = df[df.band.isin(freq_bands_of_interest)]
    df["band"] = df["band"].cat.remove_unused_categories()

    # Order channels for plotting:
    df["channel"] = df["channel"].cat.reorder_categories(("C3","C4"), ordered=True)

    g = sns.FacetGrid(df, row="band", col="channel", margin_titles=True)
    g.map(sns.lineplot, "time", "value", "condition", n_boot=10)
    axline_kw = dict(color="black", linestyle="dashed", linewidth=0.5, alpha=0.5)
    g.map(plt.axhline, y=0, **axline_kw)
    g.map(plt.axvline, x=0, **axline_kw)
    #g.set(ylim=(None, 1.5))
    g.set_axis_labels("Time (s)", "ERDS (%)")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.add_legend(ncol=2, loc="lower center")
    g.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.08)
    g.fig.suptitle("ERDS" + comment)

    if reporting:
        return g