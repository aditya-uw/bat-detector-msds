import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable
import math

import scipy
import numpy as np
import pandas as pd

from pathlib import Path
import soundfile as sf
from tqdm import tqdm
from sklearn.cluster import KMeans
import scipy.signal as signal
import scipy.special as special

import sys

# append the path of the
# parent directory
sys.path.append('..')
sys.path.append('../src/')
sys.path.append('../src/models/bat_call_detector/batdetect2/')

import src.batdt2_pipeline as batdetect2_pipeline
from pipeline import pipeline
from models.bat_call_detector.model_detector import BatCallDetector

import bout.clustering as clstr
import bout.assembly as bt

C = 343 # m/s speed of sound in air
FS = 250000
SNR_CALC_LENGTH = 0.030
SNR_OFFSET_BEFORE_CALL = 0.015
TIME_PAD_FOR_ECHO = 0.002
CORR_LENGTH = 0.025
NUM_CHANNELS_TOTAL = 8
FILE_DURATION = 600
HOUR_TAG = 'hour_0'
ZMAG_REF_TO_6 = 29.125
ZMAG_REF_TO_7 = 31.6875
XMAG_REF_TO_1 = 17.7
YMAG_REF_TO_6 = 21
YMAG_REF_TO_1 = 37.5
A_LOCS_MAT = (254/10000) * np.array([[-XMAG_REF_TO_1, -YMAG_REF_TO_1, ZMAG_REF_TO_7],
                    [-XMAG_REF_TO_1, YMAG_REF_TO_1, ZMAG_REF_TO_7],
                    [-XMAG_REF_TO_1, -YMAG_REF_TO_1, 0],
                    [-XMAG_REF_TO_1, YMAG_REF_TO_1, 0],
                    [-XMAG_REF_TO_1, -YMAG_REF_TO_6, -ZMAG_REF_TO_6],
                    [-XMAG_REF_TO_1, YMAG_REF_TO_6, -ZMAG_REF_TO_6],
                    [0, 0, ZMAG_REF_TO_7],
                    [0, 0, 0]])
ECHO_PAD = int(FS*TIME_PAD_FOR_ECHO)
SNR_CALC_LENGTH = 0.030
SNR_OFFSET_BEFORE_CALL = 0.015
TIME_PAD_FOR_ECHO = 0.002
CORR_LENGTH = 0.025
SPEC_NFFT = 32

def collect_channels(filepath, samplingrate, offset, duration_in_secs):
    channels = [[], [], [], [], [], [], [], []]

    with open(filepath, "rb") as rawFile:

        rawFile.seek(int(2*8*samplingrate*offset), 0)
        for i in range(samplingrate*duration_in_secs):
            for j in range(8):
                bytes = rawFile.read(2)
                sample_int = int.from_bytes(bytes = bytes, byteorder = "little", signed = True)
                sample_float = sample_int * (5 / 32768)

                channels[j].append(sample_float)

    return np.array(channels)

LABEL_FOR_GROUPS = {0: 'LF', 
                    1: 'HF'}


def plot_audio_seg_spectrogram(audio_features, spec_features):
    audio_seg = audio_features['audio_seg']
    fs = audio_features['sample_rate']
    start = audio_features['start']
    duration = audio_features['duration']

    vmax = spec_features['vmax']
    vmin = spec_features['vmin']
    cmap = spec_features['cmap']
    nfft = spec_features['NFFT']

    plt.figure(figsize=(15, 5))
    plt.rcParams.update({'font.size': 18})
    plt.title(f"Spectrogram", fontsize=24)
    plt.specgram(audio_seg+1e-6, NFFT=nfft, cmap=cmap, vmin=vmin, vmax=vmax, mode='magnitude', scale='dB')
    plt.yticks(ticks=np.linspace(0, 1, 6), labels=np.linspace(0, fs/2000, 6).astype('int'))
    plot_xtype = 'float'
    plt.xticks(ticks=np.linspace(0, duration*(fs/2), 11), 
            labels=np.round(np.linspace(start, start+duration, 11, dtype=plot_xtype), 2), rotation=30)
    plt.ylabel("Frequency (kHz)")
    plt.xlabel("Time (s)")
    plt.colorbar()
    plt.show()
    

def index_reference_call_based_on_feature_to_maximize(calls, detection_index, feature):
    return calls[detection_index, np.argmax(feature[detection_index]),:], np.argmax(feature[detection_index])

TEMPLATE_CHANNEL = 3
def index_reference_call_based_on_channel(calls, detection_index, channel):
    return calls[detection_index, channel,:], channel


def get_snr_from_band_limited_signal(snr_call_signal, snr_noise_signal): 

    signal_power_rms = np.sqrt(np.square(snr_call_signal).mean())
    noise_power_rms = np.sqrt(np.square(snr_noise_signal).mean())
    snr = (20 * np.log10(signal_power_rms / noise_power_rms))

    return snr

def bandpass_audio_signal(audio_seg, fs, low_freq_cutoff, high_freq_cutoff):
    nyq = fs // 2
    low_cutoff = (low_freq_cutoff) / nyq
    high_cutoff =  (high_freq_cutoff) / nyq
    b, a = scipy.signal.butter(4, [low_cutoff, high_cutoff], btype='band', analog=False)
    band_limited_audio_seg = scipy.signal.filtfilt(b, a, audio_seg)

    return band_limited_audio_seg

def highpass_audio_signal(audio_seg, fs, low_freq_cutoff):
    nyq = fs // 2
    low_cutoff = (low_freq_cutoff) / nyq
    b, a = scipy.signal.butter(4, low_cutoff, btype='high', analog=False)
    high_passed_audio_seg = scipy.signal.filtfilt(b, a, audio_seg)

    return high_passed_audio_seg

def compute_welch_psd_of_call(call, fs, audio_info):
    freqs, welch = scipy.signal.welch(call, fs=fs, detrend=False, scaling='spectrum')
    cropped_welch = welch[(freqs<=audio_info['max_freq_visible'])]
    audio_spectrum_mag = np.abs(cropped_welch)
    audio_spectrum_db =  10*np.log10(audio_spectrum_mag)
    normalized_audio_spectrum_db = audio_spectrum_db - audio_spectrum_db.max()

    thresh = -100
    peak_db = np.zeros(len(normalized_audio_spectrum_db))+thresh
    peak_db[normalized_audio_spectrum_db>=thresh] = normalized_audio_spectrum_db[normalized_audio_spectrum_db>=thresh]

    original_freq_vector = np.arange(0, len(peak_db), 1).astype('int')
    common_freq_vector = np.linspace(0, len(peak_db)-1, audio_info['num_points']).astype('int')
    interp_kind = 'linear'
    interpolated_points_from_welch = scipy.interpolate.interp1d(original_freq_vector, peak_db, kind=interp_kind)(common_freq_vector)

    return interpolated_points_from_welch

def get_section_of_call_in_file(detection, audio_file):
    fs = audio_file.samplerate
    num_frames = audio_file.frames
    file_length = num_frames/fs

    call_dur = (detection['end_time'] - detection['start_time'])
    pad = min(min(detection['start_time'] - call_dur, file_length - detection['end_time']), 0.006) / 3
    start = detection['start_time'] - call_dur - (3*pad)
    duration = ((2 * call_dur) + (4*pad))

    try:
        audio_file.seek(int(fs*start))
        audio_seg = audio_file.read(int(fs*duration))
        length_of_section = call_dur + (2*pad)
    except sf.LibsndfileError as e:
        print(f'Start time : {start} invalid, detection starts at {detection["start_time"]} in segment:{audio_file.name}')
        audio_seg = None
        length_of_section = 0

    return audio_seg, length_of_section, pad

def gather_features_of_interest(dets, kmean_welch, audio_file):
    fs = audio_file.samplerate
    features_of_interest = dict()
    features_of_interest['call_signals'] = []
    features_of_interest['welch_signals'] = []
    features_of_interest['snrs'] = []
    features_of_interest['peak_freqs_welch'] = []
    features_of_interest['peak_freqs_spec'] = []
    features_of_interest['peak_freq_times_spec'] = []
    features_of_interest['peak_freqs'] = []
    features_of_interest['classes'] = []
    nyquist = fs//2
    for index, row in dets.iterrows():
        call_dur = (row['end_time'] - row['start_time'])
        audio_seg, length_of_section, pad = get_section_of_call_in_file(row, audio_file)
        seg_start = row['start_time'] - call_dur - (3*pad)

        freq_pad = 2000
        low_freq_cutoff = row['low_freq']-freq_pad
        high_freq_cutoff = min(nyquist-1, row['high_freq']+freq_pad)
        band_limited_audio_seg = bandpass_audio_signal(audio_seg, fs, low_freq_cutoff, high_freq_cutoff)

        signal_for_peaks = band_limited_audio_seg.copy()
        signal_for_peaks[:int(fs*(length_of_section+pad))] = 0
        signal_for_peaks[-int(fs*pad):] = 0
        mpl_specgram_window = plt.mlab.window_hanning(np.ones(32))
        f, t, Sxx = scipy.signal.spectrogram(signal_for_peaks, fs, detrend=False,
                                    nfft=32, 
                                    window=mpl_specgram_window)
        plt_Sxx = 10*np.log10(Sxx)
        max_ind = np.where(plt_Sxx==np.max(plt_Sxx))
        max_value_across_bins = Sxx[max_ind]
        peak_freq = f[max_ind[0]]
        peak_freq_time = t[max_ind[1]]
        tp_valid = ((seg_start+peak_freq_time[0]) >= row["start_time"])&((seg_start+peak_freq_time[0]) <= row["end_time"])
        if ~tp_valid:
            print(f'Call starting at {row["start_time"]} and ending at {row["end_time"]}, t_peak={round(seg_start+peak_freq_time[0], 4)} and f_peak={round(peak_freq[0], 4)} valid? {tp_valid}')
            print(f't_peak={round(seg_start+peak_freq_time[0], 4)} means peak_freq_time={peak_freq_time[0]} happened because plt_Sxx==np.max(plt_Sxx)={np.max(plt_Sxx)}')

        if math.isinf(np.max(plt_Sxx)):
            features_of_interest['peak_freqs_spec'].append((row['low_freq']+row['high_freq'])/2)
            features_of_interest['peak_freq_times_spec'].append((row['start_time']+row['end_time'])/2)
        else:
            features_of_interest['peak_freqs_spec'].append(peak_freq[0])
            features_of_interest['peak_freq_times_spec'].append(seg_start+peak_freq_time[0])

        signal = band_limited_audio_seg.copy()
        signal[:int(fs*(length_of_section))] = 0
        noise = band_limited_audio_seg - signal
        snr_call_signal = signal[-int(fs*length_of_section):]
        snr_noise_signal = noise[:int(fs*length_of_section)]
        features_of_interest['call_signals'].append(snr_call_signal)

        snr = get_snr_from_band_limited_signal(snr_call_signal, snr_noise_signal)
        features_of_interest['snrs'].append(snr)

        welch_info = dict()
        welch_info['num_points'] = 100
        max_visible_frequency = 96000
        welch_info['max_freq_visible'] = max_visible_frequency
        welch_signal = compute_welch_psd_of_call(snr_call_signal, fs, welch_info)
        features_of_interest['welch_signals'].append(welch_signal)

        peaks = np.where(welch_signal==max(welch_signal))[0][0]
        features_of_interest['peak_freqs_welch'].append(max_visible_frequency*(peaks/len(welch_signal)))
        
        welch_signal = (welch_signal).reshape(1, len(welch_signal))
        features_of_interest['classes'].append(kmean_welch.predict(welch_signal)[0])

    features_of_interest['call_signals'] = np.array(features_of_interest['call_signals'], dtype='object')

    return features_of_interest

def open_and_get_call_info(audio_file, dets):
    welch_key = 'all_locations'
    output_dir = Path(f'../kmeans_training_set')
    output_file_type = 'top1_inbouts_welch_signals'
    welch_data = pd.read_csv(output_dir / f'2022_{welch_key}_{output_file_type}.csv', index_col=0, low_memory=False)
    k = 2
    kmean_welch = KMeans(n_clusters=k, n_init=10, random_state=1).fit(welch_data.values)

    features_of_interest = gather_features_of_interest(dets, kmean_welch, audio_file)

    dets.reset_index(drop=True, inplace=True)

    dets['sampling_rate'] = len(dets) * [audio_file.samplerate]
    dets.insert(0, 'SNR', features_of_interest['snrs'])
    dets.insert(0, 'peak_frequency', features_of_interest['peak_freqs'])
    dets.insert(0, 'KMEANS_CLASSES', pd.Series(features_of_interest['classes']).map(LABEL_FOR_GROUPS))

    return features_of_interest['call_signals'], dets

def classify_calls_from_file(bd2_predictions, data_params):
    file_path = Path(data_params['audio_file'])
    audio_file = sf.SoundFile(file_path)
    call_signals, dets = open_and_get_call_info(audio_file, bd2_predictions.copy())
    return dets

def run_models(file_mappings):
    """
    Runs the batdetect2 model to detect bat search-phase calls in the provided audio segments and saves detections into a .csv.

    Parameters
    ------------
    file_mappings : `List`
        - List of dictionaries generated by initialize_mappings()

    Returns
    ------------
    bd_dets : `pandas.DataFrame`
        - A DataFrame of detections that will also be saved in the provided output_dir under the above csv_name
        - 7 columns in this DataFrame: start_time, end_time, low_freq, high_freq, detection_confidence, event, input_file
        - Detections are always specified w.r.t their input_file; earliest start_time can be 0 and latest end_time can be 1795.
        - Events are always "Echolocation" as we are using a model that only detects search-phase calls.
    """

    bd_dets = pd.DataFrame()
    for i in tqdm(range(len(file_mappings))):
        cur_seg = file_mappings[i]
        bd_annotations_df = cur_seg['model']._run_batdetect(cur_seg['audio_seg']['audio_file'])
        bd_preds_classed = classify_calls_from_file(bd_annotations_df, cur_seg['audio_seg'])
        bd_offsetted = pipeline._correct_annotation_offsets(
                bd_preds_classed,
                cur_seg['original_file_name'],
                cur_seg['audio_seg']['offset']
            )
        bd_dets = pd.concat([bd_dets, bd_offsetted])

    return bd_dets

def run_pipeline_on_file(file, cfg):
    bd_preds = pd.DataFrame()

    if not cfg['output_dir'].is_dir():
        cfg['output_dir'].mkdir(parents=True, exist_ok=True)
    if not cfg['tmp_dir'].is_dir():
        cfg['tmp_dir'].mkdir(parents=True, exist_ok=True)

    if (cfg['run_model']):
        cfg["csv_filename"] = f"batdetect2_pipeline_{file.name.split('.')[0]}"
        filepath = (cfg['output_dir'] / f'{cfg["csv_filename"]}.csv')
        if not(filepath.is_file()):
            print(f'Generating detections from {file}')
            segmented_file_paths = batdetect2_pipeline.generate_segmented_paths([file], cfg)
            file_path_mappings = batdetect2_pipeline.initialize_mappings(segmented_file_paths, cfg)
            bd_preds = run_models(file_path_mappings)
            if cfg['save']:
                batdetect2_pipeline._save_predictions(bd_preds, cfg['output_dir'], cfg)
            batdetect2_pipeline.delete_segments(segmented_file_paths)
        else:
            bd_preds = pd.read_csv(filepath)

    return bd_preds

def remove_overlapping_events(channel_dets, OVERLAP_TIME_THRESHOLD=8e-3):

    dist_mat_comp_batdetect2_df = channel_dets.copy()
    association_mat = np.ones((len(dist_mat_comp_batdetect2_df), len(dist_mat_comp_batdetect2_df)), dtype='bool')
    for index in range(len(dist_mat_comp_batdetect2_df)):
        row = dist_mat_comp_batdetect2_df.iloc[index]
        dist_to_all_calls = ((dist_mat_comp_batdetect2_df['peak_frequency_time_SPECTROGRAM'] - row['peak_frequency_time_SPECTROGRAM']).values)

        considered_inds = np.where(np.abs(dist_to_all_calls)<=OVERLAP_TIME_THRESHOLD)[0]
        considered_dets = channel_dets.iloc[considered_inds]

        det_choices = np.zeros(len(considered_inds))
        likely_call_ind = considered_dets['det_prob'].argmax()

        det_choices[likely_call_ind] = 1
        association_mat[considered_inds, index] = det_choices

    removed_overlaps = channel_dets[np.logical_and.reduce(association_mat, axis=1)]

    return removed_overlaps

def get_approx_call_dur(det):
    det_dur = (det['end_time'] - det['start_time'])
    return det_dur/3

def extract_bandpassed_detection_from_wavfilepath(wav_filepath, start, length, det):
    recorded_audio = sf.SoundFile(wav_filepath)
    fs = recorded_audio.samplerate
    recorded_audio.seek(int(fs*start))
    audio_seg = recorded_audio.read(int(fs*length))
    bandpassed_audio_seg = bandpass_audio_signal(audio_seg, fs, det['low_freq']-2000, det['high_freq']+10000)
    return bandpassed_audio_seg

def extract_highpassed_detection_from_wavfilepath(wav_filepath, start, length, det):
    recorded_audio = sf.SoundFile(wav_filepath)
    fs = recorded_audio.samplerate
    recorded_audio.seek(int(fs*start))
    audio_seg = recorded_audio.read(int(fs*length))
    highpassed_audio_seg = highpass_audio_signal(audio_seg, fs, det['low_freq']-2000)
    return highpassed_audio_seg

def generalized_cross_correlation_to_find_t_delay(signal1, signal2):
    correlation = signal.correlate(signal1, signal2, mode='full')
    lags = signal.correlation_lags(len(signal1), len(signal2), mode='full')
    max_lag_index = np.argmax(correlation)
    time_delay_in_samples = lags[max_lag_index]
    return time_delay_in_samples

def index_reference_call_based_on_feature_to_maximize(calls, detection_index, feature):
    return calls[detection_index, np.argmax(feature[detection_index]),:]

def plot_audio_seg_spec(audio_features, spec_features):
    audio_seg = audio_features['audio_seg']
    fs = audio_features['sample_rate']
    start = audio_features['start']
    duration = audio_features['duration']

    vmax = spec_features['vmax']
    vmin = spec_features['vmin']
    cmap = spec_features['cmap']
    nfft = spec_features['NFFT']

    plt.figure(figsize=audio_features['figsize'])
    plt.rcParams.update({'font.size': 18})
    plt.title(f"Spectrogram of {audio_features['plot_title']}", fontsize=24)
    plt.specgram(audio_seg+1e-6, NFFT=nfft, cmap=cmap, vmin=vmin, vmax=vmax, mode='magnitude', scale='dB')
    plt.yticks(ticks=np.linspace(0, 1, 6), labels=np.linspace(0, fs/2000, 6).astype('int'))
    plot_xtype = 'float'
    if (duration > 60):
        plot_xtype = 'int'
    plt.xticks(ticks=np.linspace(0, duration*(fs/2), 11), 
               labels=np.round(np.linspace(start, start+duration, 11, dtype=plot_xtype), 2), rotation=30)
    plt.ylabel("Frequency (kHz)")
    plt.xlabel("Time (s)")
    plt.colorbar()
    plt.show()

def plot_audio_seg_signal(audio_features):
    audio_seg = audio_features['audio_seg']
    fs = audio_features['sample_rate']
    start = audio_features['start']
    duration = audio_features['duration']

    plt.figure(figsize=audio_features['figsize'])
    plt.rcParams.update({'font.size': 18})
    plt.title(f"Signal of {audio_features['plot_title']}", fontsize=24)
    plt.plot(audio_seg)
    plot_xtype = 'float'
    plt.xlim(0, duration*fs)
    plt.ylim(-3, 3)
    plt.grid(which='both')
    plt.xticks(ticks=np.linspace(0, audio_features['duration']*fs, 11), 
                labels=np.round(np.linspace(start, start+duration, 11, dtype=plot_xtype), 2), rotation=30)
    plt.ylabel("Voltage (V)")
    plt.xlabel("Time (s)")
    plt.show()

def plot_audio_seg_fft(audio_features):
    audio_seg = audio_features['audio_seg']
    fs = audio_features['sample_rate']
    start = audio_features['start']
    duration = audio_features['duration']

    plt.figure(figsize=audio_features['figsize'])
    plt.rcParams.update({'font.size': 18})
    plt.title(f"FFT of {audio_features['plot_title']}", fontsize=24)
    abs_sig = np.abs(scipy.fft.rfft(audio_seg, n=len(audio_seg)))
    abs_sig = abs_sig / len(abs_sig)
    plt.plot(20*np.log10(abs_sig/np.max(abs_sig)))
    plot_xtype = 'int'
    plt.xticks(ticks=np.linspace(0, len(abs_sig), 11), 
                labels=np.round(np.linspace(0, (fs/2), 11, dtype=plot_xtype), 2)/1e3, rotation=30)
    plt.ylabel("Voltage (dB)")
    plt.xlabel("Frequency (kHz)")
    plt.grid(which='both')
    plt.show()
    
FREQ_COLORS = {'LF':'cyan',
               'HF':'orange'}

def plot_colored_dets_over_audio(audio_features, spec_features, plot_dets):
    audio_seg = audio_features['audio_seg']
    fs = audio_features['sample_rate']
    start = audio_features['start']
    file_offset = audio_features['file_offset']
    duration = audio_features['duration']

    vmax = spec_features['vmax']
    vmin = spec_features['vmin']
    cmap = spec_features['cmap']
    nfft = spec_features['NFFT']

    plt.figure(figsize=audio_features['figsize'])
    plt.rcParams.update({'font.size': 24})
    plt.title(f"Spectrogram of {audio_features['plot_title']}", fontsize=24)
    plt.specgram(audio_seg, NFFT=nfft, cmap=cmap, vmin=vmin, vmax=vmax, mode='magnitude', scale='dB')

    ax = plt.gca()
    for i, row in plot_dets.iterrows():
        rect = patches.Rectangle(((row['start_time'] - (start+file_offset))*(fs/2), row['low_freq']/(fs/2)), 
                        (row['end_time'] - row['start_time'])*(fs/2), (row['high_freq'] - row['low_freq'])/(fs/2), 
                        linewidth=2, edgecolor=FREQ_COLORS[row['KMEANS_CLASSES']], facecolor='none', alpha=0.8)
        plt.axvline(x=(row['peak_frequency_time_SPECTROGRAM'] - start)*(fs/2), color='w', linestyle='dashed')
        plt.text(x=(row['start_time'] - (start+file_offset))*(fs/2), y=row['low_freq']/(fs/2), s=f"{round(row['SNR'], 2)}dB"
                 , color='w', fontsize=8, fontweight='bold')
        ax.add_patch(rect)

    plt.yticks(ticks=np.linspace(0, 1, 6), labels=np.linspace(0, fs/2000, 6).astype('int'))
    plot_xtype = 'float'
    if (duration > 60):
        plot_xtype = 'int'
    plt.xticks(ticks=np.linspace(0, duration*(fs/2), 11), 
               labels=np.round(np.linspace((start+file_offset), (start+file_offset)+duration, 11, dtype=plot_xtype), 2), rotation=30)
    plt.ylabel("Frequency (kHz)")
    plt.xlabel("Time (s)")
    plt.show()

def get_dets_observed_from_all_channels(file_dir, file_offset):
    FILE_TIME_TAG = f'{int(file_offset)}to{int(file_offset+FILE_DURATION)}'
    dir_name = f'{HOUR_TAG}_{FILE_TIME_TAG}'
    write_dir = file_dir / dir_name

    cfg = dict()
    cfg["time_expansion_factor"] = 1.0
    # Offset (seconds) from the beginning of the audio file to start processing
    cfg["start_time"] = 0.0
    # Input audio is divided into segments of this duration (seconds), each processed individually
    cfg["segment_duration"] = 30.0
    cfg["models"] = [BatCallDetector(detection_threshold=0.35,
                                    spec_slices=False,
                                    chunk_size=2,
                                    time_expansion_factor=1.0,
                                    quiet=False,
                                    cnn_features=True)]
    cfg['tmp_dir'] = Path('../output/tmp')
    cfg['output_dir'] = write_dir
    cfg['run_model'] = True
    cfg['should_csv'] = True
    cfg['save'] = True

    channel_dets = pd.DataFrame()
    for selected_channel in range(NUM_CHANNELS_TOTAL):
        write_file_for_indiv_channel = write_dir / f'{HOUR_TAG}_channel{selected_channel}_{FILE_TIME_TAG}.WAV'
        cfg['output_dir'] = write_dir
        cfg['input_audio'] = Path(write_file_for_indiv_channel)
        dets = run_pipeline_on_file(write_file_for_indiv_channel, cfg)
        plot_dets = dets.copy()
        plot_dets['start_time'] += file_offset
        plot_dets['end_time'] += file_offset
        channel_dets = pd.concat([channel_dets, plot_dets])

    return channel_dets

def plot_audio_bout(write_file, selected_channel_for_ref, start, length, file_offset, subset1, subset2):
    recorded_audio = sf.SoundFile(write_file)
    recorded_audio.seek(int(FS*start))
    audio_seg = recorded_audio.read(int(FS*length))
    highpassed_audio_seg = highpass_audio_signal(audio_seg, FS, 20000)

    print(f'Currently at {write_file}')
    audio_features = dict()
    audio_features['figsize'] = (15, 3)
    audio_features['file_path'] = write_file
    audio_features['audio_seg'] = highpassed_audio_seg
    audio_features['sample_rate'] = FS
    audio_features['start'] = start
    audio_features['file_offset'] = file_offset
    audio_features['duration'] = length

    spec_features = dict()
    spec_features['NFFT'] = 512 # When segments are short, NFFT should also be small to best see calls (must always be > 128)
    spec_features['cmap'] = 'jet' # This colormap shows best contrast between noise and signals
    spec_features['vmin'] = -90
    spec_features['vmax'] = 0

    audio_features['plot_title'] = f"{audio_features['file_path'].stem} (mic {selected_channel_for_ref+1})"
    plot_colored_dets_over_audio(audio_features, spec_features, pd.DataFrame())
    plot_colored_dets_over_audio(audio_features, spec_features, subset1)
    plot_colored_dets_over_audio(audio_features, spec_features, subset2)
    
def snr_quality_metric(snrcb, snreb, snrce):
    return ((snrcb+snreb)/snrce)

def extract_call_starts_for_call(call_segment, approx_call_dur):
    p_energy_cumsum = np.cumsum((call_segment)**2)/np.sum(call_segment**2)
    cumsum_start = np.where(p_energy_cumsum>0.3)[0][0]
    found_call_dur = int(FS*(approx_call_dur+0.002))
    found_call_start = cumsum_start-int(FS*0.002)
    return found_call_start, found_call_dur

def extract_reference_call_only(ref_call_segment, approx_call_dur):
    mpl_specgram_window = plt.mlab.window_hanning(np.ones(SPEC_NFFT))
    f, t, Sxx = scipy.signal.spectrogram(ref_call_segment, FS, detrend=False,
                                nfft=SPEC_NFFT, 
                                window=mpl_specgram_window)
    Sxx[np.where(Sxx==0)] = 1e-16 ### <--- replace all zeros with very small values (-160dB) outside of the scale to avoid taking log10(0)
    plt_Sxx = 10*np.log10(Sxx)
    max_ind = np.where(plt_Sxx==np.max(plt_Sxx))
    max_value_across_bins = Sxx[max_ind]
    peak_freq = f[max_ind[0]]
    peak_freq_time = t[max_ind[1]]

    found_call_dur = int(FS*(0.008))
    found_call_start = int(FS*(peak_freq_time[0] - 0.004))
    
    found_call_end = (found_call_start+found_call_dur)
    ref_mic_call_only = ref_call_segment[found_call_start:found_call_end]

    return ref_mic_call_only, found_call_start/FS, found_call_dur/FS

def extract_approx_call_windows_for_snr_calcs(filtered_det, call_start, call_dur):
    call_end = (call_start+call_dur)
    found_call = filtered_det[max(0, call_start):min(call_end, len(filtered_det))]
    found_echo = filtered_det[min(call_end+ECHO_PAD, len(filtered_det)-1):min(call_end+ECHO_PAD+call_dur, len(filtered_det))]
    found_background = filtered_det[max(0, call_start-call_dur):call_start]
    return found_call, found_background, found_echo

def extract_call_from_detection(det, selected_channel, write_dir, file_offset, approx_call_dur):
    file_time_tag = f'{int(file_offset)}to{int(file_offset+FILE_DURATION)}'
    start = det['start_time'] - file_offset - SNR_OFFSET_BEFORE_CALL
    wav_file = write_dir / f'{HOUR_TAG}_channel{selected_channel}_{file_time_tag}.WAV'
    filtered_det = extract_highpassed_detection_from_wavfilepath(wav_file, start, SNR_CALC_LENGTH, det)
    found_call_start, found_call_dur = extract_call_starts_for_call(filtered_det, approx_call_dur)
    found_call, found_background, found_echo = extract_approx_call_windows_for_snr_calcs(filtered_det, 
                                                                                                found_call_start, 
                                                                                                found_call_dur)
    return filtered_det, [found_call, found_background, found_echo]

def find_relevant_tdoa_arrays(set_for_tdoa, microphones_used, file_offset, write_dir):
    num_good_channels = microphones_used.shape[0]

    calls_for_tdoa = np.zeros((set_for_tdoa.shape[0], num_good_channels, int(FS*SNR_CALC_LENGTH)), dtype=np.float64)
    call_snrs_for_tdoa = np.zeros((set_for_tdoa.shape[0], num_good_channels), dtype=np.float64)
    snr_facs_for_tdoa = np.zeros((set_for_tdoa.shape[0], num_good_channels), dtype=np.float64)
    call_voltage_rms = np.zeros((set_for_tdoa.shape[0], num_good_channels), dtype=np.float64)
    det_durs_for_tdoa = np.zeros(set_for_tdoa.shape[0], dtype=np.float64)

    for det_num, det in set_for_tdoa.iterrows():
        approx_call_dur = get_approx_call_dur(det)
        det_durs_for_tdoa[det_num] = approx_call_dur
        channel_num = 0
        for selected_channel in (microphones_used-1):
            filtered_det, det_comps = extract_call_from_detection(det, selected_channel, write_dir, file_offset, approx_call_dur)
            found_call, found_background, found_echo = det_comps[0], det_comps[1], det_comps[2] 
            snrce = get_snr_from_band_limited_signal(found_call, found_echo)
            snrcb = get_snr_from_band_limited_signal(found_call, found_background)
            snreb = get_snr_from_band_limited_signal(found_echo, found_background)

            calls_for_tdoa[det_num, channel_num,:] = filtered_det
            call_snrs_for_tdoa[det_num, channel_num] = snrcb
            call_voltage_rms[det_num, channel_num] = (np.max(found_call) - np.min(found_call))/(2*np.sqrt(2))
            snr_facs_for_tdoa[det_num, channel_num] = snr_quality_metric(snrcb, snreb, snrce)
            channel_num+=1

    return calls_for_tdoa, call_voltage_rms, snr_facs_for_tdoa, det_durs_for_tdoa

def find_d_mics_to_ref_using_GCC(set_for_tdoa, microphones_used, calls_for_tdoa, snr_facs_for_tdoa, det_durs_for_tdoa):
    num_good_channels = microphones_used.shape[0]
    t_delay_wrt_selected_channel = np.zeros((set_for_tdoa.shape[0], num_good_channels), dtype=np.float64)
    for det_num in range(calls_for_tdoa.shape[0]):
        non_ref_calls_of_det = calls_for_tdoa[det_num]
        approx_call_dur = det_durs_for_tdoa[det_num]
        ref_call_of_det = index_reference_call_based_on_feature_to_maximize(calls_for_tdoa, det_num, snr_facs_for_tdoa)
        ref_mic_call_only = extract_reference_call_only(ref_call_of_det, approx_call_dur)

        for non_ref_call_i in range(non_ref_calls_of_det.shape[0]):
            signal1 = non_ref_calls_of_det[non_ref_call_i,:int(FS*CORR_LENGTH)]
            signal2 = ref_mic_call_only
            time_delay_in_samples = generalized_cross_correlation_to_find_t_delay(signal1, signal2)
            time_delay_in_seconds = time_delay_in_samples / FS
            t_delay_wrt_selected_channel[det_num, non_ref_call_i] = time_delay_in_seconds
            
    return t_delay_wrt_selected_channel

def assemble_source_points_in_meters(d_mics_to_ref_meters, microphones_used, A_locs_mat_tdoa_meters):
    num_good_channels = microphones_used.shape[0]
    num_nonref_channels = num_good_channels-1

    bucket = np.empty((4, 0))
    for i in range(d_mics_to_ref_meters.shape[0]):
        A_mat = np.hstack(((0 - A_locs_mat_tdoa_meters), d_mics_to_ref_meters[i,:].reshape((num_nonref_channels, 1))))

        xm = A_locs_mat_tdoa_meters[:,0]
        ym = A_locs_mat_tdoa_meters[:,1]
        zm = A_locs_mat_tdoa_meters[:,2]

        wm0 = ((d_mics_to_ref_meters[i,:]**2) - (xm**2) - (ym**2) - (zm**2))/2
        wm0 = wm0.reshape((num_nonref_channels, 1))

        A_pinv = np.linalg.pinv(A_mat)
        source_vec = np.dot(A_pinv, wm0)

        bucket = np.hstack([bucket, source_vec])

    return bucket

def plot_three_plane_trajectories(ax1, ax2, ax3, dets, good_bucket, bucket, mic_locs, start, length):
    cmap = plt.get_cmap('viridis')
    map_vals = (dets['SNR'])
    color_vals = np.linspace(start, start+length, bucket.shape[1])
    norm = plt.Normalize(color_vals.min(), color_vals.max())
    line_colors = cmap(norm(color_vals))
    ax1.plot(good_bucket[0,:], good_bucket[1,:], alpha=0.4, color='k')
    ax1.scatter(bucket[0,:], bucket[1,:], color=line_colors, alpha=1, s=50)
    ax1.scatter(x=mic_locs[:,0], y=mic_locs[:,1], facecolor='yellow', edgecolor='k')
    ax1.scatter(x=0, y=0, facecolor='yellow', edgecolor='k')

    ax2.plot(good_bucket[1,:], good_bucket[2,:], alpha=0.4, color='k')
    ax2.scatter(bucket[1,:], bucket[2,:], color=line_colors, alpha=1, s=50)
    ax2.scatter(x=mic_locs[:,1], y=mic_locs[:,2], facecolor='yellow', edgecolor='k')
    ax2.scatter(x=0, y=0, facecolor='yellow', edgecolor='k')

    ax3.plot(good_bucket[0,:], good_bucket[2,:], alpha=0.4, color='k')
    ax3.scatter(bucket[0,:], bucket[2,:], color=line_colors, alpha=1, s=50)
    ax3.scatter(x=mic_locs[:,0], y=mic_locs[:,2], facecolor='yellow', edgecolor='k')
    ax3.scatter(x=0, y=0, facecolor='yellow', edgecolor='k')

def get_d_mics_with_mic_locs(microphones_used, t_delay_wrt_selected_channel, selected_channel_for_ref):
    ind_of_selected_channel = np.where(microphones_used==(selected_channel_for_ref+1))[0]
    SELECTED_MIC_FOR_REF = microphones_used[ind_of_selected_channel]
    non_ref_microphones_used = microphones_used[microphones_used!=SELECTED_MIC_FOR_REF]
    t_delay_8_to_selected_channel = t_delay_wrt_selected_channel[:,ind_of_selected_channel]
    time_delay_mics_to_selected_ref_channel = t_delay_wrt_selected_channel - t_delay_8_to_selected_channel.reshape((len(t_delay_8_to_selected_channel), 1))
    time_delay_mics_to_selected_ref_channel_no_ref = np.delete(time_delay_mics_to_selected_ref_channel, ind_of_selected_channel, axis=1)
    d_mics_to_ref = time_delay_mics_to_selected_ref_channel_no_ref * C

    A_locs_mat_wrt_ref_channel = A_LOCS_MAT - A_LOCS_MAT[selected_channel_for_ref]
    A_locs_mat_wrt_ref_channel_only_mics_used = A_locs_mat_wrt_ref_channel[(non_ref_microphones_used-1)]

    return d_mics_to_ref, A_locs_mat_wrt_ref_channel_only_mics_used