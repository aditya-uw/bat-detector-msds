import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy

import time
from pathlib import Path

import soundfile as sf
import torch
from tqdm import tqdm
from torch import multiprocessing

import sys

# append the path of the
# parent directory
sys.path.append(f'{Path(__file__).parent}/..')
sys.path.append(f'{Path(__file__).parent}/../src/')

import batdt2_pipeline
from pipeline import pipeline
from models.bat_call_detector.model_detector import BatCallDetector
from utils.utils import gen_empty_df


LABEL_FOR_GROUPS = {
                    0: 'LF', 
                    1: 'HF'
                    }

def convert_bd2df_ravenpro(df: pd.DataFrame):
    """
    Converts a dataframe to the format used by RavenPro
    """

    ravenpro_df = df.copy()

    ravenpro_df.rename(columns={
        "start_time": "Begin Time (s)",
        "end_time": "End Time (s)",
        "low_freq": "Low Freq (Hz)",
        "high_freq": "High Freq (Hz)",
        "event": "Annotation",
        "freq_group":"Manually-Verified Phonic Group",
        'snr_nist_quick_dB':'SNR NIST Quick (dB)',
        'delta_time_s':'Delta Time (s)'
    }, inplace=True)

    ravenpro_df["Selection"] = np.arange(0, df.shape[0]).astype('int') + 1
    ravenpro_df["View"] = "Waveform 1"
    ravenpro_df["Channel"] = "1"

    return ravenpro_df

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
        audio_seg, length_of_section, pad = get_section_of_call_in_file(row, audio_file)
        
        freq_pad = 2000
        low_freq_cutoff = row['low_freq']-freq_pad
        high_freq_cutoff = min(nyquist-1, row['high_freq']+freq_pad)
        band_limited_audio_seg = batdt2_pipeline.bandpass_audio_signal(audio_seg, fs, low_freq_cutoff, high_freq_cutoff)

        signal_for_peaks = band_limited_audio_seg.copy()
        signal_for_peaks[:int(fs*(length_of_section+pad))] = 0
        signal_for_peaks[-int(fs*pad):] = 0
        mpl_specgram_window = plt.mlab.window_hanning(np.ones(256))
        f, t, Sxx = scipy.signal.spectrogram(signal_for_peaks, fs, detrend=False,
                                    nfft=256, 
                                    window=mpl_specgram_window)
        plt_Sxx = 10*np.log10(Sxx)
        max_ind = np.where(plt_Sxx==np.max(plt_Sxx))
        max_value_across_bins = Sxx[max_ind]
        peak_freq = f[max_ind[0]]
        peak_freq_time = t[max_ind[1]]
        features_of_interest['peak_freqs_spec'].append(peak_freq)
        features_of_interest['peak_freq_times_spec'].append(peak_freq_time)

        signal = band_limited_audio_seg.copy()
        signal[:int(fs*(length_of_section))] = 0
        noise = band_limited_audio_seg - signal
        snr_call_signal = signal[-int(fs*length_of_section):]
        snr_noise_signal = noise[:int(fs*length_of_section)]
        features_of_interest['call_signals'].append(snr_call_signal)

        snr = batdt2_pipeline.get_snr_from_band_limited_signal(snr_call_signal, snr_noise_signal)
        features_of_interest['snrs'].append(snr)

        welch_info = dict()
        welch_info['num_points'] = 100
        max_visible_frequency = 96000
        welch_info['max_freq_visible'] = max_visible_frequency
        welch_signal = batdt2_pipeline.compute_welch_psd_of_call(snr_call_signal, fs, welch_info)
        features_of_interest['welch_signals'].append(welch_signal)

        peaks = np.where(welch_signal==max(welch_signal))[0][0]
        features_of_interest['peak_freqs_welch'].append(max_visible_frequency*(peaks/len(welch_signal)))
        
        welch_signal = (welch_signal).reshape(1, len(welch_signal))
        features_of_interest['classes'].append(kmean_welch.predict(welch_signal)[0])

    features_of_interest['call_signals'] = np.array(features_of_interest['call_signals'], dtype='object')

    return features_of_interest

def open_and_get_call_info(audio_file, dets):
    welch_key = 'all_locations'
    output_dir = Path(f'{Path(__file__).parent}/../../duty-cycle-investigation/data/generated_welch/{welch_key}')
    output_file_type = 'top1_inbouts_welch_signals'
    welch_data = pd.read_csv(output_dir / f'2022_{welch_key}_{output_file_type}.csv', index_col=0, low_memory=False)
    k = 2
    kmean_welch = KMeans(n_clusters=k, n_init=10, random_state=1).fit(welch_data.values)

    features_of_interest = gather_features_of_interest(dets, kmean_welch, audio_file)

    dets.reset_index(drop=True, inplace=True)

    dets['sampling_rate'] = len(dets) * [audio_file.samplerate]
    dets.insert(0, 'SNR', features_of_interest['snrs'])
    dets.insert(0, 'peak_frequency_WELCH', features_of_interest['peak_freqs_welch'])
    dets.insert(0, 'peak_frequency_SPECTROGRAM', features_of_interest['peak_freqs_spec'])
    dets.insert(0, 'peak_frequency_time_SPECTROGRAM', features_of_interest['peak_freq_times_spec'])
    dets.insert(0, 'KMEANS_CLASSES', pd.Series(features_of_interest['classes']).map(LABEL_FOR_GROUPS))

    return features_of_interest['call_signals'], dets

def classify_calls_from_file(bd2_predictions, data_params):
    file_path = Path(data_params['audio_file'])
    audio_file = sf.SoundFile(file_path)
    call_signals, dets = open_and_get_call_info(audio_file, bd2_predictions)
    return dets

def apply_model(file_mapping):
    """
    Runs the batdetect2 model on a single provided audio segmens and corrects the offsets according the segment.

    Parameters
    ------------
    file_mappings : `List`
        - List of dictionaries generated by initialize_mappings()

    Returns
    ------------
    corrected_bd_dets : `pandas.DataFrame`
        - A DataFrame of detections that will also be saved in the provided output_dir under the above csv_name
        - 7 columns in this DataFrame: start_time, end_time, low_freq, high_freq, detection_confidence, event, input_file
        - Detections are always specified w.r.t their input_file; earliest start_time can be 0 and latest end_time can be 1795.
        - Events are always "Echolocation" as we are using a model that only detects search-phase calls.
    """

    bd_dets = file_mapping['model']._run_batdetect(file_mapping['audio_seg']['audio_file'])
    bd_preds_classed = classify_calls_from_file(bd_dets, file_mapping['audio_seg'])
    corrected_bd_dets = pipeline._correct_annotation_offsets(
                                                            bd_preds_classed,
                                                            file_mapping['original_file_name'],
                                                            file_mapping['audio_seg']['offset']
                                                            )

    return corrected_bd_dets

def apply_models(file_path_mappings, cfg):
    """
    Runs the batdetect2 model to detect bat search-phase calls in the provided audio segments and saves detections into a dataframe

    Parameters
    ------------
    file_mappings : `List`
        - List of dictionaries generated by initialize_mappings()
    cfg : `dict`
        - A dictionary of pipeline parameters:
        - models is the models in the pipeline that are being used.

    Returns
    ------------
    bd_preds : `pandas.DataFrame`
        - A DataFrame of detections that will also be saved in the provided output_dir under the above csv_name
        - 7 columns in this DataFrame: start_time, end_time, low_freq, high_freq, detection_confidence, event, input_file
        - Detections are always specified w.r.t their input_file; earliest start_time can be 0 and latest end_time can be 1795.
        - Events are always "Echolocation" as we are using a model that only detects search-phase calls.
    """

    torch.set_num_threads(1)
    process_pool = multiprocessing.Pool(cfg['num_processes'])

    bd_dets = tqdm(
            process_pool.imap(apply_model, file_path_mappings, chunksize=1), 
            desc=f"Applying BatDetect2",
            total=len(file_path_mappings),
        )
    
    bd_preds = gen_empty_df() 
    bd_preds = pd.concat(bd_dets, ignore_index=True)
    return bd_preds

def delete_segment(path):
    path['audio_file'].unlink(missing_ok=False)

if __name__ == '__main__':
    file_sites = {'20220730_053000':'Carp',
                '20220826_070000':'Central',
                '20220727_083000':'Foliage',
                '20220829_090000':'Foliage',
                '20210910_030000':'Foliage'}
    file_keys = list(file_sites.keys())
    save_dir = Path(f'{Path(__file__).parent}/20250114__group_threshold_sweep_results')
    save_dir.mkdir(parents=True, exist_ok=True)
    for wav_filename in file_keys:
        site = file_sites[wav_filename]
        file_path = Path(f'{Path.home()}/Documents/mila-human-wav-txt/{wav_filename}.WAV')

        packages_to_chunk = []
        chunk_instructions_and_files = dict()
        chunk_instructions_and_files['audio_file'] = file_path
        chunk_instructions_and_files['tmp_dir'] = save_dir
        chunk_instructions_and_files['start_time'] = 0.0
        chunk_instructions_and_files['segment_duration'] = 30.0
        packages_to_chunk+=[chunk_instructions_and_files]

        cfg=dict()
        cfg['num_processes'] = multiprocessing.cpu_count()
        parallel_sg_start = time.time()
        torch.set_num_threads(1)
        ctx = multiprocessing.get_context("spawn")
        pool = ctx.Pool(processes=cfg['num_processes'])
        segmented_file_paths = (tqdm(pool.imap(batdt2_pipeline.generate_segments_parallel, packages_to_chunk, chunksize=1), 
                        desc=f"Segmenting Files", total=len(packages_to_chunk),))
        segmented_file_paths = np.concatenate(list(segmented_file_paths))
        parallel_sg_end = time.time()

        args = dict()
        args['detection_threshold'] = 0.00
        args['chunk_size'] = 2

        cfg["time_expansion_factor"] = 1.0
        # Offset (seconds) from the beginning of the audio file to start processing
        cfg["start_time"] = 0.0
        # Input audio is divided into segments of this duration (seconds), each processed individually
        cfg["segment_duration"] = 30.0
        cfg["models"] = [BatCallDetector(detection_threshold=args['detection_threshold'],
                                        spec_slices=False,
                                        chunk_size=args['chunk_size'],
                                        time_expansion_factor=1.0,
                                        quiet=False,
                                        cnn_features=True)]
        
        file_path_mappings = batdt2_pipeline.initialize_mappings(segmented_file_paths, cfg)
        parallel_rm_start = time.time()
        bd_preds = apply_models(file_path_mappings, cfg)
        parallel_rm_end = time.time()

        print(f'Time taken to generate segments {parallel_sg_end-parallel_sg_start}')
        print(f'Parallel BatDetect2 time: {parallel_rm_end-parallel_rm_start}')
        print(f'Total pipeline time: {parallel_rm_end-parallel_sg_start}')

        ones = int(args['detection_threshold'])
        decimals = int(round(100*(args['detection_threshold']), 1) % 100)
        threshold_tag = f"threshold{ones}p{decimals:02}"
        save_loc_tag = f"{threshold_tag}_chunksize{int(args['chunk_size'])}_{wav_filename}"
        bd2_save_loc = Path(f"bd2__{save_loc_tag}.csv")
        rpro_save_loc = Path(f"rpro__{save_loc_tag}.txt")
        print("saving to", (save_dir / bd2_save_loc))
        bd_preds.to_csv(save_dir / bd2_save_loc)

        ravenpro_df = convert_bd2df_ravenpro(bd_preds)
        ravenpro_df.to_csv(save_dir / rpro_save_loc, sep="\t")

        parallel_del_start = time.time()
        torch.set_num_threads(1)
        ctx = multiprocessing.get_context("spawn")
        pool = ctx.Pool(processes=cfg['num_processes'])
        segmented_file_paths = (tqdm(pool.imap(delete_segment, segmented_file_paths, chunksize=1), 
                        desc=f"Deleting Files", total=len(packages_to_chunk),))
        segmented_file_paths = list(segmented_file_paths)
        parallel_del_end = time.time()


