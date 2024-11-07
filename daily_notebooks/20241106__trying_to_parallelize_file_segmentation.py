from pathlib import Path
from torch import multiprocessing
import warnings
import torch
import librosa
import os
import fsspec

from tqdm import tqdm
import soundfile as sf

import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
import datetime as dt

import sys

# append the path of the
# parent directory
sys.path.append('..')
sys.path.append(f'{Path(__file__).parent}/../src/')
sys.path.append(f'{Path(__file__).parent}/../src/models/bat_call_detector/batdetect2/')

import batdt2_pipeline as batdetect2_pipeline
from pipeline import pipeline
from utils.utils import gen_empty_df
from cfg import get_config
from bat_detect.detector import models
import bat_detect.detector.compute_features as feats
import bat_detect.detector.post_process as pp
import bat_detect.utils.audio_utils as au
import bat_detect.utils.detector_utils as du
import bat_detect.utils.wavfile as wavfile

def generate_segments(package_to_chunk):
    """
    Segments audio file into clips of duration length and saves them to output/tmp folder.
    Allows detection model to be run on segments instead of entire file as recommended.
    These segments will be deleted from the output/tmp folder after detections have been generated.

    Parameters
    ------------
    audio_file : `pathlib.Path`
        - The path to an audio_file from the input directory provided in the command line
    output_dir : `pathlib.Path`
        - The path to the tmp folder that saves all of our segments.
    start_time : `float`
        - The time at which the segments will start being generated from within the audio file
    duration : `float`
        - The duration of all segments generated from the audio file.

    Returns
    ------------
    output_files : `List`
        - The path (a str) to each generated segment of the given audio file will be stored in this list.
        - The offset of each generated segment of the given audio file will be stored in this list.
        - Both items are stored in a dict{} for each generated segment.
    """
    
    fs = fsspec.filesystem('s3', anon=True, client_kwargs={'endpoint_url': 'https://sdsc.osn.xsede.org'})
    file = fs.open(path=package_to_chunk['audio_file'])
    ip_audio = sf.SoundFile(file)

    sampling_rate = ip_audio.samplerate
    # Convert to sampled units
    ip_start = int(package_to_chunk['start_time'] * sampling_rate)
    ip_duration = int(package_to_chunk['segment_duration'] * sampling_rate)
    ip_end = ip_audio.frames

    output_files = []

    # for the length of the duration, process the audio into duration length clips
    for sub_start in range(ip_start, ip_end, ip_duration):
        sub_end = np.minimum(sub_start + ip_duration, ip_end)

        # For file names, convert back to seconds 
        op_file = package_to_chunk['audio_file'].name.replace(" ", "_")
        start_seconds =  sub_start / sampling_rate
        end_seconds =  sub_end / sampling_rate
        op_file_en = "__{:.2f}".format(start_seconds) + "_" + "{:.2f}".format(end_seconds)
        op_file = op_file[:-4] + op_file_en + ".wav"
        
        op_path = package_to_chunk['tmp_dir'] / op_file
        
        sub_length = ip_duration
        ip_audio.seek(sub_start)
        op_audio = ip_audio.read(sub_length, dtype='float32')
        output_files.append({
            "input_filepath": package_to_chunk['audio_file'],
            "audio_file": op_path,
            "audio_data":op_audio,
            "audio_sampling_rate":sampling_rate,
            "offset":  package_to_chunk['start_time'] + (sub_start/sampling_rate),
        })
        
    return output_files 

def generate_segmented_paths(packages_to_chunk):
    """
    Generates and returns a list of segments using provided cfg parameters for each audio file in audio_files.

    Parameters
    ------------
    audio_files : `List`
        - List of pathlib.Path objects of the paths to each audio file in the provided input directory.
    cfg : `dict`
        - A dictionary of pipeline parameters:
        - tmp_dir is the directory where segments will be stored
        - start_time is the time at which segments are generated from each audio file.
        - segment_duration is the duration of each generated segment

    Returns
    ------------
    segmented_file_paths : `List`
        - A list of dictionaries related to every generated segment.
        - Each dictionary stores a generated segment's path in the tmp_dir and offset in the original audio file.
    """

    segmented_file_paths = []
    for package_to_chunk in packages_to_chunk:
        segmented_file_paths += generate_segments(package_to_chunk)
    return segmented_file_paths

if __name__ == '__main__':
    fs = fsspec.filesystem('s3', anon=True, client_kwargs={'endpoint_url': 'https://sdsc.osn.xsede.org'})

    tag1 = "bio230143-bucket01/ubna_data_01/recover-20220728/UBNA_007/2022*_0[2|3|4|5|6|7|8|9]*00.WAV"
    group1 = fs.glob(path=tag1)
    tag2 = "bio230143-bucket01/ubna_data_01/recover-20220728/UBNA_007/2022*_1[0|1|2|3]*00.WAV"
    group2 = fs.glob(path=tag2)
    selected_files = sorted(group1+group2)
    selected_wav_paths = list(map(Path, selected_files))

    cfg = get_config()
    cfg['tmp_dir'] = Path(f'../output')
    cfg['output_dir'] = Path(f'../output_dir')
    cfg['should_csv'] = False
    cfg['save'] = True

    if not cfg['output_dir'].is_dir():
        cfg['output_dir'].mkdir(parents=True, exist_ok=True)
    if not cfg['tmp_dir'].is_dir():
        cfg['tmp_dir'].mkdir(parents=True, exist_ok=True)

    packages_to_chunk = []
    for path in selected_wav_paths:
        chunk_instructions_and_files = dict()
        chunk_instructions_and_files['audio_file'] = path
        chunk_instructions_and_files['tmp_dir'] = cfg['tmp_dir']

        chunk_instructions_and_files['start_time'] = cfg['start_time']
        chunk_instructions_and_files['segment_duration'] = cfg['segment_duration']
        packages_to_chunk+=[chunk_instructions_and_files]

    num_processes = 8
    torch.set_num_threads(8)
    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(processes=num_processes)

    segmented_file_paths = (tqdm(pool.imap(generate_segments, packages_to_chunk, chunksize=1), 
                    desc=f"Segmenting Files", total=len(packages_to_chunk),))
    segmented_file_paths = list(segmented_file_paths)
    print(segmented_file_paths)