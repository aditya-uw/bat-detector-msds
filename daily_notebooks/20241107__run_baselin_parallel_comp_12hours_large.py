from pathlib import Path
from torch import multiprocessing
import torch
from tqdm import tqdm
import warnings
import librosa
import os
import soundfile as sf

import numpy as np
import pandas as pd
import time
import io

import sys
import fsspec

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

    output_files = []
    fs = fsspec.filesystem('s3', anon=True, client_kwargs={'endpoint_url': 'https://sdsc.osn.xsede.org'})
    file = fs.open(path=package_to_chunk['audio_file'])
    if file.details['size']>0:
        ip_audio = sf.SoundFile(file)

        sampling_rate = ip_audio.samplerate
        # Convert to sampled units
        ip_start = int(package_to_chunk['start_time'] * sampling_rate)
        ip_duration = int(package_to_chunk['segment_duration'] * sampling_rate)
        ip_end = ip_audio.frames

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
            output_files.append({
                "input_filepath": package_to_chunk['audio_file'],
                "audio_file": op_path, 
                "offset":  package_to_chunk['start_time'] + (sub_start/sampling_rate),
            })
            
            if (not(op_path.exists())):
                sub_length = sub_end - sub_start
                ip_audio.seek(sub_start)
                op_audio = ip_audio.read(sub_length)
                sf.write(op_path, op_audio, sampling_rate, subtype='PCM_16')
        
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
    for package_to_chunk in tqdm(packages_to_chunk, desc="Segmenting Files"):
        segmented_file_paths += generate_segments(package_to_chunk)
    return segmented_file_paths

def load_audio_file(audio_file, time_exp_fact, target_samp_rate, scale=False, max_duration=False):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=wavfile.WavFileWarning)
        #sampling_rate, audio_raw = wavfile.read(audio_file)
        audio_raw, sampling_rate = librosa.load(audio_file, sr=None)
        # audio_raw = file_info['audio_seg']['audio_data']
        # sampling_rate = file_info['audio_seg']['audio_sampling_rate']

    if len(audio_raw.shape) > 1:
        raise Exception('Currently does not handle stereo files')
    sampling_rate = sampling_rate * time_exp_fact

    # resample - need to do this after correcting for time expansion
    sampling_rate_old = sampling_rate
    sampling_rate = target_samp_rate
    audio_raw = librosa.resample(audio_raw, orig_sr=sampling_rate_old, target_sr=sampling_rate, res_type='polyphase')

    # clipping maximum duration
    if max_duration is not False:
        max_duration = np.minimum(int(sampling_rate*max_duration), audio_raw.shape[0])
        audio_raw = audio_raw[:max_duration]
        
    # convert to float32 and scale
    audio_raw = audio_raw.astype(np.float32)
    if scale:
        audio_raw = audio_raw - audio_raw.mean()
        audio_raw = audio_raw / (np.abs(audio_raw).max() + 10e-6)

    return sampling_rate, audio_raw

def process_file(audio_file, model, params, args, time_exp=None, top_n=5, return_raw_preds=False, max_duration=False):

    # store temporary results here
    predictions = []
    spec_feats  = []
    cnn_feats   = []
    spec_slices = []

    # get time expansion  factor
    if time_exp is None:
        time_exp = args['time_expansion_factor']

    params['detection_threshold'] = args['detection_threshold']

    # load audio file
    sampling_rate, audio_full = load_audio_file(audio_file, time_exp,
                                   params['target_samp_rate'], params['scale_raw_audio'])

    # clipping maximum duration
    if max_duration is not False:
        max_duration = np.minimum(int(sampling_rate*max_duration), audio_full.shape[0])
        audio_full = audio_full[:max_duration]
    
    duration_full = audio_full.shape[0] / float(sampling_rate)

    return_np_spec = args['spec_features'] or args['spec_slices']

    # loop through larger file and split into chunks
    # TODO fix so that it overlaps correctly and takes care of duplicate detections at borders
    num_chunks = int(np.ceil(duration_full/args['chunk_size']))
    for chunk_id in range(num_chunks):

        # chunk
        chunk_time   = args['chunk_size']*chunk_id
        chunk_length = int(sampling_rate*args['chunk_size'])
        start_sample = chunk_id*chunk_length
        end_sample   = np.minimum((chunk_id+1)*chunk_length, audio_full.shape[0])
        audio = audio_full[start_sample:end_sample]

        # load audio file and compute spectrogram
        duration, spec, spec_np = du.compute_spectrogram(audio, sampling_rate, params, return_np_spec)

        # evaluate model
        with torch.no_grad():
            outputs = model(spec, return_feats=args['cnn_features'])

        # run non-max suppression
        pred_nms, features = pp.run_nms(outputs, params, np.array([float(sampling_rate)]))
        pred_nms = pred_nms[0]
        pred_nms['start_times'] += chunk_time
        pred_nms['end_times'] += chunk_time

        # if we have a background class
        if pred_nms['class_probs'].shape[0] > len(params['class_names']):
            pred_nms['class_probs'] = pred_nms['class_probs'][:-1, :]

        predictions.append(pred_nms)

        # extract features - if there are any calls detected
        if (pred_nms['det_probs'].shape[0] > 0):
            if args['spec_features']:
                spec_feats.append(feats.get_feats(spec_np, pred_nms, params))

            if args['cnn_features']:
                cnn_feats.append(features[0])

            if args['spec_slices']:
                spec_slices.extend(feats.extract_spec_slices(spec_np, pred_nms, params))

    # convert the predictions into output dictionary
    file_id = os.path.basename(audio_file)
    predictions, spec_feats, cnn_feats, spec_slices =\
              du.merge_results(predictions, spec_feats, cnn_feats, spec_slices)
    results = du.convert_results(file_id, time_exp, duration_full, params,
                              predictions, spec_feats, cnn_feats, spec_slices)

    # summarize results
    if not args['quiet']:
        num_detections = len(results['pred_dict']['annotation'])
        print('{}'.format(num_detections) + ' call(s) detected above the threshold.')

    # print results for top n classes
    if not args['quiet'] and (num_detections > 0):
        class_overall = pp.overall_class_pred(predictions['det_probs'], predictions['class_probs'])
        print('species name'.ljust(30) + 'probablity present')
        for cc in np.argsort(class_overall)[::-1][:top_n]:
            print(params['class_names'][cc].ljust(30) + str(round(class_overall[cc], 3)))

    if return_raw_preds:
        return predictions
    else:
        return results

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
        bd_offsetted = pipeline._correct_annotation_offsets(
                bd_annotations_df,
                cur_seg['original_file_name'],
                cur_seg['audio_seg']['offset']
            )
        bd_dets = pd.concat([bd_dets, bd_offsetted])
        
    return bd_dets

def load_model(model_path, load_weights=True):

    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if os.path.isfile(model_path):
        net_params = torch.load(model_path, map_location=device)
    else:
        print('Error: model not found.')
        sys.exit(1)

    params = net_params['params']
    params['device'] = device

    if params['model_name'] == 'Net2DFast':
        model = models.Net2DFast(params['num_filters'], num_classes=len(params['class_names']),
                                 emb_dim=params['emb_dim'], ip_height=params['ip_height'],
                                 resize_factor=params['resize_factor'])
    elif params['model_name'] == 'Net2DFastNoAttn':
        model = models.Net2DFastNoAttn(params['num_filters'], num_classes=len(params['class_names']),
                                 emb_dim=params['emb_dim'], ip_height=params['ip_height'],
                                 resize_factor=params['resize_factor'])
    elif params['model_name'] == 'Net2DFastNoCoordConv':
        model = models.Net2DFastNoCoordConv(params['num_filters'], num_classes=len(params['class_names']),
                                 emb_dim=params['emb_dim'], ip_height=params['ip_height'],
                                 resize_factor=params['resize_factor'])
    else:
        print('Error: unknown model.')

    if load_weights:
        model.load_state_dict(net_params['state_dict'])

    model = model.to(params['device'])
    model.eval()

    return model, params

def _run_batdetect(model_obj, audio_file): #
    """
    Parameters:: 
        audio_file: a path containing the post-processed wav file.

    Returns:: a pd.Dataframe containing the bat calls detections
    """
    model, params = load_model(model_obj.model_path)

    # Suppress output from this call
    text_trap = io.StringIO()
    sys.stdout = text_trap

    model_output = process_file(
        audio_file=audio_file, 
        model=model, 
        params=params, 
        args= {
            'detection_threshold': model_obj.detection_threshold,
            'spec_slices': model_obj.spec_slices,
            'chunk_size': model_obj.chunk_size,
            'quiet': model_obj.quiet,
            'spec_features' : False,
            'cnn_features': model_obj.cnn_features,
        },
        time_exp=model_obj.time_expansion_factor,
    )
    # Restore stdout
    sys.stdout = sys.__stdout__
    
    annotations = model_output['pred_dict']['annotation']

    out_df = gen_empty_df()
    if annotations:
        out_df = pd.DataFrame.from_records(annotations) 
        # out_df['detection_confidence'] = out_df['det_prob']
        # out_df.drop(columns = ['class', 'class_prob', 'det_prob','individual'], inplace=True)
    return out_df


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

    bd_dets = _run_batdetect(file_mapping['model'], file_mapping['audio_seg']['audio_file'])
    corrected_bd_dets = pipeline._correct_annotation_offsets(
                                                            bd_dets,
                                                            file_mapping['original_file_name'],
                                                            file_mapping['audio_seg']['offset']
                                                            )

    return corrected_bd_dets

def delete_segment(path):
    path['audio_file'].unlink(missing_ok=False)

def delete_segments(necessary_paths):
    """
    Deletes the segments whose paths are stored in necessary_paths

    Parameters
    ------------
    necessary_paths : `List`
        - A list of dictionaries generated from generate_segmented_paths()
    """

    for path in necessary_paths:
        delete_segment(path)

if __name__ == '__main__':
    ## grabbed from the rclone config file
    fs = fsspec.filesystem('s3', anon=True, client_kwargs={'endpoint_url': 'https://sdsc.osn.xsede.org'})
    tag = "bio230143-bucket01/ubna_data_01/recover-20220728/UBNA_007/2022*.WAV"
    group = fs.glob(path=tag)
    selected_wav_paths = list(map(Path, sorted(group)))

    cfg = get_config()
    cfg['tmp_dir'] = Path(f'{Path(__file__).parent}/../output')
    cfg['output_dir'] = Path(f'{Path(__file__).parent}/../output_dir')
    cfg['should_csv'] = False
    cfg['save'] = True

    if not cfg['output_dir'].is_dir():
        cfg['output_dir'].mkdir(parents=True, exist_ok=True)
    if not cfg['tmp_dir'].is_dir():
        cfg['tmp_dir'].mkdir(parents=True, exist_ok=True)

    print(f'generating segments in {cfg["tmp_dir"]}')
    packages_to_chunk = []
    for path in selected_wav_paths:
        chunk_instructions_and_files = dict()
        chunk_instructions_and_files['audio_file'] = path
        chunk_instructions_and_files['tmp_dir'] = cfg['tmp_dir']
        chunk_instructions_and_files['start_time'] = cfg['start_time']
        chunk_instructions_and_files['segment_duration'] = cfg['segment_duration']
        packages_to_chunk+=[chunk_instructions_and_files]

    baseline_sg_start = time.time()
    segmented_file_paths = []
    for package in tqdm(packages_to_chunk, desc="Segmenting Files"):
        segmented_file_paths+=[generate_segments(package)]
    segmented_file_paths = np.concatenate(list(segmented_file_paths))
    print(segmented_file_paths)
    file_path_mappings = batdetect2_pipeline.initialize_mappings(segmented_file_paths, cfg)
    baseline_rm_start = time.time()
    bd2_dets = run_models(file_path_mappings)
    delete_segments(segmented_file_paths)
    baseline_rm_end = time.time()
    print(f'Baseline segment-generation time: {baseline_rm_start-baseline_sg_start}')
    print(f'Baseline run models time: {baseline_rm_end-baseline_rm_start}')
    print(f'Baseline pipeline time: {baseline_rm_end-baseline_sg_start}')
    baseline_estimates = np.array([baseline_rm_start-baseline_sg_start, baseline_rm_end-baseline_rm_start, baseline_rm_end-baseline_sg_start])
    np.save(f'{Path(__file__).parent}/20241107__large_160gb_baseline_16p1c1t_estimates.npy', baseline_estimates)

    parallel_sg_start = time.time()
    num_processes = multiprocessing.cpu_count()
    torch.set_num_threads(1)
    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(processes=num_processes)
    segmented_file_paths = (tqdm(pool.imap(generate_segments, packages_to_chunk, chunksize=1), 
                    desc=f"Segmenting Files", total=len(packages_to_chunk),))
    segmented_file_paths = np.concatenate(list(segmented_file_paths))
    file_path_mappings = batdetect2_pipeline.initialize_mappings(segmented_file_paths, cfg)
    parallel_rm_start = time.time()
    print(f'Time taken to generate segments {parallel_rm_start-parallel_sg_start}')
    num_processes = multiprocessing.cpu_count()
    torch.set_num_threads(1)
    pool = multiprocessing.Pool(processes=num_processes)
    print(f'Parsing {len(file_path_mappings)} chunks with {num_processes} processors and chunksize 1 and {torch.get_num_threads()} threads per processor')
    results = tqdm(pool.imap(apply_model, file_path_mappings, chunksize=1), 
                    desc=f"Applying BatDetect2", total=len(file_path_mappings),)
    bd_preds = gen_empty_df() 
    bd_preds = pd.concat(results, ignore_index=True)
    num_processes = multiprocessing.cpu_count()
    torch.set_num_threads(1)
    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(processes=num_processes)
    segmented_file_paths = (tqdm(pool.imap(delete_segment, segmented_file_paths, chunksize=1), 
                    desc=f"Deleting Files", total=len(packages_to_chunk),))
    segmented_file_paths = list(segmented_file_paths)
    parallel_rm_end = time.time()
    print(f'Time taken to generate segments {parallel_rm_start-parallel_sg_start}')
    print(f'Parallel BatDetect2 time: {parallel_rm_end-parallel_rm_start}')
    print(f'Total pipeline time: {parallel_rm_end-parallel_sg_start}')
    parallel_estimates = np.array([parallel_rm_start-parallel_sg_start, parallel_rm_end-parallel_rm_start, parallel_rm_end-parallel_sg_start])
    np.save(f'{Path(__file__).parent}/20241107__large_160gb_parallel_16p1c1t_estimates.npy', parallel_estimates)