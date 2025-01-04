import os
import librosa
import pandas as pd
import torch
import io
import sys

from batdetect2 import api
import batdetect2.detector.compute_features as feats
import batdetect2.utils.detector_utils as du
import batdetect2.utils.audio_utils as au
from models.detection_interface import DetectionInterface
from batdetect2.types import (
    DetectionModel,
    ProcessingConfiguration
)
from utils.utils import gen_empty_df

# print(sys.path)
import batdetect2.api as api

class BatCallDetector(DetectionInterface):
    """
    A class containing the bat detect model and feeding buzz model. The parameters of this class are explained in cfg.py 
    """
    def __init__(self, detection_threshold, spec_slices, chunk_size, time_expansion_factor, quiet, cnn_features):
        self.detection_threshold = detection_threshold
        self.spec_slices = spec_slices
        self.chunk_size = chunk_size
        self.time_expansion_factor = time_expansion_factor
        self.quiet = quiet
        self.cnn_features = cnn_features
        

    def get_name(self):
        return "BatDetectorMSDS"

    def _run_batdetect(self, audio_file)-> pd.DataFrame: #
        """
        Parameters:: 
            audio_file: a path containing the post-processed wav file.

        Returns:: a pd.Dataframe containing the bat calls detections
        """
        config = api.get_config(detection_threshold=self.detection_threshold,
                                spec_slices = self.spec_slices,
                                chunk_size = self.chunk_size,
                                time_expansion_factor = self.time_expansion_factor,
                                quiet = self.quiet,
                                cnn_features = self.cnn_features)
        model = api.MODEL
        device = api.DEVICE

        # Suppress output from this call
        text_trap = io.StringIO()
        sys.stdout = text_trap

        # store temporary results here
        predictions = []
        spec_feats = []
        cnn_feats = []
        spec_slices = []

        # Get original sampling rate
        file_samp_rate = librosa.get_samplerate(audio_file)
        orig_samp_rate = file_samp_rate * (config.get("time_expansion") or 1)

        # load audio file
        sampling_rate, audio_full = au.load_audio(
            audio_file,
            time_exp_fact=config.get("time_expansion", 1) or 1,
            target_samp_rate=config["target_samp_rate"],
            scale=config["scale_raw_audio"],
            max_duration=config.get("max_duration"),
        )

        # loop through larger file and split into chunks
        # TODO: fix so that it overlaps correctly and takes care of
        # duplicate detections at borders
        for chunk_time, audio in du.iterate_over_chunks(
            audio_full,
            sampling_rate,
            config["chunk_size"],
        ):
            # Run detection model on chunk
            pred_nms, features, spec = du._process_audio_array(
                audio,
                sampling_rate,
                model,
                config,
                device,
            )
            num_rawdets = pred_nms['start_times'].shape[0]

            raw_dets = pd.DataFrame()
            for key in pred_nms.keys(): 
                if key != 'class_probs':                 
                    raw_dets[key] = pred_nms[key]

            class_probs = []
            for i in range(num_rawdets):
                class_probs_for_det = pred_nms['class_probs'][:,i]
                class_probs.append(class_probs_for_det)
            raw_dets['class_probs'] = class_probs
            raw_dets['chunk_time'] = [chunk_time]*len(raw_dets)
            inscope_rawdets = raw_dets[raw_dets['end_times']<=audio.shape[0]/sampling_rate]
            outofscopes_rawdets = raw_dets[raw_dets['end_times']>=audio.shape[0]/sampling_rate]

            inscope_features = features[inscope_rawdets.index,:]
            inscope_pred_nms = dict()
            for key in pred_nms.keys():
                if key == 'class_probs':
                    inscope_pred_nms[key] = pred_nms[key][:,inscope_rawdets.index]
                else:
                    inscope_pred_nms[key] = pred_nms[key][inscope_rawdets.index]
            # convert to numpy
            spec_np = spec.detach().cpu().numpy().squeeze()

            # add chunk time to start and end times
            inscope_pred_nms["start_times"] += chunk_time
            inscope_pred_nms["end_times"] += chunk_time

            predictions.append(inscope_pred_nms)

            # extract features - if there are any calls detected
            if inscope_pred_nms["det_probs"].shape[0] == 0:
                continue

            if config["spec_features"]:
                spec_feats.append(feats.get_feats(spec_np, inscope_pred_nms, config))

            if config["cnn_features"]:
                cnn_feats.append(inscope_features[0])

            if config["spec_slices"]:
                # FIX: This is not currently working. Returns empty slices
                spec_slices.extend(feats.extract_spec_slices(spec_np, inscope_pred_nms))

        # Merge results from chunks
        predictions, spec_feats, cnn_feats, spec_slices = du._merge_results(
            predictions,
            spec_feats,
            cnn_feats,
            spec_slices,
        )

        # convert results to a dictionary in the right format
        model_output = du.convert_results(
            file_id=os.path.basename(audio_file),
            time_exp=config.get("time_expansion", 1) or 1,
            duration=audio_full.shape[0] / float(sampling_rate),
            params=config,
            predictions=predictions,
            spec_feats=spec_feats,
            cnn_feats=cnn_feats,
            spec_slices=spec_slices,
            nyquist_freq=orig_samp_rate / 2,
        )

        # summarize results
        if not config["quiet"]:
            du.summarize_results(model_output, predictions, config)
        
        # Restore stdout
        sys.stdout = sys.__stdout__

        annotations = model_output['pred_dict']['annotation']

        out_df = gen_empty_df()
        if annotations:
            out_df = pd.DataFrame.from_records(annotations) 
        return out_df
    
    # def _run_feedbuzz(self, audio_file) -> pd.DataFrame: # TODO: type annotations
    #     """
    #      Parameters:: 
    #         audio_file: a path containing the post-processed wav file.

    #     Returns:: a pd.Dataframe containing the feeding buzz detections
    #     """
    #     out_df = gen_empty_df()
    #     template_dict = fbh.load_templates(self.template_dict_path)
    #     out_df = fbh.run_multiple_template_matching(
    #                                         PATH_AUDIO=audio_file,
    #                                         out_df=out_df,
    #                                         peak_distance=self.peak_distance, #self.peak_distance is a tuple for some reason.
    #                                         peak_th=self.peak_th,
    #                                         template_dict=template_dict,
    #                                         num_matches_threshold=self.num_matches_threshold, 
    #                                         buzz_feed_range=self.buzz_feed_range, 
    #                                         alpha=self.alpha)
        
    #     # A flag for end user to differentiate between feeding buzz and bat calls.
    #     out_df['event'] = 'Feeding Buzz'
    #     return out_df
    

    # def _removing_collision(self,curr_row:tuple, compare_df:pd.DataFrame): 
    #     """
    #     Remove collision between feeding buzz false positive and bat calls true positive values.
    #     Parameters::
    #         curr_row: tuple
    #         The tuple with columns start_time, end_time,low_freq,high_freq

    #         compare_df: pd.DataFrame
    #         The dataframe that contains bat calls true positive values

    #     Return:: a boolean
    #     """
    #     # TODO: Decide if bounding box interect is a good idea (might remove TP), maybe better to compare in center
    #     XB1 = curr_row.start_time
    #     XB2 = curr_row.end_time
    #     YB1 = curr_row.low_freq
    #     YB2 = curr_row.high_freq
    #     SB = (XB2 - XB1) * (YB2 - YB1)
    
    #     for i in compare_df.itertuples():
    #         XA1 = i.start_time #min_t
    #         XA2 = i.end_time #max_t
    #         YA1 = i.low_freq #min_f
    #         YA2 = i.high_freq #max_f

    #         if (XB2 >= XA2 and XA1 >= XB1 and YB2 >= YA2 and YA1 >= YB1 ):
    #             return 1
    #     return 0
    
        

    # def _buzzfeed_fp_removal(self,bd_output:pd.DataFrame, fb_output:pd.DataFrame)-> pd.DataFrame:
    #     """
    #     Creates a loop for feeding buzz to remove false positive.
    #     Parameters::
    #         bd_output: pd.DataFrame
    #             DataFrame containing bat calls true positive values, result from Bat Detect pipeline.

    #         fb_output: pd.DataFrame
    #             DataFrame containing feeding buzz detections, result from Template Matching pipeline.

    #     Return: pd.DataFrame
    #     """
    #     collide = np.zeros(len(fb_output))
    #     for curr in fb_output.itertuples():
    #         collide[curr.Index] = self._removing_collision(curr,bd_output)
        
    #     fb_output['Collide'] = collide
    #     fb_df_filtered = fb_output[fb_output['Collide']== 0] 
    #     del fb_df_filtered['Collide']
        
    #     return fb_df_filtered
    
    # def run(self, audio_file):
    #     """
    #     Creates a loop for feeding buzz to remove false positive.
    #     Parameters::
    #         bd_output: pd.DataFrame
    #             DataFrame containing bat calls true positive values, result from Bat Detect pipeline.

    #         fb_output: pd.DataFrame
    #             DataFrame containing feeding buzz detections, result from Template Matching pipeline.
                
    #     Return: pd.DataFrame
    #     """
    #     bd_output = self._run_batdetect(audio_file)
    #     fb_output = self._run_feedbuzz(audio_file)
    #     fb_final_output = self._buzzfeed_fp_removal(bd_output, fb_output)

    #     return pd.concat([bd_output,fb_final_output])
    