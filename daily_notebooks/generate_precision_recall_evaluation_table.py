import numpy as np
import pandas as pd

def return_confusion_matrix_from_comparing_two_detectors(human_df, machine_df, TP_CLASSIFICATION_THRESHOLD):
    dist_matrix = np.zeros((len(machine_df), len(human_df)))
    association_matrix = np.zeros((len(machine_df), len(human_df)), dtype='bool')
    for index in range(len(machine_df)):
        batdetect2_row = machine_df.iloc[index]
        dist_to_all_calls = ((human_df['peak_frequency_time'] - batdetect2_row['peak_frequency_time_SPECTROGRAM']).values)
        dist_matrix[index,:] = dist_to_all_calls
    
        dist_to_all_calls[np.abs(dist_to_all_calls)>=TP_CLASSIFICATION_THRESHOLD] = 1
        dist_to_all_calls[np.abs(dist_to_all_calls)<TP_CLASSIFICATION_THRESHOLD] = 0
        association_matrix[index,:] = ~dist_to_all_calls.astype('bool')

    human_bd2_true_positives = human_df.loc[np.logical_or.reduce(association_matrix, axis=0)]
    bd2_false_negatives = human_df.loc[~(np.logical_or.reduce(association_matrix, axis=0))]
    bd2_true_positives = machine_df.loc[np.logical_or.reduce(association_matrix, axis=1)]
    bd2_false_positives = machine_df.loc[~(np.logical_or.reduce(association_matrix, axis=1))]

    return {'true_positives':len(bd2_true_positives), 
            'false_positives':len(bd2_false_positives), 
            'false_negatives':len(bd2_false_negatives), 
            'true_negatives':0}

def get_precision_and_recall_from_metrics(true_positives, false_positives, false_negatives):
    denom_precision = (true_positives + false_positives)
    denom_recall = (true_positives + false_negatives)
    if (denom_precision>0):
        precision = true_positives / denom_precision
    else:
        precision = np.NaN
    if (denom_recall>0):
        recall = true_positives / denom_recall
    else:
        recall = np.NaN
    
    return precision, recall

def gather_evaluation_results_between_bd2_and_human(bd2_human_df, batdetect2_df_thresh, TP_CLASSIFICATION_THRESHOLD):
    file_batdetect2_cf = return_confusion_matrix_from_comparing_two_detectors(bd2_human_df, batdetect2_df_thresh, TP_CLASSIFICATION_THRESHOLD)
    precision, recall = get_precision_and_recall_from_metrics(file_batdetect2_cf['true_positives'], 
                                                              file_batdetect2_cf['false_positives'], 
                                                              file_batdetect2_cf['false_negatives'])

    return file_batdetect2_cf, precision, recall

def apply_SNR_threshold_on_both_sets(bd2_human_df, batdetect2_df, SNR_thresh):
    bd2_human_df_SNR = bd2_human_df[bd2_human_df['adityas_method_snr_dB']>=SNR_thresh].copy()
    batdetect2_df_SNR = batdetect2_df[batdetect2_df['SNR']>=SNR_thresh].copy()

    return bd2_human_df_SNR, batdetect2_df_SNR

def make_new_row_in_eval_df(file_batdetect2_cf, precision, recall):
    row = pd.DataFrame([file_batdetect2_cf])
    row['precision'] = precision
    row['recall'] = recall

    return row

def generate_evaluation_df(bd2_human_df, batdetect2_df, TP_CLASSIFICATION_THRESHOLD):
    batdetect2_eval = pd.DataFrame()

    snr_increment = 1.0
    snr_thresholds = np.arange(0, 20, snr_increment)
    for snr_thresh in snr_thresholds:
        bd2_human_df_SNR, batdetect2_df_SNR = apply_SNR_threshold_on_both_sets(bd2_human_df, batdetect2_df, snr_thresh)

        detthresh_increment = 0.01
        detection_thresholds = np.arange(0.01, 0.6+detthresh_increment, detthresh_increment)
        for det_thresh in detection_thresholds:
            det_thresh = round(det_thresh, 2)
            print(snr_thresh, det_thresh)
            batdetect2_df_SNR_detthresh = batdetect2_df_SNR[batdetect2_df_SNR['det_prob']>=det_thresh].copy()
            file_batdetect2_cf, precision, recall = gather_evaluation_results_between_bd2_and_human(bd2_human_df_SNR, 
                                                                                                    batdetect2_df_SNR_detthresh,
                                                                                                    TP_CLASSIFICATION_THRESHOLD)
            row = make_new_row_in_eval_df(file_batdetect2_cf, precision, recall)
            row.insert(0, 'num_human_annotations', [len(bd2_human_df_SNR)])
            row.insert(0, 'num_bd2_detections', [len(batdetect2_df_SNR_detthresh)])
            row.insert(0, 'detection_threshold', [det_thresh])
            row.insert(0, 'SNR_threshold', [snr_thresh])
            batdetect2_eval = pd.concat([batdetect2_eval, row])
            assert(len(batdetect2_df_SNR_detthresh['freq_group'].unique())<=2)

    batdetect2_eval.reset_index(drop=True, inplace=True)
    return batdetect2_eval