import torch
from bpnet_customize.io_ import PeakGenerator
from bpnet_customize.io_ import extract_loci
from bpnet_customize.bpnet import BPNet
import numpy as np
import merge

# Predict parameters
default_predict_parameters = {
    'batch_size': 64,
    'in_window': 2114,
    'out_window': 1000,
    'verbose': False,
    'chroms': ['chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 
        'chr9', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 
        'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX'],
    'sequences': None,
    'loci': None,
    'controls': None,
    'model': None,
    'profile_filename': 'y_profile.npz',
    'counts_filename': 'y_counts.npz'
}

predict_para = merge.merge_parameters("../json/predict.json", default_predict_parameters)

# Predict
model = torch.load(predict_para['model']).cuda()

examples = extract_loci(
    sequences=predict_para['sequences'],
    controls=predict_para['controls'],
    loci=predict_para['loci'],
    chroms=predict_para['chroms'],
    max_jitter=0,
    verbose=predict_para['verbose']
)

# valid_data = extract_loci(
#     sequences=predict_para['sequences'],
#     signals=predict_para['signals'],
#     controls=predict_para['controls'],
#     loci=predict_para['loci'],
#     chroms=predict_para['chroms'],
#     in_window=predict_para['in_window'],
#     out_window=predict_para['out_window'],
#     max_jitter=0,
#     verbose=predict_para['verbose']
# )

if predict_para['controls'] == None:
    X = examples
    if model.n_control_tracks > 0:
        X_ctl = torch.zeros(X.shape[0], model.n_control_tracks, X.shape[-1])
    else:
        X_ctl = None
else:
    X, X_ctl, valid_data = examples

y_profiles, y_counts = model.predict(X, X_ctl=X_ctl, valid_data=valid_data,
                                      batch_size=predict_para['batch_size'])

np.savez_compressed(predict_para['profile_filename'], y_profiles)
np.savez_compressed(predict_para['counts_filename'], y_counts)
