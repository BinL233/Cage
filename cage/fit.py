import torch
import numpy as np
import merge

from bpnet_customize.io_ import PeakGenerator
from bpnet_customize.io_ import extract_loci
from bpnet_customize.bpnet import BPNet

default_fit_parameters = {
    'n_filters': 64,
    'n_layers': 8,
    'profile_output_bias': True,
    'count_output_bias': True,
    'name': None,
    'batch_size': 64,
    'in_window': 2114,
    'out_window': 1000,
    'max_jitter': 128,
    'reverse_complement': True,
    'max_epochs': 50,
    'validation_iter': 100,
    'lr': 0.001,
    'alpha': 1,
    'verbose': False,

    'min_counts': 0,
    'max_counts': 99999999,

    'training_chroms': ['chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 
        'chr9', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 
        'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX'],
    'validation_chroms': ['chr8', 'chr10'],
    'sequences': None,
    'loci': None,
    'signals': None,
    'controls': None,
    'random_state': None
    }

# Fit parameter

fit_para = merge.merge_parameters("../json/fit.json", default_fit_parameters)

# Fit

training_data = PeakGenerator(
    loci=fit_para['loci'], 
    sequences=fit_para['sequences'],
    signals=fit_para['signals'],
    controls=fit_para['controls'],
    chroms=fit_para['training_chroms'],
    in_window=fit_para['in_window'],
    out_window=fit_para['out_window'],
    max_jitter=fit_para['max_jitter'],
    reverse_complement=fit_para['reverse_complement'],
    min_counts=fit_para['min_counts'],
    max_counts=fit_para['max_counts'],
    random_state=fit_para['random_state'],
    batch_size=fit_para['batch_size'],
    verbose=fit_para['verbose']
)

valid_data = extract_loci(
    sequences=fit_para['sequences'],
    signals=fit_para['signals'],
    controls=fit_para['controls'],
    loci=fit_para['loci'],
    chroms=fit_para['validation_chroms'],
    in_window=fit_para['in_window'],
    out_window=fit_para['out_window'],
    max_jitter=0,
    verbose=fit_para['verbose']
)

if fit_para['controls'] is not None:
    valid_sequences, valid_signals, valid_controls = valid_data
    n_control_tracks = 2
else:
    valid_sequences, valid_signals = valid_data
    valid_controls = None
    n_control_tracks = 0
    
trimming = (fit_para['in_window'] - fit_para['out_window']) // 2

model = BPNet(n_filters=fit_para['n_filters'], 
    n_layers=fit_para['n_layers'],
    n_outputs=len(fit_para['signals']),
    n_control_tracks=n_control_tracks,
    profile_output_bias=fit_para['profile_output_bias'],
    count_output_bias=fit_para['count_output_bias'],
    alpha=fit_para['alpha'],
    trimming=trimming,
    name=fit_para['name'],
    verbose=fit_para['verbose']).cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=fit_para['lr'])

if fit_para['verbose']:
    print("Training Set Size: ", training_data.dataset.sequences.shape[0])
    print("Validation Set Size: ", valid_sequences.shape[0])

model.fit(training_data, optimizer, X_valid=valid_sequences, 
    X_ctl_valid=valid_controls, y_valid=valid_signals, 
    max_epochs=fit_para['max_epochs'], 
    validation_iter=fit_para['validation_iter'], 
    batch_size=fit_para['batch_size'])