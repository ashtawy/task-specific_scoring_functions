#!/usr/bin/env python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import argparse
from sklearn import metrics
import pandas as pd
import numpy as np
import timeit
from read_plc_data import read_plc_data
from model_utils import train_test_model
import os
import sys
import multiprocessing
#-------------------------------------------------------------------------------
def parse_and_process_args():
    description = """ Train and test task-specific SFs"""
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--sfname', required=True,
        choices = ['BT-Score', 'BT-Dock', 'BT-Screen',
                   'RF-Score', 'X-Score'], 
        help = """Enter the name of the scoring function you would like 
                  to train and test.""",
        type = str)
    parser.add_argument('--task', required=True,
        choices = ['score', 'dock', 'screen'], 
        help = """Choose the task for which you would like 
                  to train and test the scoring function.""",
        type = str)
    parser.add_argument('--predictions_out_fname', required = False, 
        default=None,
        help = """File name to which the PREDICTIONS of 
                  of the task-specific SF are saved.  
                  """,
        type = str)
    parser.add_argument('--performance_out_fname', required = False, 
        default=None,
        help = """File name to which the PERFROMANCE 
                  statistics of the task-specific SF are saved.    
                  """,
        type = str)

    parser.add_argument('--n_cpus', required = False,  
        default=None,
        help = """The number of CPU cores to use. All CPU cores will
                  be used if it was not assigned.    
                  """,
        type = int)
    parser.add_argument("--verbose", help="increase output verbosity",
                        action="store_true")
    args = parser.parse_args()
    if args.predictions_out_fname is None:
        args.predictions_out_fname = os.path.join('data', 'output', args.task, 
                                                  args.sfname + '_predictions.csv')
    if args.performance_out_fname is None:
        args.performance_out_fname = os.path.join('data', 'output', args.task, 
                                                  args.sfname + '_performance.csv')
    return args
#-------------------------------------------------------------------------------
def get_ba_data(tr_df, task):
    """
    This function is used to replace task-specific dependent labels
    such as ligand poses RMSD values for docking with binding affinity
    data. In the data frame tr_df, valid binding affinity values for the
    docking task are associated with ligand poses whose RMSD = 0,
    which are essentially the native conformations ('label' == 0). The rows
    with BA data of positive values are the actual active ligands for 
    the screening task ('label' > 0).  
    """
    if task == 'dock':
      tr_df = tr_df[tr_df['label']==0].copy()
    elif task == 'screen':
        tr_df = tr_df[tr_df['label'] > 0].copy()
    tr_df['label'] = tr_df['ba'].copy()
    return tr_df
#-------------------------------------------------------------------------------
def main():
    args = parse_and_process_args()
    sfname = args.sfname.lower()
    task = args.task.lower()
    preds_ofname = args.predictions_out_fname
    perf_ofname = args.performance_out_fname

    if ((sfname == 'bt-screen' and task != 'screen')
      or (sfname == 'bt-dock' and task != 'dock')):
      error_msg = 'ERROR: Scoring function %s is incompatible with the %sING task'
      print(error_msg%(sfname.upper(), task.upper()))
      print('ABORTING.')
      sys.exit() 


    if sfname in ['bt-score', 'bt-screen', 'bt-dock']:
      descriptor_sets = ['xscore', 'affiscore', 'rfscore', 'gold',
                        'repast', 'smina', 'chemgauss', 'autodock41', 
                        'ligscore', 'dsx', 'cyscore', 'padel', 
                        'nnscore', 'retest', 'ecfp', 'dpocket']
    elif sfname == 'rf-score':
      descriptor_sets = ['rfscore']
    elif sfname == 'x-score':
      descriptor_sets = ['xscore'] 
    rem_y = sfname in ['rf-score', 'x-score']
    
    model_params = {'n_trees': 3000, 'depth': 10,
                    'eta': 0.01, 'l_rate': 0.01}
    
    tr_dpath = os.path.join('data', 'input', task, 'primary-train')
    ts_dpath = os.path.join('data', 'input', task, 'core-test')

    train, ftrs_formula = read_plc_data(task, descriptor_sets=descriptor_sets, 
                                        rem_y=rem_y, data_path=tr_dpath,
                                        verbose=args.verbose)
    test, ftrs_formula = read_plc_data(task, descriptor_sets=descriptor_sets, 
                                       rem_y=rem_y, data_path=ts_dpath,
                                       verbose=args.verbose)
    if ((sfname in ['rf-score', 'x-score'])
        or (sfname == 'bt-score' and task != 'score')):
      train = get_ba_data(train, task)
    n_cpus = multiprocessing.cpu_count() if args.n_cpus is None else args.n_cpus
    model_params = {'n_cpus': n_cpus}
    predictions, performance = train_test_model(task, sfname, train, test, model_params)
    print('\nPerformance of %s on the %sing task:'%(args.sfname, args.task))
    print(performance.to_string(index=False))
    if preds_ofname is not None:
        if args.verbose:
          print('Writing predictions to ' + preds_ofname)
        predictions.to_csv(preds_ofname, index=False)
    if perf_ofname is not None:
        if args.verbose:
          print('Writing performance statistics to ' + perf_ofname)
        performance.to_csv(perf_ofname, index=False)
if __name__== '__main__':
    main()


