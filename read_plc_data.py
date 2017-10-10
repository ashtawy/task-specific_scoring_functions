import pandas as pd
import numpy as np
import os
#-------------------------------------------------------------------------------
def read_plc_data(task, descriptor_sets, rem_y=False, 
                  data_path='./data/',
                  lco='ltco', cluster_id=90, 
                  verbose=False):

    """Reads protein-ligand complex features and labels.

    Args:
        task: 'score', 'screen', or 'dock'
        descriptor_sets: a list of descriptor names to be used. E.g., ['xscore', 'repast']
        rem_y: whether to remove the value predicted by the scoring function
               that generates the descriptor set (e.g., X-Score).
        lco: The leave cluster out strategy. LTCO for leave-target-clusters-out
             and LLCO for leave-ligand-clusters-out
        cluster_id: The clustering ID. For LTCO, it is the BLAST similarity
                    cutoff value of 90% (cluster_id=90). 
                    For LLCO, it is the number of clusters generated based on 
                    the pair-wise Euclidean distance between ligands described
                    by the 740+ PaDEL descriptors. We generate 100 ligand clusters
                    and therefore cluster_id=100 when lco=LLCO.
    Returns:
        A pandas dataframe with:
        * descriptors from the descriptor_sets (i.e., the independent variables X). 
        * 'label' for the dependent variable which is binding affinity 
           when task='score', binary activity label (1 or 0) when task='screen',
           and the ligands's pose distance from the native confirmation in terms of RMSD
           when task='dock'. 
        * 'grp_ids' which are complex PDB codes.
        * 'clstr_ids' which indicates the target or ligands cluster of each complex in grp_ids 
    """
    
    grp_ids = None
    clstr_ids = None
    X = None
    y = None
    tbl_fname = os.path.join(data_path, 'complexes_and_y.csv')
    table = pd.read_csv(tbl_fname)
    N = table.shape[0]
    cmplx_names = table['X1_complex_code'].values
    grp_ids = get_prefixes(cmplx_names)
    ba = table['X2_meas_aff'].values
    y = ba.copy()

    if task == 'dock':
        if 'X8_pose_rmsd' in table.columns:
            y = table['X8_pose_rmsd'].values
    if task == 'screen':
        y[y != 0] = 1
    if lco == 'lco' or lco == 'ltco':
        clstr_ids = get_protein_clusters(data_path, cluster_id)
    elif lco == 'llco':
        clstr_ids = get_ligand_clusters(data_path, cluster_id)

    X = pd.DataFrame()
    ftrs_formula = ''
    all_cnames = []
    for descriptor_set in descriptor_sets:
        ds_fname = os.path.join(data_path, descriptor_set + '.csv')
        ds_fname_gzip = ds_fname + '.gzip'
        if os.path.exists(ds_fname):
            if verbose:
                print('Now reading: %s'%(ds_fname))
            x_ds = pd.read_csv(ds_fname)
        elif os.path.exists(ds_fname_gzip):
            if verbose:
                print('Now reading: %s'%(ds_fname_gzip))
            x_ds = pd.read_csv(ds_fname_gzip, compression='gzip')
        else:
            print('ERROR: UNABLE TO FIND ANY OF:')
            print(ds_sname + '\n' + ds_fname_gzip)
        x_ds.fillna(x_ds.mean(), inplace=True)
        ds_sname = get_short_dsname(descriptor_set)
        x_ds = change_descriptor_names_and_rem_y(x_ds, ds_sname, rem_y)
        ftrs_formula += ds_sname + str(x_ds.shape[1])
        all_cnames += list(x_ds.columns.values)
        X = pd.concat([X, x_ds], axis=1, ignore_index=True)
        X.columns = all_cnames
    Xy_grps = X
    Xy_grps['label'] = y
    Xy_grps['ba'] = ba
    Xy_grps['grp_ids'] = grp_ids
    Xy_grps['clstr_ids'] = clstr_ids
    return [Xy_grps, ftrs_formula]
#-------------------------------------------------------------------------------
def get_short_dsname(l_dsname):
    ls_dsname_dic = {'affiscore': 'a', 'autodock': 'u', 'autodock2': 'U', 
                    'autodock41': 'U',
                    'blast_protein': 'b', 'blast80_protein': 'b', 
                    'blast': 'b', 'blast_protein_extended': 'B', 
                    'repast': 'b',
                    'chemgauss': 'h', 'cyscore': 'c', 'dpocket': 'f', 
                    'dsx': 'd', 'gold': 'g', 'ligscore': 'l', 
                    'nnscore': 'n', 'padel': 'p', 'ecfp': 'e', 
                    'rfscore': 'r', 'rfscore_original': 'r', 
                    'rfscore_standard': 'r', 'rfscore_extended': 'R', 
                    'rfscore_xExtended': 'R', 'smina': 's', 'tmalign': 't', 
                    'tmalign_protein': 't', 'tmalign_protein_extended': 'T',
                    'retest': 't', 
                    'xscore': 'x', 'zernike': 'z', 'sda3': 'S'}
    s_dsname = None
    if l_dsname.lower() in ls_dsname_dic:
        s_dsname = ls_dsname_dic[l_dsname.lower()]
    return s_dsname
#-------------------------------------------------------------------------------
def change_descriptor_names_and_rem_y(ds_df, sname_prefix, rem_y):
    od_names = ds_df.columns.values
    nd_names = []
    nd_names_to_rem = []
    for od_name in od_names:
        tokens = od_name.split('_')
        oprefix = tokens[0]
        tokens[0] = sname_prefix
        nd_name = '_'.join(tokens)
        if oprefix[0].lower() == 'y':
            nd_name += '_y'
            nd_names_to_rem.append(nd_name)
        nd_names.append(nd_name)
    ds_df.columns = nd_names
    if rem_y and len(nd_names_to_rem) > 0:
        ds_df.drop(nd_names_to_rem, axis=1, inplace=True)
    return ds_df
#-------------------------------------------------------------------------------
def get_protein_clusters(file_prefix, sim_cutoff):
    clsts_fname = os.path.join(file_prefix, 'target_clusters.csv')
    raw_clusters = None
    if os.path.exists(clsts_fname):
        clsts_table = pd.read_csv(clsts_fname)
        clm_name = 'X_' + str(sim_cutoff)
        if sim_cutoff != 100 and clm_name in clsts_table.columns:
            raw_clusters = clsts_table[clm_name].values
        elif sim_cutoff == 100:
            mnames = clsts_table['X1_complex_code'].values
            u_mnames = np.unique(mnames)
            raw_clusters = pd.match(mnames, u_mnames)
        else:
            raw_clusters = None
    return raw_clusters
#-------------------------------------------------------------------------------
def get_ligand_clusters(file_prefix, n_clstrs):
    clsts_fname = os.path.join(file_prefix, 'ligand_clusters.csv')
    raw_clusters = None
    if os.path.exists(clsts_fname):
        clsts_table = pd.read_csv(clsts_fname)
        clm_name = 'X_' + str(n_clstrs)
        if clm_name in clsts_table.columns:
            raw_clusters = clsts_table[clm_name].values
        else:
            raw_clusters = None
    return raw_clusters
#-------------------------------------------------------------------------------
def get_prefixes(codes):
    return [x.split('_')[0] for x in codes]
#-------------------------------------------------------------------------------
def get_suffixes(codes):
    return [x.split('_')[1] if len(x.split('_'))==2 else '' for x in codes]
#-------------------------------------------------------------------------------