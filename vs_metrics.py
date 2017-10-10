import pandas as pd
from pandas import DataFrame
from sklearn.metrics import roc_auc_score
import numpy as np
import math 
#-------------------------------------------------------------------------------
def shuffle(df, n=1, axis=0):     
    sdf = df.copy()
    for _ in range(n):
        sdf.apply(np.random.shuffle, axis=axis)
    return sdf
#-------------------------------------------------------------------------------
def top_n_stats(sorted_data, x):
    n_t = len(sorted_data)
    top_x_n = int(math.ceil(n_t*x/float(100)))
    tdx = sorted_data[:top_x_n]
    ntbx = len(tdx[tdx.true_labels > 0])
    ntdx = len(tdx[tdx.true_labels <= 0])
    return [top_x_n, ntbx, ntdx]
#-------------------------------------------------------------------------------
def compute_screening_power(target_id, true_labels, pred_labels, decreasing=True):
    is_regression = len(np.unique(true_labels)) > 10
    n_pred_labels = (pred_labels - min(pred_labels))*1.0/(max(pred_labels) - min(pred_labels))
    d = DataFrame([n_pred_labels, target_id, true_labels]).transpose()
    d.columns = ['pred_labels', 'target_id', 'true_labels']
    d = d.reindex(np.random.permutation(d.index))
    unq_targets = np.unique(target_id)
    n_unq_targets = len(unq_targets)
    ef1 = ef5 = ef10 = s1 = s5 = s10 = 0.0
    cntr = 0
    for t in range(n_unq_targets):
        td = d[d.target_id == unq_targets[t]]
        td = td.sort_values(['pred_labels'], ascending=[1 - int(decreasing)])
        n_t = len(td)
        if(n_t >= 10):
            top_1_n, ntb1, ntd1 = top_n_stats(td, 1)
            top_5_n, ntb5, ntd5 = top_n_stats(td, 5)
            top_10_n, ntb10, ntd10 = top_n_stats(td, 10)
            ntbt = len(td[td.true_labels > 0])
            ntdt = len(td[td.true_labels <= 0])
            if ntbt >= 3 and ntdt > 5:
                ef1 = ef1 + (ntb1/(ntbt*0.01))
                ef5 = ef5 + (ntb5/(ntbt*0.05))
                ef10 = ef10 + (ntb10/(ntbt*0.1))
                s1 = s1 + 1 if ntb1 > 0 else s1
                s5 = s5 + 1 if ntb5 > 0 else s5
                s10 = s10 + 1 if ntb10 > 0 else s10
                cntr = cntr + 1.0
    res = [ef1/cntr, ef5/cntr, ef10/cntr, s1, s1/cntr, s5, s5/cntr, s10, s10/cntr]
    res = list(np.round(np.array(res), 2))
    """
    print('EF1% = ' + str(res[0]) + ', EF5% = ' + str(res[1]) + ', EF10% = ' + str(res[2]))
    print('S1 = ' + str(res[3]) + ', S1% = ' + str(100.0*res[4]))
    print('S5 = ' + str(res[5]) + ', S5% = ' + str(100.0*res[6]))
    print('S10 = ' + str(res[7]) + ', S10% = ' + str(100.0*res[8]))
    """
    return res
#-------------------------------------------------------------------------------
def compute_docking_power(ligand_id, true_rmsds, pred_labels, decreasing=False):
    is_regression = len(np.unique(true_rmsds)) > 10
    d = DataFrame([pred_labels, ligand_id, true_rmsds]).transpose()
    d.columns = ['pred_labels', 'ligand_id', 'true_rmsds']
    d = d.reindex(np.random.permutation(d.index))
    unq_ligands = np.unique(ligand_id)
    n_unq_ligands = len(unq_ligands)
    r = [0.0]*11
    r0n = r1n = r2n = r3n = 0
    for l in range(n_unq_ligands):
        ld = d[d.ligand_id == unq_ligands[l]]
        ldwon = ld[ld.true_rmsds != 0]
        if(np.std(ld.pred_labels) != 0):
            ld = ld.sort_values(['pred_labels'], ascending=[1 - int(decreasing)])
            ldwon = ldwon.sort_values(['pred_labels'], ascending=[1 - int(decreasing)])

            r0 = 1 if(len(ld[ld.true_rmsds == 0])>0) else 0
            r0n = r0n+1 if(len(ld[ld.true_rmsds == 0])>0) else r0n
            r1n = r1n+1 if(len(ld[ld.true_rmsds < 1])>0) else r1n
            r2n = r2n+1 if(len(ld[ld.true_rmsds < 2])>0) else r2n
            r3n = r3n+1 if(len(ld[ld.true_rmsds < 3])>0) else r3n

            r[0] = r[0] + 1 if(ld.true_rmsds.iloc[0] < 1) else r[0]
            r[1] = r[1] + 1 if(ld.true_rmsds.iloc[0] < 2) else r[1]
            r[2] = r[2] + 1 if(ld.true_rmsds.iloc[0] < 3) else r[2]

            r[3] = r[3] + 1 if(ldwon.true_rmsds.iloc[0] < 2) else r[3]

            r[6] = r[6] + 1 if(ld.true_rmsds.iloc[0] < 2 
                            or ld.true_rmsds.iloc[1] < 2) else r[6]
            r[7] = r[7] + 1 if(ld.true_rmsds.iloc[0] < 2 
                            or ld.true_rmsds.iloc[1] < 2 
                            or ld.true_rmsds.iloc[2] < 2) else r[7]
            r[8] = r[8] + 1 if(ld.true_rmsds.iloc[0] == 0) else r[8]
            r[9] = r[9] + 1 if(ld.true_rmsds.iloc[0] == 0 
                            or ld.true_rmsds.iloc[1] == 0 
                            or ld.true_rmsds.iloc[2] == 0) else r[9]
            r[10] = r[10] + 1 if(ld.true_rmsds.iloc[0] == 0 
                              or ld.true_rmsds.iloc[1] == 0 
                              or ld.true_rmsds.iloc[2] == 0 
                              or ld.true_rmsds.iloc[3] == 0 
                              or ld.true_rmsds.iloc[4] == 0) else r[10]
            rmsd = ld.true_rmsds.iloc[0]
    r[0] /= r1n
    r[1] /= r2n
    r[2] /= r3n

    r[3] /= r2n

    r[4] = r[1]
    r[5] = r[2]

    r[6] /= r2n
    r[7] /= r2n
    r[8] = 0 if(r0n==0) else r[8]/r0n
    r[9] = 0 if(r0n==0) else r[9]/r0n
    r[10] = 0 if(r0n==0) else r[10]/r0n

    r = list(np.round(np.array(r)*100.0,2))
    """
    print('S11 = %.2f, S21 = %.2f, S31 = %.2f'%(r[0], r[1], r[2]))
    print('S21_wo_native = %.2f, S22 = %.2f, S23 = %.2f'%(r[3], r[6], r[7]))
    print('S01 = %.2f, S03 = %.2f, S05 = %.2f'%(r[8], r[9], r[10]))
    """
    return r[0:3] + r[3:4] + r[6:11] 
#-------------------------------------------------------------------------------
def compute_ranking_power(predictions):
    r123, r213, r132, r231, r312, r321 = [0, 1, 2, 3, 4, 5]
    rpwr = np.array([0.0]*6)
    n_clstrs = 0
    for i in range(0, len(predictions), 3):
        n_clstrs += 1
        c = predictions[i:(i+3)]
        if len(np.unique(c)) == len(c):
            v = np.argsort(c[::-1]) + 1
            if v[0]==3 and v[1]==2 and v[2]==1: rpwr[r123] += 1.0
            if v[0]==2 and v[1]==3 and v[2]==1: rpwr[r213] += 1.0
            if v[0]==3 and v[1]==1 and v[2]==2: rpwr[r132] += 1.0
            if v[0]==2 and v[1]==1 and v[2]==3: rpwr[r231] += 1.0
            if v[0]==1 and v[1]==3 and v[2]==2: rpwr[r312] += 1.0
            if v[0]==1 and v[1]==2 and v[2]==3: rpwr[r321] += 1.0
    return list(np.round(rpwr*100.0/n_clstrs, 1))
#-------------------------------------------------------------------------------