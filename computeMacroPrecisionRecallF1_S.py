# -*- coding: utf-8 -*-

'''
python3
'''

import numpy as np
import pandas as pd
import argparse
import os
import collections

def computeTP(y_true, y_predict, cidx):
    TP = 0
    for i in range(len(y_true)):
        if y_true[i] == cidx and y_predict[i] == cidx:
            TP += 1
    return TP

def computeFP(y_true, y_predict, cidx):
    FP = 0
    for i in range(len(y_true)):
        if y_true[i] != cidx and y_predict[i] == cidx:
            FP += 1
    return FP

def computeTN(y_true, y_predict, cidx):
    TN = 0
    for i in range(len(y_true)):
        if y_true[i] != cidx and y_predict[i] != cidx:
            TN += 1
    return TN

def computeFN(y_true, y_predict, cidx):
    FN = 0
    for i in range(len(y_true)):
        if y_true[i] == cidx and y_predict[i] != cidx:
            FN += 1
    return FN

def computeMacroPrecisionRecallF1(y_true_series, y_predict_series):
    precisionlist = []
    recalllist = []
    classlist = sorted(collections.Counter(y_true_series.reshape(-1).tolist()).keys())
    for cidx in classlist:
        TP = computeTP(y_true_series, y_predict_series, cidx)
        FP = computeFP(y_true_series, y_predict_series, cidx)
#        TN = computeTN(y_true_series, y_predict_series, cidx)
        FN = computeFN(y_true_series, y_predict_series, cidx)

        precision = float(TP) / (TP + FP)
        recall = float(TP) / (TP + FN)
        precisionlist.append(precision)
        recalllist.append(recall)
    
    macro_precision = sum(precisionlist) / len(classlist)
    macro_recall = sum(recalllist) / len(classlist)
    macro_f1 = 2*macro_precision*macro_recall / (macro_precision + macro_recall)
    return macro_f1, macro_precision, macro_recall 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute the Macro Precision, Recall, and F1.')
    parser.add_argument('fp_true', metavar='Truelabels', type=str, 
                        help='The CSV file of ture lables with the index and header, the columns are the index of classes and the rows are the samples as a Mx1 matrix.')
    parser.add_argument('fp_predict', metavar='Prediction', type=str, nargs='+', 
                        help='The CSV file of predction with the index and header, the columns are the index of classes and the rows are the samples as a Mx1 matrix.')
    args = parser.parse_args()
    
    print(args.fp_true)
    print(args.fp_predict)
    
    y_true = pd.read_csv(args.fp_true, index_col=0, dtype=np.float).values

    print('Results,marcoPrecision,MacroRecall,MacroF1')
    for fp in args.fp_predict:
        y_pre = pd.read_csv(fp, index_col=0, dtype=np.float).values
        p, r, f1 = computeMacroPrecisionRecallF1(y_true, y_pre)
        print('%s,%f,%f,%f' % (os.path.basename(fp).split('.')[0], p, r, f1))

