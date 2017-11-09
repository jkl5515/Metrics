# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import argparse
import os

def computeTP(y_true, y_predict):
    TP = 0
    for i in range(len(y_true)):
        if y_true[i] == 1. and y_predict[i] == 1.:
            TP += 1
    return TP

def computeFP(y_true, y_predict):
    FP = 0
    for i in range(len(y_true)):
        if y_true[i] == 0. and y_predict[i] == 1.:
            FP += 1
    return FP

def computeTN(y_true, y_predict):
    TN = 0
    for i in range(len(y_true)):
        if y_true[i] == 0. and y_predict[i] == 0.:
            TN += 1
    return TN

def computeFN(y_true, y_predict):
    FN = 0
    for i in range(len(y_true)):
        if y_true[i] == 1. and y_predict[i] == 0.:
            FN += 1
    return FN

def computeMacroPrecisionRecallF1(y_true_matrix, y_predict_matrix, n_class):
    precisionlist = []
    recalllist = []
    for cidx in range(n_class):
        TP = computeTP(y_true_matrix[:, cidx], y_predict_matrix[:, cidx])
        FP = computeFP(y_true_matrix[:, cidx], y_predict_matrix[:, cidx])
#        TN = computeTN(y_true_matrix[:, cidx], y_predict_matrix[:, cidx])
        FN = computeFN(y_true_matrix[:, cidx], y_predict_matrix[:, cidx])
        precision = float(TP) / (TP + FP)
        recall = float(TP) / (TP + FN)
        precisionlist.append(precision)
        recalllist.append(recall)
    
    macro_precision = sum(precisionlist) / n_class
    macro_recall = sum(recalllist) / n_class
    macro_f1 = 2*macro_precision*macro_recall / (macro_precision + macro_recall)
    return macro_f1, macro_precision, macro_recall 

def computeMicroPrecisionRecallF1(y_true_matrix, y_predict_matrix, n_class):
    TPlist = []
    FPlist = []
#    TNlist = []
    FNlist = []
    for cidx in range(n_class):
        TP = computeTP(y_true_matrix[:, cidx], y_predict_matrix[:, cidx])
        FP = computeFP(y_true_matrix[:, cidx], y_predict_matrix[:, cidx])
#        TN = computeTN(y_true_matrix[:, cidx], y_predict_matrix[:, cidx])
        FN = computeFN(y_true_matrix[:, cidx], y_predict_matrix[:, cidx])
        TPlist.append(TP)
        FPlist.append(FP)
#        TNlist.append(TN)
        FNlist.append(FN)
        
    TP_avg = sum(TPlist) / float(n_class)
    FP_avg = sum(FPlist) / float(n_class)
#    TN_avg = sum(TNlist) / float(n_class)
    FN_avg = sum(FNlist) / float(n_class)
    
    micro_precision = float(TP_avg) / (TP_avg + FP_avg)
    micro_recall = float(TP_avg) / (TP_avg + FN_avg)
    micro_f1 = 2*micro_precision*micro_recall / (micro_precision + micro_recall)
    return micro_f1, micro_precision, micro_recall

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute the Macro Precision, Recall, and F1.')
    parser.add_argument('fp_true', metavar='Truelabels', type=str, 
                        help='The CSV file of ture lables with the index and header, the columns are the classes and the rows are the samples as a MxN matrix.')
    parser.add_argument('fp_predict', metavar='Prediction', type=str, nargs='+', 
                        help='The CSV file of predction with the index and header, the columns are the classes and the rows are the samples as a MxN matrix.')
    parser.add_argument('-nc', dest='nclass', metavar='N_Classes', type=int, required=True, 
                        help='The number of the classes.')
    args = parser.parse_args()
    
    print(args.nclass)
    print(args.fp_true)
    print(args.fp_predict)
    
    y_true = pd.read_csv(args.fp_true, index_col=0).values
    if y_true.shape[1] != args.nclass:
        exit("The specificed N_Classes is not equal to the number of the classes in %s." % args.fp_true)
    for fp in args.fp_predict:
        y_pre = pd.read_csv(fp, index_col=0).values
        if y_pre.shape[1] != args.nclass:
            exit("The specificed N_Classes is not equal to the number of the classes in %s." % fp)
    
    print('Results,marcoPrecision,MacroRecall,MacroF1')
    for fp in args.fp_predict:
        y_pre = pd.read_csv(fp, index_col=0).values
        p, r, f1 = computeMacroPrecisionRecallF1(y_true, y_pre, args.nclass)
        print('%s,%f,%f,%f' % (os.path.basename(fp).split('.')[0], p, r, f1))

