# Metrics
ML Machine Learning Metrics

## computeMacroPrecisionRecallF1.py
usage: computeMacroPrecisionRecallF1.py [-h] -nc N_Classes Truelabels Prediction [Prediction ...]

Compute the Macro Precision, Recall, and F1.

positional arguments:
  * Truelabels     The CSV file of ture lables with the index and header, the
                 columns are the classes and the rows are the samples as a MxN
                 matrix.
  * Prediction     The CSV file of predction with the index and header, the
                 columns are the classes and the rows are the samples as a MxN
                 matrix.

optional arguments:
  -h, --help     show this help message and exit
  -nc N_Classes  The number of the classes.
  
## computeMicroPrecisionRecallF1.py
usage: computeMicroPrecisionRecallF1.py [-h] -nc N_Classes
                                        Truelabels Prediction [Prediction ...]

Compute the Micro Precision, Recall, and F1.

positional arguments:  
  Truelabels     The CSV file of ture lables with the index and header, the  
                 columns are the classes and the rows are the samples as a MxN  
                 matrix.  
  Prediction     The CSV file of predction with the index and header, the  
                 columns are the classes and the rows are the samples as a MxN  
                 matrix.  

optional arguments:  
  -h, --help     show this help message and exit   
  -nc N_Classes  The number of the classes.  
