# Metrics
ML Machine Learning Metrics

## computeMacroPrecisionRecallF1.py
usage: computeMacroPrecisionRecallF1.py [-h] -nc N_Classes Truelabels Prediction [Prediction ...]<br/>

Compute the Macro Precision, Recall, and F1.<br/>

positional arguments:<br/>
  >Truelabels     >The CSV file of ture lables with the index and header, the columns are the classes and the rows are the samples as a MxN matrix<br/>
  >Prediction     >The CSV file of predction with the index and header, the columns are the classes and the rows are the samples as a MxN matrix<br/>

optional arguments:<br/>
  >-h, --help     show this help message and exit<br/>
  >-nc N_Classes  The number of the classes<br/>
  
## computeMicroPrecisionRecallF1.py
usage: computeMicroPrecisionRecallF1.py [-h] -nc N_Classes Truelabels Prediction [Prediction ...]<br/>

Compute the Micro Precision, Recall, and F1.<br/>

positional arguments:<br/>
  Truelabels     >The CSV file of ture lables with the index and header, the columns are the classes and the rows are the samples as a MxN matrix<br/>
  Prediction     >The CSV file of predction with the index and header, the columns are the classes and the rows are the samples as a MxN  
                 matrix<br/>

optional arguments:<br/>
  -h, --help     show this help message and exit<br/>
  -nc N_Classes  The number of the classes<br/>
