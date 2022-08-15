# Classify names by gender

### Quick-start
Run main.py

## Models
I tested out two models, both available via main.py

### Naive Bayes
The first is a Naive Bayes model.
It's designed to be simple and fast,
at the cost of being not very accurate.

### LSTM
The second is a bidirectional LSTM.
It's designed to be a bit more sophisticated.
It takes longer to train and uses hyperparameters.

### requirements.txt
requirements.txt was generated using
conda list -e > requirements.txt

ml.yml was generated using
conda env export > ml.yml

A conda environment satisfying all requirements can be generated quickly using
conda env create -f ml.yml

