# Can Ensemble Deep Learning Identify People by Their Gait Using Data Collected from Multi-Modal Sensors in Their Insole?
This repository is the official implementation of "Can Ensemble Deep Learning Identify People by Their Gait Using Data Collected from Multi-Modal Sensors in Their Insole?"
# Requirements
To install requirements:
```
pip install -r requirements.txt
```
# Training
The averaging ensemble model utilizes a CNN and a RNN model trained independently. The trained models are saved in folder especified in the source code. The sequence required to train the whole model is as follows:
```
cd models
python Experiments_CNN.py
python Experiments_RNN.py
python Experiments_Ensemble.py
```
We repeat the experiment 20 times. 
