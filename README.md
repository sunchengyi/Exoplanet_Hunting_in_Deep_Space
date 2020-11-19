# Exoplanet_Hunting_in_Deep_Space
In the notebook, 'NASA_ExoplanetHunting.ipynb', 
the training data are processed as follows:
- 1-d uniformly filter --> detrend --> extract spectral --> normalize.  

Then we train a LinearSVC by using the normalized spectral of detrended
data and test it on the test data.  
The score on the test data is: **1.0 precision, 0.6 recal and 0.75 f1 score**.  

Construct a CNN-1D model to classify the data.
Firstly, train the model by using the standardized data and the uniformly filtered data. 
After 30 epochs, on the validation data (20% of the origni trainging data), 
the scores are **1.0 precision, 0.71 recall and 0.83 f1 score**.
Secondly, after resampling the training data to balance the labels, train the model.
After 30 epochs, on the validation data (20% of the origni trainging data), 
the scores are **1.0 precision, 0.57 recall and 0.73 f1 score**.

It seems that the model becomes a litter worse after balanced the labels.

The file, 'utils.py', includes the functions used in the notebook.
