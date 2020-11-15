# Exoplanet_Hunting_in_Deep_Space
In the notebook, 'NASA_ExoplanetHunting.ipynb', 
the training data are processed as follows:
- 1-d uniformly filter --> detrend --> extract spectral --> normalize.  

Then we train a LinearSVC by using the normalized spectral of detrended
data and test it on the test data.  
The score on the test data is: **1.0 precision, 0.6 recal and 0.75 f1 score**.  

In the next step,  CNN-1D will be used to classify the data. Now just complete the conv-1d model construction.  

The file, 'utils.py', includes the functions used in the notebook.
