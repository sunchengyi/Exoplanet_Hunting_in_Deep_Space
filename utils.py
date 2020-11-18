
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback
import numpy as np

from sklearn.metrics import precision_score, recall_score, accuracy_score,\
                            f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import Normalizer

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt

from scipy.signal import spectrogram
import scipy
from scipy.ndimage import uniform_filter1d


from tqdm.notebook import tqdm, trange


# dictionary from label to name
label_dict = {0: 'without Exoplanet', 1: 'with Exoplanet'}

#-------------------------------------------------------------------------
def labelTransform(df):
    """
    Split 'df.values' as the label data (first column) and the feature data 
    (except the first column). And labels in label data is changed: (1 to 0) 
    and (2 to 1). Finally, return the feature data and label data.
    """
    data = df.values
    return data[:, 1:], data[:, 0]-1

#-------------------------------------------------------------------------
def scores_predict(y_pred, data_y):
    """
    Evaluate the confusion matirx and the metrics on 'y_pred' and 'data_y' data.
    The metric list to be used are ['precision','recall', 'accuracy','f1', 
    'f1_macro', 'f1_weighted' ]
    """
    scores = {'conf_matrix': confusion_matrix(data_y, y_pred),
              'precision': precision_score(data_y, y_pred),
              'recall': recall_score(data_y, y_pred),
              'accuracy': accuracy_score(data_y, y_pred),
              'f1': f1_score(data_y, y_pred),
              'f1_micro': f1_score(data_y, y_pred, average='micro'),
              'f1_macro': f1_score(data_y, y_pred, average='macro'),
              'f1_weighted': f1_score(data_y, y_pred, average='weighted')}
    print("Confusion matrix:" )
    print(scores['conf_matrix'])
    print("precision: ", scores['precision'])
    print("recall: ", scores['recall'])
    print("accuracy: ", scores['accuracy'])
    print("f1: ", scores['f1'])
    print("f1_macro: ", scores['f1_macro'])
    print("f1_micro: ", scores['f1_micro'])
    print("f1_weighted: ", scores['f1_weighted'])
    return scores

#-------------------------------------------------------------------------
def label_balance(X, y):
    balancing = SMOTE(sampling_strategy='minority', random_state=42)
    return balancing.fit_resample(X, y)

#-------------------------------------------------------------------------    
def myCVSplitter(X, y, n_splits, shuffle=False, random_state=1):
    '''
    Cross validation splitter to yield the batches of over-sampled training data
    and origin test data. 
    '''    
    SKFold = StratifiedKFold(n_splits, shuffle=shuffle, random_state=random_state)
    for train_idx, test_idx in SKFold.split(X, y):
        print(f"Length of Training: {len(train_idx)}, length of Testing:{len(test_idx)}")
        x_train_batch = X[train_idx]
        y_train_batch = y[train_idx]
        x_train_batch_res, y_train_batch_res = label_balance(x_train_batch, y_train_batch)
        yield x_train_batch_res, y_train_batch_res, X[test_idx], y[test_idx]

#-------------------------------------------------------------------------
def cv_pred(classifier, X, y, n_splits):
    y_pred = None
    y_label = None
    # loop over the 'n_splits'fold.
    for x_train, y_train, x_test, y_test in myCVSplitter(X, y, n_splits):
        classifier.fit(x_train, y_train)
        y_pred_batch = classifier.predict(x_test)
        if y_pred is None:
            y_pred = y_pred_batch
            y_label = y_test
        else:
            y_pred = np.hstack((y_pred, y_pred_batch))
            y_label = np.hstack((y_test, y_label))
    return y_pred, y_label

#-------------------------------------------------------------------------
def model_evaluator(X, y, model, n_splits=5):
    Y_pre, Y_label = cv_pred(model, X, y, n_splits)
    return scores_predict(Y_pre, Y_label)

#-------------------------------------------------------------------------
def plot_flux_spectra(x):
    '''
    Visualize the data
    '''
    plt.figure(figsize=(20,3))
    # f, (ax1, ax2) = plt.subplots(1,2, gridspec_kw={'width_ratios': [3, 1]})
    ax1 = plt.subplot(121)
    ax1.plot(x)
    #--------------------------
    ax2 = plt.subplot(164)
    NFFT = len(x)-1
    Fs = 1
    noverlap = NFFT-1
    Pxx, freqs, bins, im = ax2.specgram(x, NFFT=NFFT, Fs=Fs, noverlap=noverlap)
    ax3 = plt.subplot(165)
    ax3.plot(Pxx[:,0])
    ax4 = plt.subplot(166)
    ax4.plot(Pxx[:,1])
    plt.show()
    return Pxx

#-------------------------------------------------------------------------
def plot_flux_spectra_scipy(x, title_flux, title_spctrum):
    plt.figure(figsize=(20,3))
    # f, (ax1, ax2) = plt.subplots(1,2, gridspec_kw={'width_ratios': [3, 1]})
    ax1 = plt.subplot(121)
    ax1.set_title(title_flux)
    ax1.set_ylabel('Flux')
    ax1.set_xlabel('Time')
    ax1.plot(x)
    #--------------------------
    #ax2 = plt.subplot(164)
    NFFT = len(x)-1
    Fs = 1
    noverlap = NFFT-1
    freqs, bins, Pxx = spectrogram(x, nperseg=NFFT, fs=Fs, noverlap=noverlap)
    ax3 = plt.subplot(143)
    ax3.plot(Pxx[:,0])
    #ax3.title(f'Flux of star {i_row} {label}')
    ax3.set_title(title_spctrum)
    ax3.set_ylabel('Amplitude')
    ax3.set_xlabel('Frequency')

    #ax4 = plt.subplot(144)
    #ax4.plot(Pxx[:,1])
    plt.show()
    return Pxx
#-------------------------------------------------------------------------
def plot_flux_spectra_scipy_fft(x):
    '''
    Doo the Fourier transformation directly.
    '''
    plt.figure(figsize=(20,3))
    # f, (ax1, ax2) = plt.subplots(1,2, gridspec_kw={'width_ratios': [3, 1]})
    ax1 = plt.subplot(121)
    ax1.plot(x)
    #--------------------------
    #ax2 = plt.subplot(164)
    #NFFT = len(x)-1
    #Fs = 1
    #noverlap = NFFT-1
    Pxx = scipy.fft(x)
    ax3 = plt.subplot(165)
    ax3.plot(Pxx)
    ax4 = plt.subplot(166)
    ax4.plot(Pxx)
    plt.show()
    return Pxx
#-------------------------------------------------------------------------
def get_spectra(x):
    '''
    Get the spectra from the scaled 'x'.
    ---------
    Arguments
    :x: 2-D np.array
    ------
    Return
    :spectra: 2-D np.array
    '''
    spectra = None
    NFFT = x.shape[1] - 1
    Fs = 1
    noverlap = NFFT-1
    for row in tqdm(x):
        Pxx, freqs, bins, im = plt.specgram(row, NFFT=NFFT, Fs=Fs, noverlap=noverlap)
        if spectra is None:
            spectra = Pxx[:,0]
        else:
            spectra = np.vstack((spectra, Pxx[:,0]))
    return spectra

#-------------------------------------------------------------------------
def get_spectra_scipy(x):
    '''
    Get the spectra from the scaled 'x'.
    ---------
    Arguments
    :x: 2-D np.array
    ------
    Return
    :spectra: 2-D np.array
    '''
    NFFT = x.shape[1] - 1
    Fs = 1
    noverlap = NFFT-1
    freqs, bins, Pxx = spectrogram(x, nperseg=NFFT, fs=Fs, axis=1, noverlap=noverlap)
    return Pxx[:, :, 0]

#-------------------------------------------------------------------------
def reduce_upper_outliers(df,reduce = 0.01, half_width=4):
    '''
    Since we are looking at dips in the data, we should remove upper outliers.
    The function is taken from here:
    https://www.kaggle.com/muonneutrino/exoplanet-data-visualization-and-exploration
    '''
    length = len(df.iloc[0,:])
    remove = int(length*reduce)
    for i in df.index.values:
        values = df.loc[i,:]
        sorted_values = values.sort_values(ascending = False)
       # print(sorted_values[:30])
        for j in range(remove):
            idx = sorted_values.index[j]
            #print(idx)
            new_val = 0
            count = 0
            idx_num = int(idx[5:])
            #print(idx,idx_num)
            for k in range(2*half_width+1):
                idx2 = idx_num + k - half_width
                if idx2 <1 or idx2 >= length or idx_num == idx2:
                    continue
                new_val += values['FLUX.'+str(idx2)] # corrected from 'FLUX-' to 'FLUX.'
                
                count += 1
            new_val /= count # count will always be positive here
            #print(new_val)
            if new_val < values[idx]: # just in case there's a few persistently high adjacent values
                df.at[i,idx] = new_val
    return df

#-------------------------------------------------------------------------
def data_transformer_final(data, size_filter=15, mode_filter='wrap'):
    """
    Transform 'data' following the processes:
    '1-d uniformly filter --> detrend --> extrac spectral --> normalize'.
    ---------
    Arguments
    data: 2-d np.array with shape (n_samples, n_features).
    size_filter=15: integer, the length of uniform filter.
    mode_filter='wrap': string, the paramter, 'mode' in 
        'scipy.ndimage.uniform_filter1d'.
    ------
    Return
    data_new: 2-d np.array with shape (n_samples, n_new_features).
    """
    # detrended data after uniformly filtered
    data_unif_detrended = data - uniform_filter1d(data, size_filter, axis=1, mode=mode_filter)
    # spectral
    data_unif_detr_spectral = get_spectra_scipy(data_unif_detrended) # spectra
    normalizer = Normalizer()
    #normalize
    data_unif_detr_spec_nor = normalizer.fit_transform(data_unif_detr_spectral)

    return data_unif_detr_spec_nor

#-------------------------------------------------------------------------
def conv1d_block(filters, name, kernel_size=15, pool_size=4):
    return [layers.Conv1D(filters=filters, kernel_size=kernel_size, use_bias=False, 
                          name=name),
            layers.BatchNormalization(name=f"{name}_bn"),
            layers.Activation('relu',name=f"{name}_act"),
            layers.MaxPool1D(pool_size=pool_size,strides=pool_size, 
                             name=f"{name}_max")]
#-------------------------------------------------------------------------
class Metrics_f1_precision_recall(Callback):
    '''
    Custom a callback to calculate the f1 score, precison and recall at the 
    end of each epoch on the validation data.
    '''
    def __init__(self, val_x, val_y, batch_size=500):
        Callback.__init__(self)
        self.val_x = val_x
        self.val_y = val_y
        self.batch_size = batch_size
    def on_epoch_end(self, epoch, logs=None):
        val_pred = self.model.predict(self.val_x, batch_size=self.batch_size).round()
        val_f1 = f1_score(self.val_y, val_pred)
        val_precision = precision_score(self.val_y, val_pred)
        val_recall = recall_score(self.val_y, val_pred)
        print(f' - val_f1: {val_f1:.3f} - val_precision: {val_precision: .3f} '
              f'- val_recall: {val_recall: .3f}')


