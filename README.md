# Spectrum-Machine-Learning

## Requirements
* Install dependencies:  
`pip install -r requirements.txt`
* PyTorch dependencies[^1]:  
NVIDIA GPU with CUDA Toolkit > v11.6
[^1]: Only required for use with NVIDIA GPU

## File Descriptions
### spectrum_fit_cnn:
Spectrum machine learning script to train and test both encoder and decoder.  
`main` function can be run to train and test encoder or decoder.  

Configurable parameters are:
* num_epochs (int) - Number of epochs to train
* val_frac (float) - Fraction of training data to split into validation data
* data_dir (str) - Directory containing background and response files
* spectra_path (str) - Path to supervised spectra file
* synth_path (str) - Path to synthetic spectra file
* params_path (str) - Path to supervised parameters file
* synth_params_path (str) - Path to synthetic parameters file
* fix_params (list) - List of parameters to fix with index (from 1) and value
* param_limits (dict) - Lower & upper limits for each free parameter and index of logarithmic parameters

### synthesize_spectra:
Generates synthetic spectra for decoder training.  
`main` function can be run to generate synthetic spectra.  

Configurable parameters are:
* clear_synth (bool) - If existing synthetic spectra should be removed
* batches (int) - Number of times to save progress until completion
* total_synth_num (int) - Total number of synthetic spectra to generate
* synth_dir (str) - Directory to save synthetic spectra
* synth_data (str) - File to save synthetic spectra
* param_limits (dict) - Id, lower & upper limit for each free parameter and if it is in logarithmic space

### data_preprocessing
Converts data from fits files into numpy array file and preprocesses the data to match Xspec's preprocessing.  
`preprocess` can be run to convert all files from a directory into a numpy array file.  
`spectrum_data` can be run to convert single fits file into a numpy array.  

Configurable parameters for `preprocess` are:
* data_dir (str) - Path to fits spectra directory
* labels_path (str) - Path to parameters file for each spectrum

Options for the function `spectrum_data` are:
* spectrum_path (str) - Path to the spectrum fits file
* background_dir (str = '') - Path to the directory containing the background fits file
* cut_off (list = [0.3, 10]) - Lower & upper limit of accepted energies in keV

### data_preprocessing_comparison
Compares the performance of data_preprocessing to Xspec's preprocessing.  
'main' function can be run to generate comparison graphs.  

Configurable parameters are:
* plot_diff (bool) - Whether to plot the difference between preprocessing and Xspec
* data_root (str) - Path to the root directory that contains the spectra, background and response files
* spectrum (str) - Path to the spectrum file relative to data_root