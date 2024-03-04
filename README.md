# Fast Spectra Predictor Network (FSP-Net)

FSP-Net is an autoencoder with two halves: an encoder (blue box) and a decoder (red box).  
The encoder is used to predict spectral parameters from spectra,
while the decoder reconstructs spectra from spectral parameters.

See the [Wiki](../../wiki) for more information on how to use this repository.

The research paper can be found on [MNRAS](https://doi.org/10.1093/mnras/stae629) or the [Arxiv](https://arxiv.org/abs/2310.17249).

![Diagram showing the encoder and decoder that make up the autoencoder](./FSP-Net.png)

## Requirements

* Install dependencies:  
`pip install -r requirements.txt`  
* PyTorch's dependencies[^1]:  
  NVIDIA GPU with [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) ~= v11.6
  [^1]: Only required for use with NVIDIA GPU
* PyXspec dependency[^2]:  
  Xspec from [HEASoft](https://heasarc.gsfc.nasa.gov/lheasoft/install.html) provided by NASA
  [^2]: Only required if using `synthesize_spectra.py` or `spectrum_fit.py`&rarr;`pyxspec_test`

## System Specification

The system specs used to develop the network are:
* Operating system: Ubuntu
* CPU: Intel 6C/12T, 2.6 GHz - 5 GHz (i7-10750H)
* GPU: NVIDIA 4 GB VRAM, 1035 - 1200 MHz (1650 Ti Max-Q)

NVIDIA GPUs with more than 1 GB of VRAM[^3] are highly recommended.  
All CPUs that can run PyTorch should work fine but will be significantly slower.  
The code has not been developed for other brands of GPUs or NPUs,
so it won't take advantage of them.  
SSDs are also highly recommended as this can speed up data loading times.
[^3]: If your NVIDIA GPU has less than 1 GB of VRAM,
set `kwargs={}` in `spectrum_fit.py`&rarr;`initialization`

## Training Times
Training the decoder for 200 epochs with 100,000 spectra
takes around 20 minutes on the GPU and 370 minutes on the CPU.  
Training the encoder for 200 epochs with 10,800 spectra
takes around 3 minutes on the GPU and 43 minutes on the CPU.

## Data Compatibility
The network is designed for data from black hole X-ray binaries from the
Neutron star Interior Composition Explorer (**NICER**) instrument.  
It has **not been tested** on data collected by **other instruments** or for **different sources**,
so retraining will likely be required.  
However, the idea is that this network can be applied to several applications,
so feel free to adapt it and train it to different use cases.
