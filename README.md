# Diffusion Downscaling

## Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Technical Details](#technical)
- [Get Help](#help)

## About <a name = "about"></a>

Machine-learning based statistical downscaling using novel diffusion models. 

We train diffusion models on reanalysis data in order to probabilistically downscale GCM outputs. This can be used to generate high resolution precipitation maps in different climates to evaluate the effects of climate change. 

## Getting Started <a name = "getting_started"></a>

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Conda
- Python 3.9
- NVIDIA CUDA drivers (compatible with >= torch 2.0)

Install Anaconda or Miniconda. Set up your conda environment by executing the following command in the terminal, assuming `my_env` is the name of your conda environment:

```
conda create -n "my_env" python=3.9
conda activate my_env
```

### Installing

Installation of the library can then be done by navigating to the `diffusion_downscaling` directory and running:
```
pip install -e .
```

Different CUDA drivers / NVIDIA related libraries may lead to different pytorch requirements. The main thing is to ensure torch > 2.0. For instance, on the Fathom GPUs we require:
```
pip install torch==2.1.1 torchvision==0.16.1
```

## Usage <a name = "usage"></a>

There are several scripts that execute the main functionality of the code. The most important of these are:

- `scripts/train.py` for model training
- `scripts/inference.py` for sampling and evaluation

Both of these scripts take paths to configuration files as command line args. These configuration files offer a range of options and customisability for both training and evaluation. 

### Preprocessing

Datasets are built by combining data from multiple data sources (MSWEP, MSWX and ERA5); we select the desired spatial and temporal domain, conservatively remap the datasets to the desired resolutions, and merge all processed data into a a single netCDF file. 

`scripts/build_dataset.py` performs all of this processing by wrapping the command line tool `cdo` and merging all the various data sources. The user can select the desired time range and spatial domain, as well as which variables to select from each data source. 

### Training

Model training through `scripts/train.py` requires a config file, with examples found in `scripts/configs/configs`. This config files specifies:
- the model architecture, 
- the training setup (such as epochs, batch size)
- the training data and variables
- further settings such as optimiser parameters and floating point precision. 

See [example.py](scripts/configs/configs/example.py) for an annotated example of the possible options. 
 

Example usage:
```
cd scripts
python train.py -c configs/configs/example.py
```
Optionally resume training from a checkpoint:
```
python train.py -c configs/configs/example.py -C path/to/checkpoint.ckpt
```

Each config file requires the user to specify the predictor and predictand variables, as well as how to scale these variables. For reusability, this configuration is split out into separate variable config files, with examples under [scripts/configs/variables](scripts/configs/variables).

### Inference

Model inference requires two config files: 

- The original config file used for training, since this contains the model architecture and data required to load the model and data utilities.
- A sampling config file, specifying what data to use for coarse predictors (dataset / geographical domain) as well as what sampling parameters to use.

Examples of the sampling config files can be found under `scripts/configs/sampling`. See [example.py](scripts/configs/sampling/example.py) for an annotated example.

Sampling files support running a product over lists of parameters to help run parameter sweeps and find the best sampling configurations. 

Example usage:
```
python inference.py -c configs/configs/example.py -s configs/configs/example.py
```

## Technical Details <a name = "technical"></a>

Diffusion models are a novel class of generative models. These models generally learn to sample from the probability distribution over its outputs, often given some conditional information. Diffusion models have recently surpassed the performance of earlier Machine Learning techniques, such as Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs). This repository implements the training of diffusion models, as well as several alternatives that serve as benchmarks for the model performance.

### Data Processing and Loading

The data processing and loading code can be found under [diffusion_downscaling/data/](src/diffusion_downscaling/data/). Data processing, which involves using the command line tool `cdo` to resample fields to the desired grid spacing, as well as merging large amounts of daily data, is done in [data/dataset_building.py](src/diffusion_downscaling/data/dataset_building.py]) and [data/processing.py](src/diffusion_downscaling/data/processing.py]). The script in [scripts/build_dataset.py](scripts/build_dataset.py) utilises this code to create a single merged NetCDF file with all the coarse predictors, high-res static predictors, and downscaled predictand fields. 

Data loading, on the other hand, provides a bridge between the raw merged data and the machine learning models. This has two components: 

1. Data scaling
2. PyTorch Dataloaders for high-performance loading of the data from disk into the models during training and inference. 

The code for scaling the data is found in [data/scaling.py](src/diffusion_downscaling/data/scaling.py]). This provides a range of simple classes for scaling each predictor/predictand between [-1, 1]. This particular range is a requirement for the output fields due to the nature of diffusion models. On the other hand, since the inputs fields and output fields should be scaled similarly so that one does not dominate the other as inputs to the model during training, we apply the same scaling utilities to predictors as well. 

The dataloading code, which sublasses the PyTorch `Dataset` object to ensure good loading performance, is found in [data/data_loading.py](src/diffusion_downscaling/data/data_loading.py]). We implement several datasets for different functionalities:

1. Upsampling xarray datasets; this loads in inputs and outputs, and upsamples and coarse predictors to the native downscaled resolution. Defaults to a fixed domain.
2. Variable location; extends the Upsampling dataset but randomly selects sub-patches of the field during training. This is designed for training over large domains that cannot / should not be fit into the model at once.
3. Fixed location; extends the Variable dataset but fixes the region it loads in. Can be used for training or evaluating over a fixed region.

### Training and Validation

We use the PyTorch Lightning framework for training and validation. While this likely incurs some overhead, it greatly simplifies the code required to run training with a number of conveniences, such as automatic logging, model checkpointing, and various model optimisation presets. 

PyTorch Lightning requires the user to specify how to compute the loss required to train a model, as well how to evaluate the model. This is done under [diffusion_downscaling/lightning/models](src/diffusion_downscaling/lightning/models), with various different models: [models/deterministic.py](src/diffusion_downscaling/lightning/models/deterministic.py), [models/gan.py](src/diffusion_downscaling/lightning/models/gan.py), and [models/diffusion.py](src/diffusion_downscaling/lightning/models/diffusion.py).

### Diffusion Models

There are a wide variety of ways these statements can be formalised. We follow Karras et al. (2022), which presents the EDM formalism. This provides a range of improvements for training and sampling from diffusion models:

1) Preconditioning and training; these are implemented in [diffusion_downscaling/lightning/models/karras_diffusion.py](src/diffusion_downscaling/lightning/models/karras_diffusion.py), which sets out how to process the noisy inputs to the diffusion model and how to train the model to learn the gradient. 
2. Efficient sampling; the main utilities for this are given under [diffusion_downscaling/sampling/k_sampling.py](src/diffusion_downscaling/sampling/k_sampling.py), using a range of tricks to improve the sampling efficiency and quality. 


### Benchmarks

Also provided is code used to serve as benchmarks for our best performing diffusion model.

1) Regression-based CNN (using the same UNet backbone), implemented in [lightning/models/deterministic.py](src/diffusion_downscaling/lightning/models/deterministic.py). 

2) GAN, modified UNet backbone. Modified model code in [diffusion_downscaling/yang_cm/cwgan_patch.py](src/diffusion_downscaling/yang_cm/cwgan_patch.py), and training code here: [lightning/models/gan.py](src/diffusion_downscaling/lightning/models/gan.py)

3) Variance-preserving SDE diffusion model, an earlier formulation of the diffusion process that Karras improves upon. Code can be found in [diffusion_downscaling/lightning/models/karras_diffusion.py](src/diffusion_downscaling/lightning/models/karras_diffusion.py) under the `VPDenoiser`.

### Inference and Sampling

There are a range of utilities provided to implement the sampling algorithms for diffusion models, as well as a high-level class to run sampling and an optional evaluation step. 

Various integration techniques and noise schedule utilities, adapted from Karras el al., can be found in [diffusion_downscaling/sampling/k_sampling.py](src/diffusion_downscaling/sampling/k_sampling.py). The best performing sampling configuration has proven to be dpm2_heun, a customised integrator for the diffusion process. The Karras noise schedule has also been slightly adapted. Various configuration files with the specifics of the integration parameters and the noise schedule can be found in the example config file, [example.py](scripts/configs/sampling/example.py).

A high-level class for running sampling and optional evaluation is `Sampler` in [diffusion_downscaling/sampling/sampling.py](src/diffusion_downscaling/sampling/sampling.py). This can be used to run sampling from all the model classes in the repository (various diffusion models, GANs, regression models). It can also be used to run a customisable evaluation step given observation data. 

### Evaluation

Finally, we provide a wide range of utilities and helper classes for evaluating trained models at the inference stage, grouped under [diffusion_downscaling/evaluation](src/diffusion_downscaling/evaluation). These have been wrapped in a few utilities which take various options for specifying which metrics to compute as input. A preset evaluation routine containing these is optionally run at the end of `inference.py`. 

The preset utilities can be found in [diffusion_downscaling/evaluation/utils.py](src/diffusion_downscaling/evaluation/utils.py). These can compute a range of metrics with respect to a set of provided observations:

- Pixel-level metrics, such as MSE, bias, CRPS, etc., available at multiple spatial pooling scales
- Sample quality metrics such as comparison of the radially averaged log spectral density of samples
- Probabilistic metrics, looking at expected calibration error, rank histogram errors, event ratios between observed and sampled fields, etc.

The legwork for these metrics are done in three files: [eval.py](src/diffusion_downscaling/evaluation/eval.py), [metrics.py](src/diffusion_downscaling/evaluation/metrics.py), [spatial_aggregation.py](src/diffusion_downscaling/evaluation/spatial_aggregation.py).

In addition to this, a wide variety of plotting tools are provided in [plotting.py](src/diffusion_downscaling/evaluation/plotting.py). These are also run at the end of the `inference.py` script. 
## Maintainers<a name = "help"></a>

Alex Saoulis (a.saoulis@fathom.global)

Sebastian Moraga (s.moraga@fathom.global)
