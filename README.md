# Deep Learning for MRI Water-Fat Separation

<p align="center">
  <a href='https://arxiv.org/abs/2412.08741' style='padding-left: 0.5rem;'>
    <img src='https://img.shields.io/badge/arXiv-2403.13319-brightgreen' alt='arXiv'>
  </a>
  <a href='https://link.springer.com/article/10.1007/s00330-023-09576-2' style='padding-left: 0.5rem;'>
    <img src='https://img.shields.io/badge/Media.Journal-2025.103503-red' alt='Media Journal'>
  </a>
  <a href='https://link.springer.com/10.1007/s00330-024-11164-x' style='padding-left: 0.5rem;'>
    <img src='https://img.shields.io/badge/Media.Journal-2025.103503-red' alt='Media Journal'>
  </a>
  <a href='https://doi.org/10.1016/j.media.2025.103811' style='padding-left: 0.5rem;'>
    <img src='https://img.shields.io/badge/Media.Journal-2025.103503-red' alt='Media Journal'>
  </a>
</p>

## Intro
Deep learning shows promise for improving MRI-based fat quantification in terms of processing times, scan time acceleration, and artifact removal (i.e., fat-water swaps, noise-related bias, etc).

In this repository, several deep learning models for MRI water-fat separation, based on different approaches, are proposed.

### Description of the codes in the main folder
1. Supervised training:
	- ```train-IDEAL-TEaug.py```: Training of water-fat separation models considering Graph Cuts results as labels, as described in Meneses et al., 2023.
	- ```train-IDEAL-TEaug.py```: Training of water-fat separation models using a physics-based data augmentation process, as described in Meneses et al., 2024.
2. Unsupervised training:
	- ```train-IDEAL-unsup.py```: Training of water-fat separation models using a purely physics-based loss; uncertainty awareness can also be enabled, as described in Meneses et al., 2025.
	- ```train-IDEAL-single.py```: Self-supervised model for water-fat separation on a single sample; the use of bipolar gradient-echo multi-echo MRI can also be enabled.
3. Generative modeling:
	- ```train-IDEAL-GAN.py```: Training of a Physics-based Variational Autoencoder (PI-VAE) to synthesize gradient-echo multi-echo MRI along with their respective quantitative maps.
	- ```train-ldm.py```: Training of a Latent Diffusion Model (LDM) to synthesize plausible latent spaces based on a pre-trained PI-VAE.
	- ```gen_LDM_dataset.py```: Dataset generation using a previously trained PI-LDM.
4. Evaluation:
	- ```ROI-analysis.py```: PDFF bias assessment considering liver ROIs.
	- ```ROI-realPhantom.py```: PDFF bias assessment considering the [Multi-site and multi-vendor fat-water phantom dataset](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.26228).

### Current Jupyter notebooks:
1. ```bipolar-test-unsup.ipynb```: Try a subject-specific self-supervised model (trained using ```train-IDEAL-single.py```).
2. ```try-models.ipynb```: Notebook for testing all the developed DL models for my PhD thesis:
	- U-Net (trained using ```train-sup.py```)
	- MDWF-Net (trained using ```train-sup.py```)
	- VET-Net (trained using ```train-IDEAL-TEaug.py```)
	- AI-DEAL (trained using ```train-IDEAL-unsup.py```)

### Recommended dependencies - can be installed using ```conda env create -f environment.yml```:
- ```tensorflow```: 2.8.1
- ```tensorflow-addons```: 0.16.1
- ```tensorflow-probability```: 0.16.0
- ```matplotlib```: 3.4.2
- ```scikit-image```: 0.19.3
- ```xlsxwriter```
- ```tqdm```
- ```oyaml```
- ```einops```

