# Deep Learning for MRI Water-Fat Separation

For additional information, please refer to the publications below:

- [Liver PDFF estimation using a multi-decoder water-fat separation neural network with a reduced number of echoes](https://link.springer.com/article/10.1007/s00330-023-09576-2), Juan Pablo Meneses, Cristobal Arrieta, Gabriel della Maggiora, Cecilia Besa, Jesús Urbina, Marco Arrese, Juan Cristóbal Gana, Jose E. Galgani, Cristian Tejos & Sergio Uribe.

- [Unbiased and reproducible liver MRI-PDFF estimation using a scan protocol-informed deep learning method](https://link.springer.com/10.1007/s00330-024-11164-x), Juan P. Meneses, Ayyaz Qadir, Nirusha Surendran, Cristobal Arrieta, Cristian Tejos, Marcelo E. Andia, Zhaolin Chen & Sergio Uribe.

- [A Physics-based Generative Model to Synthesize Training Datasets for MRI-based Fat Quantification](https://arxiv.org/abs/2412.08741), Juan P. Meneses, Yasmeen George, Christoph Hagemeyer, Zhaolin Chen, Sergio Uribe

### Description of the codes in the main folder
1. ```train-IDEAL-single.py```: self-supervised model for water-fat separation using bipolar gradient-echo multi-echo MRI

### Current Jupyter notebooks:
1. ```bipolar-test-unsup.ipynb```: Try a subject-specific self-supervised model (trained using ```train-IDEAL-single.py```).

### Recommended dependencies:
- ```tensorflow```: 2.8.1
- ```tensorflow-addons```: 0.16.1
- ```tensorflow-probability```: 0.16.0
- ```matplotlib```: 3.4.2
- ```scikit-image```: 0.19.3
- ```xlsxwriter```
- ```tqdm```
- ```oyaml```
- ```einops```

