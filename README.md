
# Pancreatic cancer segmentation in unregistered multi-parametric MRI with adversarial learning and multi-scale supervision

This repository holds the PyTorch code of our Neurocomputing paper *Pancreatic cancer segmentation in unregistered multi-parametric MRI with adversarial learning and multi-scale supervision*. 

All the materials released in this library can **ONLY** be used for **RESEARCH** purposes and not for commercial use.

The authors' institution (**Biomedical Image and Health Informatics Lab, School of Biomedical Engineering, Shanghai Jiao Tong University**) preserve the copyright and all legal rights of these codes.


## Author List

Jun Li, Chaolu Feng, Qing Shen, Xiaozhu Lin, Xiaohua Qian


## Abstract

Automated pancreatic cancer segmentation is crucial for successful clinical aid diagnosis and surgical planning. However, the tiny size and inconspicuous boundaries of pancreatic cancer lesions lead to poor segmentation performance with single-modality imaging. The commonly used registration-based multimodal fusion strategies may not only introduce uncertainties arising from registration, but also underutilize the complementary information between different modalities. Thus, to achieve different modalitybased tumor segmentation, we propose for the ﬁrst time a registration-free multi-modal and multiscale adversarial segmentation network (MMSA-Net), which consist of a shared encoder and a dual decoder. Speciﬁcally, MMSA-Net combine two complementary modules, inter-modality adversarial learning and intra-modality multi-scale adversarial supervision, to obtain mode-speciﬁc segmentation results while facilitating multi-modal fusion. The inter-modality adversarial learning module facilitates the fusion of modality-shared features among different modalities by strengthening the similarity of features extracted by the shared encoder. The intra-modality multi-scale adversarial supervision emphasizes modality-speciﬁc features in different decoding paths, inherits and fuses modality-shared features while preserving the feature speciﬁcity of each modality, thus outputting competitive modality-speciﬁc segmentation results for each modality. Quantitative and qualitative experimental results on multiparametric MRI pancreatic cancer data show that our method can effectively improve the performance of multi-modal segmentation. The method proposed in this work is expected to be another potential paradigm for addressing multi-modal segmentation tasks in addition to registration. Our source codes will be released at https://github.com/SJTUBME-QianLab/PancreaticCancer_Seg_AdvLearning, once the manuscript is accepted for publication.


## Requirements

* `pytorch 1.1.0`
* `numpy 1.17.2`
* `python 3.6.1`



## Citing the Work

If you find our code useful in your research, please consider citing:

```
@article{LI2022310,
title = {Pancreatic cancer segmentation in unregistered multi-parametric MRI with adversarial learning and multi-scale supervision},
journal = {Neurocomputing},
volume = {467},
pages = {310-322},
year = {2022},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2021.09.058},
url = {https://www.sciencedirect.com/science/article/pii/S0925231221014363},
author = {Jun Li and Chaolu Feng and Qing Shen and Xiaozhu Lin and Xiaohua Qian},
keywords = {Pancreatic cancer segmentation, Multi-parametric MRI, Multi-scale adversarial supervision, Adversarial learning}
}
```

## Contact

For any question, feel free to contact

```
Jun Li : dirk_li@sjtu.edu.cn
```

## Acknowledgements

This code is developed on the code base of [SegAN](https://github.com/YuanXue1993/SegAN). Many thanks to the authors of this work. 