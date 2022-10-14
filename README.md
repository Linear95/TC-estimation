# Total Correlation (TC) Estimation

This repo contains the implementation of the NeurIPS 2020 workshop paper: [Estimating Total Correlation with Mutual Information Bounds](https://openreview.net/pdf?id=UsDZut_p2LG).

<p align="center">
  <img src="https://github.com/Linear95/TC-estimation/blob/master/figures/est_results.png" width="70%" height="70%">
</p>
The above figure shows the estimation performance of our TC-Tree and TC-Line estimators based on different mutual information estimators.

## Introduction

We implement the `TCLineEstimator` and `TCTreeEstimator` in [`tc_estimators.py`](https://github.com/Linear95/TC-estimation/blob/master/tc_estimators.py).

Both TC estimators are based on MI estimators ([NWJ](https://media.gradebuddy.com/documents/2949555/12a1c544-de73-4e01-9d24-2f7c347e9a20.pdf), [MINE](http://proceedings.mlr.press/v80/belghazi18a), [InfoNCE](https://arxiv.org/pdf/1807.03748.pdf), [CLUB](https://arxiv.org/abs/2006.12013)) in [`mi_estimators.py`](https://github.com/Linear95/TC-estimation/blob/master/mi_estimators.py).

In [`tc_estimation.ipynb`](https://github.com/Linear95/TC-estimation/blob/master/tc_estimation.ipynb), we conduct a toy simulation to test the estimation ability of our TC estimators.


## Method

Mutual information (MI) is a fundamental measurement of correlation between two variables:
<p align="center">
  <img src="https://github.com/Linear95/TC-estimation/blob/master/figures/eq_mi_definition.png"  height="40">
</p>

Total correlation (TC) is an extension of MI for multi-variate scenarios:
<p align="center">
  <img src="https://github.com/Linear95/TC-estimation/blob/master/figures/eq_tc_definition.png" height="40">
</p>

We introduce two calculation paths to decomposite the total correlation into mutual information terms:

- Line-like decomposition:
<p align="center">
  <img src="https://github.com/Linear95/TC-estimation/blob/master/figures/eq_line_decomp.png" height="18">
</p>


- Tree-like decomposition:
<p align="center">
  <img src="https://github.com/Linear95/TC-estimation/blob/master/figures/eq_tree_decomp.png" height="22">
</p>

The calculation paths are demonstrated in the following figure:
<p align="center">
  <img src="https://github.com/Linear95/TC-estimation/blob/master/figures/decomp_scheme.png" height="120">
</p>

## Citation
Welcome to cite our paper if the code is useful:

```latex
@article{cheng2020estimating,
  title={Estimating total correlation with mutual information bounds},
  author={Cheng, Pengyu and Hao, Weituo and Carin, Lawrence},
  journal={arXiv preprint arXiv:2011.04794},
  year={2020}
}
```

