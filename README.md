# Estimating Total Correlation with Mutual Information Bounds

In this repo, we extend the mutual information (MI) estimation into multi-variate scenarios. Details about our method is summarized in our NeurIPS 2020 workshop: [Estimating Total Correlation with Mutual Information Bounds](https://openreview.net/pdf?id=UsDZut_p2LG).

## Introduction

We know that mutual information measures the dependency between two variables:
$$ \mathcal{I}(\bm{x}; \bm{y}) = \mathbb{E}_{p(\bm{x}, \bm{y}) [\log \frac{p(\bm{x}, \bm{y})}{p(\bm{x}) p(\bm{y})}] $$

Run 'test.py' to check the examlpe of total correlation estimator.