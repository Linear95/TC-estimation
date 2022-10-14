# Total Correlation (TC) Estimation

This repo contains the implementation of the NeurIPS 2020 workshop paper: [Estimating Total Correlation with Mutual Information Bounds](https://openreview.net/pdf?id=UsDZut_p2LG).

Mutual information (MI) is a fundamental measurement of variable correlation:
$$\left \{ \begin{array}
\mathcal{I}(\bm{x}; \bm{y}) = \mathbb{E}_{p(\bm{x}, \bm{y}) [\log \frac{p(\bm{x}, \bm{y})}{p(\bm{x}) p(\bm{y})}]
\end{array} \right .$$


We designed two 

## Introduction


$$\left \{ \begin{array}{rll}
\nabla \cdot \mathbf{E} &=& \displaystyle \frac {\rho} {\varepsilon_0} \\
\nabla \cdot \mathbf{B} &=& 0 \\
\nabla \times \mathbf{E} &=& \displaystyle - \frac{\partial \mathbf{B}} {\partial t} \\
\nabla \times \mathbf{B} &=& \displaystyle \mu_0\mathbf{J} + \mu_0 \varepsilon_0 \frac{\partial \mathbf{E}} {\partial t}  \\
\end{array} \right .$$

We know that mutual information measures the dependency between two variables:
$$ \begin{equation}
\mathcal{I}(\bm{x}; \bm{y}) = \mathbb{E}_{p(\bm{x}, \bm{y}) [\log \frac{p(\bm{x}, \bm{y})}{p(\bm{x}) p(\bm{y})}]
\end{equation}
$$

Run 'test.py' to check the examlpe of total correlation estimator.
