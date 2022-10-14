# Estimating Total Correlation with Mutual Information Estimators

In this repo, we extend the mutual information (MI) estimation into multi-variate scenarios. Details about our method is summarized in our NeurIPS 2020 workshop: [Estimating Total Correlation with Mutual Information Bounds](https://openreview.net/pdf?id=UsDZut_p2LG).

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
