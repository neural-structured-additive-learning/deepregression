
# deepregression

[![R build status](https://github.com/davidruegamer/deepregression/workflows/R-CMD-check/badge.svg)](https://github.com/davidruegamer/deepregression/actions)
[![Codecov test coverage](https://codecov.io/gh/davidruegamer/deepregression/branch/master/graph/badge.svg)](https://codecov.io/gh/davidruegamer/deepregression?branch=master)
[![MIT license](http://img.shields.io/badge/license-MIT-brightgreen.svg)](http://opensource.org/licenses/MIT)

# Installation

To install the package, use the following command:
``` r
devtools::install_github("davidruegamer/deepregression")
```
Note that the installation requires additional packages (see below) and their installation is currently forced by `deepregression`.

# Requirements

The requirements are given in the `DESCRIPTION`. If you load the package manually using `devtools::load_all`, make sure the following packages are availabe:

  - Matrix
  - dplyr
  - keras
  - mgcv
  - reticulate
  - tensorflow
  - tfprobability

If you set up a Python environment for the first time, install `reticulate` and run the `check_and_install` function from the `deepregression` package. This tries to install miniconda, TF 2.1, TFP 0.9 and keras 2.4.3, which seems to be the most reliable setup for `deepregression` at the moment.

# How to cite this?

For the methodology, please cite the following preprint:

    @article{rugamer2020unifying,
      title={Semi-Structured Deep Distributional Regression: Combining Structured Additive Models and Deep Learning},
      author={R{\"u}gamer, David and Kolb, Chris and Klein, Nadja},
      journal={arXiv preprint arXiv:2002.05777},
      year={2020}
    }
    
For the software, please cite:

    @article{rugamer2021deepregression,
      title={deepregression: a Flexible Neural Network Framework for Semi-Structured Deep Distributional Regression}, 
      author={David R{\"u}gamer and Ruolin Shen and Christina Bukas and Lisa Barros de Andrade e Sousa and Dominik Thalmeier and Nadja Klein and Chris Kolb and Florian Pfisterer and Philipp Kopper and Bernd Bischl and Christian L. M{\"u}ller},
      year={2021},
      eprint={2104.02705},
      archivePrefix={arXiv},
      journal={arXiv preprint arXiv:2104.02705}
    }

# How to use this?

See [the tutorial](https://arxiv.org/pdf/2104.02705.pdf) for a detailed introduction.

# Python version

A Python version of the package is available [here](https://github.com/HelmholtzAI-Consultants-Munich/PySDDR). 

# Related literature

The following works are based on the ideas implemented in this package:

* [Original Semi-Structured Deep Distributional Regression Proposal](https://arxiv.org/abs/2002.05777)
* [Neural Mixture Distributional Regression](https://arxiv.org/abs/2010.06889)
* [Deep Conditional Transformation Models](https://arxiv.org/abs/2010.07860)
* [Semi-Structured Deep Piecewise Exponential Models](https://arxiv.org/abs/2011.05824)
* [Combining Graph Neural Networks and Spatio-temporal Disease Models to Predict COVID-19 Cases in Germany](https://arxiv.org/abs/2101.00661)

# People that contributed

Many thanks to following people for helpful comments, issues, suggestions for improvements and discussions: 

* Andreas Bender
* Christina Bukas
* Christian Mueller
* Nadja Klein
* Chris Kolb
* Philipp Kopper
* Fabian Scheipl
* Max Schneider
* Ruolin Shen
* Almond Stoecker
* Dominik Thalmeier
* Kang Yang
