---
title: "An introduction to Radial basis functions"
publishdate: 2024-03-15T11:59:05+02:00
author: Johann Bouwer
description: A how-to on implementing gradients into the radial basis function
  surrogate model.
draft: false
toc: true
tags: ["Gradients", "RBF", "surrogate models"]
categories: ["python"]
OverviewFig: "test.png"
_build:
  list: always
  publishResources: true
  render: always
---

Good day 👋

🧠 This is the first in a series of blog posts that deal with radial basis function surrogate models 🧑🏽‍🏫. The end goal will be to implement gradient enhanced models that complete powerful transformation procedures, but, we must first understand and implement the simplest version of these models.

Kind regards,

Johann Bouwer

![Alt Text](test.png)

## Introduction

Radial basis function (RBF) surrogate models refer to a family surrogate models that use a linear summation of basis functions. These models are useful when some computationally expensive function, such as a Finite Element simulation, is replaced with a computationally inexpensive model. This simply means we attempt to predict the behaviour of a function that takes a large amount of time solve with a model that can be quickly sampled at numerous locations.

We can express these models as a summation of \\( k \\) basis functions

$$
RBF(\boldsymbol{x}) = \sum_{i=1}^k w_i\phi_i(\boldsymbol{x},
\boldsymbol{c}_i, \epsilon),
$$

where \\( w_i\\) is the weight associated with each basis function \\( \phi_i \\). The basis function is then dependent on the input vector \\( \boldsymbol{x}\\) where the model is being sampled, the centre of each basis function \\( \boldsymbol{c}\\), and the shape parameter \\( \epsilon \\).

These basis functions depend on a distance measure between two points in some \\( n\\)-dimensional space. Common options include:

- Inverse quadratic: \\(\phi(\boldsymbol{x}, \boldsymbol{c}, \epsilon) = \frac
  {1}{1 + \epsilon||\boldsymbol{x} - \boldsymbol{c}||}\\),
- Multi-quadratic: \\(\phi(\boldsymbol{x}, \boldsymbol{c}, \epsilon) = \frac{1}
  {\sqrt{||\boldsymbol{x} - \boldsymbol{c}|| + \epsilon^2}}\\),
- Gaussian: \\(\phi(\boldsymbol{x}, \boldsymbol{c}, \epsilon) = e^
  {-\epsilon||\boldsymbol{x} - \boldsymbol{c}||^2}\\),

The most common option is the Gaussian basis function, so we will implement it here, but it can be easily replaced if a different function is required. A useful characteristic of the Gaussian basis function that will be used later is how conveniently the gradient of the basis function can be found.

## Function Only Training 🔨

Training a RBF model involves taking some same data set \\( X, y \\), where \\(X \\) are the locations of where the function values \\( y \\) are known, and finding the ideal weights of model associated with each basis function. [Scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html#scipy.interpolate.RBFInterpolator) has an efficient and powerful implementation of the RBF model. What this implementation lacks however, is the ability to include sampled gradient information into the model. To understand the `gradient-enhanced` version of model, the `function` version must first be understood.

The `function` version for RBF model is expressed as a system of equations

$$
   RBF(\boldsymbol{x}) = \boldsymbol{\Phi}(\boldsymbol{x}, \boldsymbol{c},
\epsilon)\boldsymbol{w},
$$

where \\( \boldsymbol{w} \\) is the column vector of the weights for each
basis function or kernel and \\(\boldsymbol{\Phi}\\) is now a matrix of the basis vectors

$$
\boldsymbol{\Phi} =
\begin{bmatrix}
    \phi(\boldsymbol{x}_1, \boldsymbol{c}_1, \epsilon) & \phi(\boldsymbol{x}
_1, \boldsymbol{c}_2, \epsilon) & ... & \phi(\boldsymbol{x}_1, \boldsymbol{c}
_k, \epsilon) \\\
   \phi(\boldsymbol{x}_2, \boldsymbol{c}_1, \epsilon) & \phi(\boldsymbol{x}
_2, \boldsymbol{c}_2, \epsilon) & ... & \phi(\boldsymbol{x}_2, \boldsymbol{c}
_k, \epsilon) \\\
    : & : & : & : \\\
    \phi(\boldsymbol{x}_p, \boldsymbol{c}_1, \epsilon) & \phi(\boldsymbol{x}
_p, \boldsymbol{c}_2, \epsilon) & ... & \phi(\boldsymbol{x}_p, \boldsymbol{c}
_k, \epsilon)
\end{bmatrix},
$$

with the size \\( p \times k\\) where \\( p \\) is the number of samples in the data set and \\( k \\) is the number of basis functions. To simplify notation, let \\( [\boldsymbol{x}_1, \boldsymbol{x}_2, ..., \boldsymbol{x}_p]^T \\) be represented by \\( \boldsymbol{X} \\) and let \\( [\boldsymbol{c}_1, \boldsymbol{c}_2, ..., \boldsymbol{c}_k]^T \\) be represented by \\( \boldsymbol{C} \\). This notation simply indicates that we have \\( p \\) samples of \\( \boldsymbol{x} \\) and a \\( k \\) number of centres (\\(\boldsymbol{c} \\)), while transpose operator indicates that the vectors of \\( \boldsymbol{x} \\) and \\( \boldsymbol{c} \\) are stored in the rows of \\( \boldsymbol{X} \\) and \\( \boldsymbol{C} \\), respectively. The `function` version of the RBF model is represented by

$$
\boldsymbol{y} = \boldsymbol{\Phi}(\boldsymbol{X}, \boldsymbol{C}, \epsilon) \boldsymbol{w}.
$$

The number of basis functions \\( \phi_i \\) in the model is dictates the flexibility of model. This is exactly the same as regression formulations, but we represent the polynomial variables using basis functions instead of variations of the polynomial variables/indeterminates (see [here]() for more details on polynomial nomenclature). Meaning, we need to make a choice regarding how much data we have available for the centres \\( \boldsymbol{c}_i \\). If we select the number of basis functions to be equal to the number of samples (\\( p = k \\)) the model will fully interpolate the data. Otherwise, if we use fewer basis function, a least squares regression fit is completed. ⚡

Typically, in the interpolation case, the locations of the basis functions, i.e., \\( \boldsymbol{c}_i \\), are placed at the sample locations \\( \boldsymbol{x} \\). The remaining shape parameter \\(\epsilon \\) is found using either a heuristic or a [cross validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) procedure. In the case where the \\( \Phi \\) matrix is square, i.e., we are constructing an interpolation model with (\\( p = k \\)), we can solve for the weight vector directly as it is the solution to a square system of equations. This is done by setting the sampled function values \\( \boldsymbol{y} \\) equal to the \\( RBF(\boldsymbol{x}) \\) model. On the other-hand, if a regression model is more useful we must make use of the least squares formulation detailed as:

$$
\boldsymbol{\Phi}(\boldsymbol{X}, \boldsymbol{C}, \epsilon)^T \boldsymbol{y} = \boldsymbol{\Phi}(\boldsymbol{X}, \boldsymbol{C}, \epsilon)^T \boldsymbol{\Phi}(\boldsymbol{X}, \boldsymbol{C}, \epsilon) \boldsymbol{w}.
$$

A regression model is usually implemented in the cases where we have"noisy" data, or, a full interpolation matrix is too large to solve such in large data sets.

## Implementing RBFs in Python 🦾
Let's create a basic python class that can fit a RBF model to the data and sample the model at any number of points in the domain.

First, let's set up the constructor:
```python
import numpy as np
from scipy.spatial.distance import cdist

class RBFModel:
    """
    A class to implement a function version for the RBF model
    (FV-RBF)
    """

    def __init__(self, X, y, C=None):
        """
        Constructor for the RBF model.

        Parameters:
        X : np.ndarray
            Input data with shape (n_samples, n_features).
        y : np.ndarray
            Target data with shape (n_samples, ).
        C : np.ndarray, optional
            Centres for the RBF model, default is X. Shape: (n_centres, n_features).
        """
        self.X = X
        self.y = y
        self.C = X if C is None else C

        self._n_samples, self._n_features = self.X.shape
        self._n_centres = self.C.shape[0]

```
The constructor simply assigns the data as attributes in the class and if a specific set of centres are not given we set the centres equal to the samples.

Next, we can create a method that will fit the model to the data. We can also make use of scipys cdist function to find the euclidean distance between all the data points and the centres of the model (the \\(||\boldsymbol{x} - \boldsymbol{c}|| \\) portion of the basis functions). We also want the function to fit either an interpolation model or a regression model based on the number of centres and samples.

```python
import numpy as np
from scipy.spatial.distance import cdist

class RBFModel:

    def __init__(self, X, y, C=None):
        ...

    def FV_fit(self, epsi=1):
            """
            Fit the Function Value Radial Basis Function (FV-RBF) model.

            Parameters:
            epsi : float, optional
                The shape parameter for the RBF. Default is 1.
            """
            self.epsi = epsi

            # Calculate the distance between each pair of points
            dist_matrix = cdist(self.X, self.C, metric='euclidean')

            # Calculate the RBF kernel matrix using the Gaussian basis function
            kernel_matrix = np.exp(-self.epsi * (dist_matrix ** 2))

            # Solve the linear system to find the coefficients/weigths
            if self._n_centres == self._n_samples: #interpolation fit
                self.coefficients = np.linalg.solve(kernel_matrix, self.y)

            else: # regression fit
                self.coefficients = np.linalg.solve(kernel_matrix.T @ kernel_matrix,
                                                    kernel_matrix.T @ self.y)

```
At the moment we simply assume the shape factor to be 1, but for real data some experimentation is needed to find the optimum shape parameter 👏 .

Lastly, we need to be able to sample the model at any given number of points. We can use the pythons inbuilt `__call__` method for this, and have a matrix of the points where we want the model to make predictions as an input to the method

```python
class RBFModel:

    def __call__(self, Xnew):
            """
            Predict using the RBF model.

            Parameters:
            Xnew : np.ndarray
               New points for prediction. Shape: (n_samples, n_features).

            Returns:
            y_pred : np.ndarray
                Predicted output. Shape: (n_samples, 1).
            """
            # Calculate the distance between each input point and the
            # centres of the model
            dist_matrix = cdist(Xnew, self.C, metric='euclidean')

            # Calculate the RBF kernel matrix
            kernel_matrix = np.exp(-self.epsi * (dist_matrix ** 2))

            # Calculate the predicted output
            y_pred = kernel_matrix @ self.coefficients

            return y_pred
```
Again, it is assumed that the vectors \\( \boldsymbol{x} \\) that you want to evaluate over are stored in the rows of the `Xnew` matrix in the `__call__` method.

We now have a basic python class that will fit either a full interpolation RBF model or a regressed model! 🔥

## Numerical Example

Let's test the created python class on a simple 1-dimensional example. We will use the function

$$
f(x) = \sin(10x) + x.
$$

The python code for this example is given by

```python
def Example(x):
    return np.sin(10*x) + x
```

![The 1-dimensional function](Example.png)

Next, we want to generate the data and then fit the model. To generate the data we can use the [pyDOE](https://pypi.org/project/pyDOE/) library, but this line can be replaced with numpy random function. The code below just uses [Latin hypercube sampling](https://en.wikipedia.org/wiki/Latin_hypercube_sampling) to cover the domain of interest. In this example, we assume that the domain is \\( x \in [1, 7] \\). After sampling the function at the data locations, we then create the model using the `RBFModel` class and use the `FV_fit` method to fit the model to the data.

```python
from pyDOE import lhs
X = lhs(1, 7, criterion='m') #samples locations
y = Example(X) #sample the function

model = RBFModel(X, y) #create the model object
model.FV_fit(epsi = 1) #fit the model
```

We can now sample the model across the entire domain. To do this we create a column vector of the locations we want the model to make predictions at, and then pass this vector into the model (which uses the `__call__` method we wrote behind the scenes).

```python
X_pred = np.linspace(0, 1, 100).reshape(-1,1) #locations for predictions
y_pred = model(X_pred) #model predictions
```

Below is a plot of the model predictions overlaid with target functions, as well as the generated data that the model is constructed on.

![The RBF Model overlaid with the target function](epsi1.png)

We can see using only 7 samples we can already create a RBF model that gives accurate predictions. Pretty cool, right? 🕶️

### Impact of the Shape Parameter

At the moment the shape parameter is left at 1. This creates the obvious question of what the impact the shape parameter has on the model. If we refit the model using \\( \epsilon = 10, 100 \\), we obtain the following results:

![The RBF Model overlaid with the target function](epsi_impact.png)

As the name implies, the shape parameter impacts the overall shape of the model. A larger value creates a "steeper" basis function, which in turn means the model becomes "bumpy", while a smaller value creates a "shallower" basis function and a "smoother" model.

> **Note:** In some implementations of the RBF model the shape parameter is placed as a divider, i.e., \\( \frac{||x - c||} {\epsilon}\\), in which case the inverse behaviour is true, meaning larger is a smoother model and smaller is a bumpy model.

It is also important to note that for any shape parameter the model still fully interpolates, i.e., it passes thorough, all the sampled data. Therefore, to find the optimum shape parameter the data is separated into training and testing sets and the value that performs the best on the testing set is used to construct the model. This is referred to as a cross-validation strategy.

## Conclusion

This post presents a simple python class to implement a function value radial basis function model. From this simple class more powerful versions of the model can be constructed. In future posts we will address:

- The case where gradient information is available, i.e., the `gradient-enhanced` model.
- Higher dimensional problems.
- How to address the isotropic nature of radial basis function.

As always, thanks for reading! 🧙‍♂️
