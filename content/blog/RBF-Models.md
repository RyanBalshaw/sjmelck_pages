---
title: "RBF Models"
publishdate: 2023-07-08T11:59:05+02:00
author: Johann Bouwer
description: A how-to on implementing gradients into the radial basis function
  surrogate model.
draft: true
toc: true
tags: ["Gradients", "RBF", "surrogate models"]
categories: ["python"]
_build:
  list: always
  publishResources: true
  render: always
---

# A basic python implementation of Function Value Radial Basis Function surrogate models (FV-RBF)
![Alt Text](test.png)

## Table of Contents

1. [Introduction](#introduction)
2. [Function Only Training](#training-the-model)
3. [Python Implementation](#python-implementation)
4. [Numerical Example](#numerical-example)

## Introduction

Radial basis function surrogate models refer to a family surrogate models 
that use a linear summation of basis functions. These models are useful when 
some computationally function, such as a Finite Element simulation, is 
replaced with a computationally inexpensive model. 

We can express these models as a summation of \\( k \\) basis functions
$$
RBF(\boldsymbol{x}) = \sum_{i=1}^k w_i\phi_i(\boldsymbol{x}, 
\boldsymbol{c}_i, \epsilon),
$$
where \\( w_i\\) is the weight associated with each basis function \\( 
\phi_i \\). The basis function is then dependent on the input vector 
\\( \boldsymbol{x}\\) where the model is being sampled, the centre of each 
basis function \\( \boldsymbol{c}\\), and the shape parameter \\( \epsilon \\).

These basis functions depend on a distance measure between two points in 
some \\( n\\)-dimensional space. Common options include:

- Inverse quadratic: \\(\phi(\boldsymbol{x}, \boldsymbol{c}, \epsilon) = \frac
  {1}{1 + \epsilon||\boldsymbol{x} - \boldsymbol{c}||}\\),
- Multi-quadratic: \\(\phi(\boldsymbol{x}, \boldsymbol{c}, \epsilon) = \frac{1}
  {\sqrt{||\boldsymbol{x} - \boldsymbol{c}|| + \epsilon^2}}\\),
- Gaussian: \\(\phi(\boldsymbol{x}, \boldsymbol{c}, \epsilon) = e^
  {-\epsilon||\boldsymbol{x} - \boldsymbol{c}||^2}\\),

The most common option is the Gaussian basis function, which is useful for 
the gradient enhanced case as the math is far more convenient.

## Function Only Training

Training a RBF model involves taking some same data set \\( X, y \\), where 
\\(X \\) are the locations of where the function values \\( y \\) are known, 
and finding the ideal weights of model associated with each basis function.
[SCIPY](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html#scipy.interpolate.RBFInterpolator) 
has an efficient and powerful implementation of the RBF model. What this 
implementation lacks however, is the ability to include sampled gradient 
information into the model. To understand the gradient enhanced version of 
model, the function version must first be understood.

The RBF model is expressed as a system of equations 
$$
   RBF(\boldsymbol{x}) = \boldsymbol{\Phi}(\boldsymbol{x}, \boldsymbol{c}, 
\epsilon)\boldsymbol{w},
$$
where \\( \boldsymbol{w} \\) is the column vector of the weights for each 
basis function or kernel, and \\(\boldsymbol{\Phi}\\) is now matrix
$$
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
with the size \\( p \times k\\) where \\( p \\) is the number of samples in the 
data set and \\( k \\) is the number of basis functions. 

The number of basis functions in the model is dictates the flexibility of 
model, meaning, if we select the number of basis functions to be equal to 
the number of samples (\\( p = k \\)) the model will fully interpolate the 
data. Otherwise, if we use fewer basis function, a least squares regression 
fit is completed. 

In the case where the \\( \Phi \\) matrix is square, i.e we are constructing 
an interpolation model, we can solve for the weight vector directly. This is 
done by setting the sampled function values \\( y \\) equal to the \\( \Phi \\) 
matrix where the \\( x_1, x_2, ..., x_p \\) values are the locations in the 
dataset \\( X \\)

$$
y = \Phi(X, c, \epsilon) w.
$$
Typically in the interpolation case the locations of the basis functions, i.e 
\\( c \\) are placed at the sample location \\( x\\). The reaming \\( 
\epsilon \\) is found using a heuristic or 
[cross validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)).

On the other-hand, if a regression model is more useful we must make use of the 
least squares formulation

$$
\Phi(X, c, \epsilon)^Ty = \Phi(X, c, \epsilon)^T\Phi(X, c, \epsilon) w
$$

## Python Implementation 
Lets create a basic python class that can fit a RBF model to the data and 
sample the model at any number of points in the domain.

First, lets set up the constructor
```python
import numpy as np
class RBFModel:
    """
    A class to implement a Function Value based Radial Basis Function model 
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
        
        return
```
The constructor simply assigns the data as attributes in the class and if a 
specific set of centres are not given we set the centres equal to the samples. 
Next we can create a method that will fit the model to the data. We can also make 
use of scipys cdist function to find the euclidean distance between all the 
data points and the centres of the model (the \\(||\boldsymbol{x} - 
\boldsymbol{c}|| \\) portion of the basis functions). We also want the 
function to fit either an interpolation model or a regression model based on 
the number of centres and samples.
```python
class RBFModel:
    
    :
    :
    
    def FV_fit(self, epsi=1):
            """
            Fit the Function Value Radial Basis Function (FV-RBF) model.
    
            Parameters:
            epsi : float, optional
                The shape parameter for the RBF. Default is 1.
            """
            self.epsi = epsi
            
            from scipy.spatial.distance import cdist
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

            return
```
At the moment we simply assume the shape factor to be 1, but for real data 
some experimentation is needed to find the optimum shape parameter.

Lastly, we need to be able to sample the model at any given number of points.
We can use the pythons inbuilt ''\_\_call\_\_'' keyword for this, and have a 
matrix 
of the points where we want the model to make predictions as an input to the 
method

```python
class RBFModel:
    
    :
    :
    
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
We now have a basic python class that will fit either a full interpolation 
RBF model or a regressed model.

## Numerical Example

Let's test the created python class on a simple 1-dimensional example. We 
will use the function
$$
f(x) = \sin(10x) + x.
$$

```python
def Example(x):
    
    return np.sin(10*x) + x
```
figure

create model [pyDOE](https://pypi.org/project/pyDOE/)
```python
from pyDOE import lhs
X = lhs(1, 7, criterion='m')
y = Example(X)

model = RBFModel(X, y)
model.FV_fit(epsi = 1)
```

sample model 
```python
X_pred = np.linspace(0, 1, 100).reshape(-1,1)
y_pred = model(X_pred)
```
figure

### Impact of the Shape Parameter 

- show impact of epsilon
- always interpolates, shape is differnt
- steeper shallower





