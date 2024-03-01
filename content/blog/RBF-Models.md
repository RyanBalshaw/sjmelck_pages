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

# Gradient Enhanced Radial Basis Function Surrogate Models (GE-RBF)

## Table of Contents

1. [Introduction](#introduction)
2. [Function Only Training](#training-the-model)
3. [Adding the Gradients](#adding-the-gradients)
4. [Python Implementation](#python-implementation)

![Alt Text](test.png)

## Introduction

Radial basis function surrogate models refer to a family surrogate models 
that use a linear summation of basis functions. These models are useful when 
some computationally function, such as a Finite Element simulation, is 
replaced with a computationally inexpensive model. We can express these 
models as a summation of \\( k \\) basis functions
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
and finding the ideal weights of model.
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
where \\( p \\) is the number of samples in the data set.

## Adding the Gradients

$$
\frac{d\phi(\boldsymbol{x}, \boldsymbol{c}, \epsilon)}{d\boldsymbol{x}} = 
-2\epsilon \phi(\boldsymbol{x}, \boldsymbol{c}, \epsilon)(\boldsymbol{x} - 
\boldsymbol{c}),
$$

## Python Implementation

```python
from scipy.spatial.distance import cdist


class RBFModel:
    """
    A class to implement Radial Basis Function (RBF) models including 
    Function Value (FV), Gradient Enhanced (GE), and Gradient Only (GO) models.
    """

    def __init__(self, X, y, C=None, dy=None):
        """
        Constructor for the RBF model.

        Parameters:
        X : np.ndarray
            Input data with shape (n_samples, n_features).
        y : np.ndarray
            Target data with shape (n_samples, ).
        C : np.ndarray, optional
            Centres for the RBF model, default is X. Shape: (n_centres, n_features).
        dy : np.ndarray, optional
            Gradient values with shape (n_samples, n_features).
        """
        self.X = np.copy(X)
        self.X_org = np.copy(X)
        self.y = np.copy(y)
        self.dy = np.copy(dy)
        self.dy_org = None if dy is None else np.copy(dy)
        self.C = np.copy(X) if C is None else np.copy(C)

        self._n_samples, self._n_features = self.X.shape
        self._n_centres = self.C.shape[0]

        self.eig = np.ones(self._n_features)
        self.scalers = np.copy(self.eig)
        self.evec = np.identity(self._n_features)

    def FV_fit(self, epsi=1):
        """
        Fit the Function Value Radial Basis Function (FV-RBF) model.

        Parameters:
        epsi : float, optional
            The parameter for the RBF. Default is 1.
        """
        self.epsi = epsi

        # Calculate the distance between each pair of points
        dist_matrix = cdist(self.X, self.C, metric='euclidean')

        # Calculate the RBF kernel matrix
        kernel_matrix = np.exp(-self.epsi * (dist_matrix ** 2))

        # Solve the linear system to find the coefficients
        self.coefficients = np.linalg.solve(kernel_matrix, self.y)

        self.cond = np.linalg.cond(kernel_matrix)

    def GE_fit(self, epsi=1):
        """
        Fit the Gradient Enhanced Radial Basis Function (GE-RBF) model.

        Parameters:
        epsi : float, optional
            The parameter for the RBF. Default is 1.
        """
        self.epsi = epsi

        # Calculate the distance between each pair of points
        dist_matrix = cdist(self.X, self.C, metric='euclidean')

        # Calculate the RBF kernel matrix
        kernel_matrix = np.exp(-self.epsi * (dist_matrix ** 2))

        # Calculate the RBF gradient kernel matrix (n_samples, n_samples, n_feat)
        Xm = np.repeat(self.X[:, None, :], self._n_centres, axis=1)
        Cm = np.repeat(self.C[None, :, :], self._n_samples, axis=0)

        XC = (Xm - Cm).transpose(2, 0, 1)

        kernel_gradients = -2 * self.epsi * XC * kernel_matrix[None, :, :]

        # Reshape the kernel gradient matrix
        kernel_gradients = kernel_gradients.reshape(
            self._n_samples*self._n_features, self._n_centres)

        # Create the full matrix
        kernel_matrix = np.vstack((kernel_matrix, kernel_gradients))

        # Least squares fit
        Y = np.vstack((self.y.reshape(-1, 1),
                       self.dy.reshape(self._n_samples * self._n_features, 1, order='F')))

        self.coefficients = np.linalg.solve(
            kernel_matrix.T @ kernel_matrix, kernel_matrix.T @ Y)

        self.cond = np.linalg.cond(kernel_matrix)

    def GO_fit(self, X_FV, y, epsi=1):
        """
        Fit the Gradient Only Radial Basis Function (GO-RBF) model.

        Parameters:
        X_FV : np.ndarray
            Sample locations for the function value.
        y : np.ndarray
            Function values.
        epsi : float, optional
            The parameter for the RBF. Default is 1.
        """
        self.epsi = epsi

        # Calculate the kernel gradient matrix
        dist_matrix = cdist(self.X, self.C, metric='euclidean')

        # Calculate the RBF kernel matrix
        kernel_matrix = np.exp(-self.epsi * (dist_matrix ** 2))

        # Calculate the RBF gradient kernel matrix (n_samples, n_samples, n_feat)
        Xm = np.repeat(self.X[:, None, :], self._n_centres, axis=1)
        Cm = np.repeat(self.C[None, :, :], self._n_samples, axis=0)

        XC = (Xm - Cm).transpose(2, 0, 1)

        kernel_gradients = -2 * self.epsi * XC * kernel_matrix[None, :, :]

        # Reshape the kernel gradient matrix
        kernel_gradients = kernel_gradients.reshape(
            self._n_samples*self._n_features, self._n_centres)

        dist_matrix_FV = cdist(X_FV, self.C, metric='euclidean')

        # Calculate the RBF kernel matrix for the function value points
        kernel_matrix_FV = np.exp(-self.epsi * (dist_matrix_FV ** 2))

        # Create the full matrix
        kernel_matrix = np.vstack((kernel_matrix_FV, kernel_gradients))

        # Least squares fit
        Y = np.vstack((y.reshape(-1, 1),
                       self.dy.reshape(self._n_samples * self._n_features, 1, order='F')))

        self.coefficients = np.linalg.solve(
            kernel_matrix.T @ kernel_matrix, kernel_matrix.T @ Y)

        return

    def __call__(self, Xnew, OnlyFunc=False):
        """
        Predict using the RBF model.

        Parameters:
        Xnew : np.ndarray
           New points for prediction. Shape: (n_samples, n_features).
        only_func : bool, optional
           If True, only function value is returned. Default is False.

        Returns:
        y_pred : np.ndarray
            Predicted output. Shape: (n_samples, ).
        dy_pred : np.ndarray
            Predicted gradients. Shape: (n_samples, n_features).
        """
        # Transform the domain
        Xt = Xnew @ self.evec * self.scalers

        # Calculate the distance between each input point and the training points
        dist_matrix = cdist(Xt, self.C, metric='euclidean')

        # Calculate the RBF kernel matrix
        kernel_matrix = np.exp(-self.epsi * (dist_matrix ** 2))

        # Calculate the predicted output
        y_pred = kernel_matrix @ self.coefficients

        if OnlyFunc:
            return y_pred

        # Calculate the gradient matrix
        Xm = np.repeat(Xt[:, None, :], self._n_centres, axis=1)
        Cm = np.repeat(self.C[None, :, :], Xt.shape[0], axis=0)

        XC = (Xm - Cm).transpose(2, 0, 1)

        kernel_gradients = -2 * self.epsi * XC * kernel_matrix[None, :, :]

        # Reshape the kernel gradient matrix
        kernel_gradients = kernel_gradients.reshape(
            Xt.shape[0]*self._n_features, self._n_centres)

        # chain rule including the rotation.
        dy_pred = ((kernel_gradients @
                   self.coefficients).reshape(Xnew.shape, order='F') * self.scalers) @ (self.evec.T )

        return y_pred, dy_pred
```

