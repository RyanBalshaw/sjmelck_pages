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

Radial Basis Functions (RBF) surrogate models are used to approximate 
complex and computationally expensive functions. Most commonly, some data set 
\\( X\\) and function values \\( y\\) are used to construct a computationally 
inexpensive model. [SCIPY](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html#scipy.interpolate.RBFInterpolator) 
has an implementation of this.

![Alt Text](test.png)


- the maths
- the function based method in python
- the maths for gradients
- the GE method in python

$$
F(x) = x^2
$$

```python
def test(x):
	return x
```

