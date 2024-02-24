---
title: "RBF Models"
publishdate: 2023-07-08T11:59:05+02:00
author: Johann Bouwer
description: A how-to on implementing gradients into the radial basis function surrogate model.
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

[SCIPY](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html#scipy.interpolate.RBFInterpolator)

![Alt Text](C:\Users/bouwe/OneDrive/Desktop/GitCode/sjmelck_pages/assets/images/RBF-Models/test.png)


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

