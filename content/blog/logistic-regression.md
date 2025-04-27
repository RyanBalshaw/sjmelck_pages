---
title: "Logistic Regression"
publishdate: 2025-04-23T15:51:58+02:00
author: Ryan Balshaw
description: test
draft: true
toc: true
tags: ["tag1", "tag2", "tag3"]
categories: ["category1"]
hasMermaid: false
_build:
  list: always
  publishResources: true
  render: always
---

# Introduction

In this write-up, the logistic regression model is covered in detail. This model is used to predict binary outcomes and is well known for its ability to drive interpretability. The objective is to start with the basics of the model formulation, consider how to interpret the model parameters, introduce the concept of how to regularise the model, and

# 1. The logistic regression model

Logistic regression is a foundational approach to binary classification problem used to model dichotomous outcome variables.  This model defines a linear discriminative classification model, i.e., a model of $p(y\vert \mathbf{x})$ with $y \in [0, 1], \mathbf{x} \in \mathbb{R}^{d}$. To define this model, a linear mapping of data $\mathbf{x}$ to a single variable $z$, which is directly a one layer neural network mapping to a single node with a learnable bias parameter, is required. This linear mapping is given by
$$
z(\mathbf{x}, \boldsymbol{\zeta}) = \mathbf{w}^T \mathbf{x} + b,
$$
where $\boldsymbol{\zeta}\in \mathbb{R}^{d+1}=[\mathbf{w}^T, b]^T$ is a vector containing $\mathbf{w} \in \mathbb{R}^{d}$, a learnable weight vector, and $b$, a learnable scalar offset parameter, respectively.

> Note: This is just one type of linear model, other choices exist, e.g., kernel-based methods such as radial basis functions (RBFs)!

To convert this linear model to a classification setting, a discriminative conditional distribution for a specific label $y$, i.e., $p(y=1\vert \mathbf{x}, \boldsymbol{\zeta})$, is constructed. This representation is given by
$$
p(y = 1\vert \mathbf{x}, {\boldsymbol{\zeta}}) = \sigma(z_{\boldsymbol{\zeta}}(\mathbf{x}))=p_i,
$$
where $\sigma(u) = \frac{1}{1 + e^{-u}}$ represents the sigmoid function with a property that $1 - \sigma(u)=\sigma(-u)$. This conditional model is assumed to follow the following format
$$
p(y = 0\vert \mathbf{x}, {\boldsymbol{\zeta}}) = 1 - p(y=1\vert \mathbf{x}, {\boldsymbol{\zeta}}) = 1 - \sigma(z_{\boldsymbol{\zeta}}(\mathbf{x}))  = \sigma(-z_{\boldsymbol{\zeta}}(\mathbf{x}))
$$
such that a Bernoulli distribution can be used to define an estimator for the free parameters $\widehat{\boldsymbol{\zeta}}$ via conditional maximum likelihood estimation. $\left(y_i, \mathbf{x}_i \right)$, where $i \in \left[1,N \right]$ and $y_i \in [0, 1]$, the conditional label distribution can be described as

$$
p(y|\mathbf{x}, \boldsymbol{\zeta}) = \text{Bernoulli}(y|\sigma(z_{\boldsymbol{\zeta}}(\mathbf{x}))).
$$
Using this distribution, the probability mass function, a probability of a discrete random variable, for a single observation becomes
$$
p(y|\mathbf{x}, \boldsymbol{\zeta}) = \sigma(z_{\boldsymbol{\zeta}}(\mathbf{x}))^y \cdot (1-\sigma(z_{\boldsymbol{\zeta}}(\mathbf{x})))^{1-y}.
$$
Given a dataset
$\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ that consists of independent and identically distributed (i.i.d.) samples from some true data distribution $p_{data} \left(\mathbf{x}, y \right)$, where $\mathbf{X} \in \mathbb{R}^{N \times d} = \left[\mathbf{x}_1, \cdots, \mathbf{x}_N \right]^T$, the likelihood function of the conditional distribution is given by

$$
l(\boldsymbol{\zeta}, \mathbf{X}, y) = \prod^{N}_{i=1}p_i^{y_i}\cdot(1 - p_i)^{1-y_i},
$$

where $p_i = p \left(y=1 \vert \mathbf{x}_i, {\boldsymbol{\zeta}} \right)$. The log-likelihood (LL) function $L(\boldsymbol{\zeta}, \mathbf{X}, \mathbf{y})$ then becomes

$$
L(\boldsymbol{\zeta}, \mathbf{X}, \mathbf{y}) = \sum_{i=1}^{N} y_i \ln p_i + (1 - y_i) \ln (1 - p_i),
$$
^f3baf4

Equation [[#^f3baf4]] can be used to define the conditional maximum likelihood estimator for the estimate $\widehat{\boldsymbol{\zeta}}$ in numerical, i.e., non closed-form, format
$$
\widehat{\boldsymbol{\zeta}} = \max_{\boldsymbol{\zeta}}L(\boldsymbol{\zeta}, \mathbf{X}, \mathbf{y}).
$$
The negative of Equation [[#^f3baf4]] is commonly referred to as the [cross-entropy loss](https://en.wikipedia.org/wiki/Cross-entropy) due to its direct relationship to cross entropy in information theory. Moreover, for clarity, $\mathbf{x}, y$ are observed and not necessarily in our control, hence the best estimate $\widehat{\boldsymbol{\zeta}}$ is identified based on the observations captured in $\mathcal{D}$.

# 2. Interpreting the model formulation

## 2.1 Mathematical interpretation
To interpret the model, some work needs to be done to figure out what linear function is being modelled. Currently, only $z(\mathbf{x}, \boldsymbol{\zeta})$ contains linearity, while $p(y = 1\vert \mathbf{x}, {\boldsymbol{\zeta}})$ is non-linear due to $\sigma(u)$. One way to interpret the linearity of the model is to try and invert $\sigma(u)$, which conveniently is given by the logit function
$$
\sigma^{-1}(p_i) = \ln \frac{p_i}{1-p_i}.
$$
Thus, the logit function can be applied as follows to
$$
\ln \frac{p(y = 1\vert \mathbf{x}, {\boldsymbol{\zeta}})}{p(y = 0\vert \mathbf{x}, {\boldsymbol{\zeta}})} = \mathbf{w}^T \mathbf{x} + b = z(\boldsymbol{\zeta}, \mathbf{x}).
$$
This is primarily useful as it provides an indication of what is being learnt by the model: the mapping $z$ represents the log-odds between the two classes. How does it do this? Consider these three scenarios:
1. $p(y = 1\vert \mathbf{x}, {\boldsymbol{\zeta}}) = p(y = 0\vert \mathbf{x}, {\boldsymbol{\zeta}})$
2. $p(y = 1\vert \mathbf{x}, {\boldsymbol{\zeta}}) > p(y = 0\vert \mathbf{x}, {\boldsymbol{\zeta}})$
3. $p(y = 1\vert \mathbf{x}, {\boldsymbol{\zeta}}) < p(y = 0\vert \mathbf{x}, {\boldsymbol{\zeta}})$

In the first scenario, we recover
$$
\begin{aligned}
\ln(1) &= \mathbf{w}^T \mathbf{x} + b \\
0 &= \mathbf{w}^T \mathbf{x} + b.
\end{aligned}
$$
This indicates that where $z(\boldsymbol{\zeta}, \mathbf{x}) = 0$ we have equal classes. In the second scenario we recover
$$
+\delta = \mathbf{w}^T \mathbf{x} + b,
$$
which indicates that when $z(\boldsymbol{\zeta}, \mathbf{x}) > 0$, the value $\mathbf{x}$ is more likely to be from class $y=1$. The third scenario, you might guess, recovers
$$
-\delta=\mathbf{w}^T\mathbf{x}+b,
$$
indicating that when $z(\boldsymbol{\zeta}, \mathbf{x}) < 0$, the value $\mathbf{x}$ is more likely to be from class $y=0$.  Hence, the sign of the plane formed by $z(\boldsymbol{\zeta}, \mathbf{x})$ provides an indication of which class $\mathbf{x}$ will be assigned to. Additionally, the equation $z(\boldsymbol{\zeta}, \mathbf{x})=0$ defines a hyperplane in the feature space that serves as the decision boundary for classification, separating the $\mathbf{x}$ space into two half-spaces corresponding to the two classes.

## 2.2 Visual interpretation

Consider a 2D problem where we have data sampled from two Gaussians
![[Pasted image 20250425171002.png]]


![[Pasted image 20250425171018.png]]

![[Pasted image 20250425171032.png]]

# 3. Combating over-fitting via regularisation

Naively maximising Equation [[#^f3baf4]]  using gradient ascent is problematic due to over-fitting if you observe a sub-set of the true conditional distribution being approximated. There are two general solutions to this problem:
1. Penalised maximum likelihood estimation with a regularisation term $R(\boldsymbol{\zeta})$.
2. Using informative priors, $p(\boldsymbol{\zeta})$, over the parameters.

For the sake of this discussion, I will only focus on solution one as this is the process used by the paper being discussed. However, solution one and two are linked, so I will make sure that this link is clear.

The penalised maximum likelihood estimator manifests as
$$
\widehat{\boldsymbol{\zeta}} = \max_{\boldsymbol{\zeta}} \left[ \left(\sum_{i=1}^{N} y_i \ln p_i + (1 - y_i) \ln (1 - p_i) \right) - \alpha \cdot R(\boldsymbol{\zeta})\right],
$$

^f88314

where $\alpha$ is a regularisation term used to control the enforcement of the parameter penalty term $R(\boldsymbol{\zeta})$. Note that the sign given to the penalisation term is negative as we are considering maximisation. Two common choices exist for $R(\boldsymbol{\zeta})$, $L_1$ regularisation and $L_2$ regularisation, otherwise known as lasso regularisation and ridge regression, respectively. An alternative is elastic net regularisation, which linearly combines $L_1$ and $L_2$ regularisation with a interpolation parameter $p$. The $L_1$ and $L_2$ regularisation terms manifest as the Euclidean and Manhattan norms
$$
R_2(\boldsymbol{\zeta}) = \frac{1}{2}\cdot \Vert \boldsymbol{\zeta} \Vert_2^2
$$
and
$$
R_1(\boldsymbol{\zeta}) = \Vert \boldsymbol{\zeta} \Vert_1
$$
respectively. The choice between $R_1(\boldsymbol{\zeta})$ and $R_2(\boldsymbol{\zeta})$ comes down the (1) ease of gradient computation, and (2) parameter interpretability. $R_2(\boldsymbol{\zeta})$ has a simpler derivative, while $R_1(\boldsymbol{\zeta})$ drives weight interpretability by favouring sparsity. The Bayesian perspective of the prior $p(\boldsymbol{\zeta})$ over the parameters can identified, whereby $R_2(\boldsymbol{\zeta})$ is akin to a zero-mean, unit variance Gaussian prior distribution and $R_1(\boldsymbol{\zeta})$ is akin to a Laplace prior distribution centred around zero, which is considered sparse. Recall that Bayes theorem states
$$
p(\boldsymbol{\zeta} \vert y, \mathbf{x}) = \frac{p(y \vert \mathbf{x}, \boldsymbol{\zeta})p(\boldsymbol{\zeta})}{\int_{y}p(y \vert \mathbf{x}, \boldsymbol{\zeta})p(\mathbf{\zeta}) dy }.
$$

^d6b6d3

Equation [[#^d6b6d3]] represents the likelihood multiplied by the prior (the joint distribution $p(y, \boldsymbol{\zeta} \vert \mathbf{x})$) normalised by the marginal likelihood/model evidence. The form of Equation [[#^f88314]] can be recovered by considering the log of the numerator in isolation.

In summary, at this point in time we have a model formulation that maps from $\mathbf{x}$ to $y$ and an objective function with terms to combat overfitting. Now, all we need is data, an optimisation procedure, and code! This is the fun part, so let's gather what we need for optimisation.

# 4. Estimating the model parameters $\widehat{\boldsymbol{\zeta}}$

The first choice that needs to be made, at least in principle, is how we want to solve for $\boldsymbol{\zeta}$. For those coming from an optimisation background, the decision of what to do is trivial; just maximise $L(\boldsymbol{\zeta})$ with or without regularisation. However, I want to take an alternative route, just because I can. This route is called empirical risk minimisation (ERM), which sounds quite fancy but it is simple in practice. ERM introduces the risk $\mathbb{E}_{p(\mathbf{x}, y)} \{ - \log(p(y|\mathbf{x}, \boldsymbol{\zeta}) \}$, which manifests as the expectation over the negative of the log of the conditional label distribution. ERM approximates the population risk by the empirical risk given a sample set $\mathcal{D}$, the observed dataset, as
$$
\mathcal{L}(\boldsymbol{\zeta})=\frac{1}{N} \sum_{i=1}^N - \log\left[ p(y|\mathbf{x}, \boldsymbol{\zeta}) \right] = -\frac{1}{N}L(\boldsymbol{\zeta}, \mathbf{X}, \mathbf{y}),
$$
^c3dead

which is just the normalised negative log-likelihood (NLL). This empirical risk function is then minimised, hence the name: ERM. To minimise this function, often gradient-based optimisation is used. This is a more "statistical" approach to obtain an optimal estimate $\widehat{\boldsymbol{\zeta}}$, but in actual fact this is just a round-about way to get to saying: _just minimise the sample-normalised NLL function_. However, I find it provides a suitable approach to reproduce the terms typically seen in other derivations of the logistic regression model.   A standard ERM procedure is mini-batch gradient descent, which uses a batch of data $\mathbf{D}_b$, a subset of $\mathcal{D}$. Assuming $\mathcal{L}(\boldsymbol{\zeta})$ is differentiable, we can identify different terms for different optimisation procedures. The first of these is the gradient vector.

> Note that while I have not included the regularisation  in this formulation, you can add it in by adding it to $\mathcal{L}(\boldsymbol{\zeta})$:  $\mathcal{L}_R(\boldsymbol{\zeta}) = \mathcal{L}(\boldsymbol{\zeta}) + \alpha R(\boldsymbol{\zeta})$. If the decision is made to use gradient descent, a first-order iterative method, then the constraint is that $R(\boldsymbol{\zeta})$ must be differentiable. Otherwise,
## 4.1 The gradient vector

The gradient vector $\nabla_{\mathbf{x}} f$, where $\nabla$ is the vector differential operator, is given by
$$
\nabla_{\mathbf{x}} f = \begin{bmatrix}
\frac{\partial f}{x_1} \\
\vdots \\
\frac{\partial f}{x_n} \\
\end{bmatrix},
$$
which obeys the relation to the Jacobian matrix $\mathbf{J}_f$
$$
\mathbf{J}_f = \nabla^T f
$$
when $f: \mathbb{R}^n \rightarrow \mathbb{R}$ is a scalar-valued function or
$$
\mathbf{J}_\mathbf{f} = \begin{bmatrix}
\nabla^T f_1 \\
\vdots \\
\nabla^T f_m \\
\end{bmatrix}
$$
when $f : \mathbb{R}^n \rightarrow \mathbb{R}^m$ is a vector-valued function. Moreover, a relation between matrix calculus and the gradient vector exists
$$
\nabla f = \left( \frac{\partial f}{\partial \mathbf{x}} \right)^T
$$

which indicates that $\mathbf{J}_f = \frac{\partial f}{\partial \mathbf{x}}$ and $\mathbf{J}_\mathbf{f} = \frac{\partial \mathbf{f}}{\partial \mathbf{x}}$.

This relationship provides a simple view of how to use chain rule, whereby
$$
\mathbf{J}_{g \circ f}(\mathbf{x}) = \mathbf{J}_g(\mathbf{f}(\mathbf{x}))\mathbf{J}_f(\mathbf{x}),
$$
which can be readily written as
$$
\frac{\partial g}{\partial \mathbf{x}} = \frac{\partial g}{\partial \mathbf{f}}\frac{\partial \mathbf{f}}{\partial \mathbf{x}}.
$$

> Note: This form of chain rule applies in some of the cases, but not in matrix-by-scalar derivatives or scalar-by-matrix derivatives.

Given that $\circ$ is the composition operator, $(g \circ f)(x) = g(f(x))$. Given this relationship, chain rule using the gradient vector can be written as
$$
\begin{aligned}
\nabla_{\mathbf{x}}(g\circ f)(\mathbf{x}) &= \left(\mathbf{J}_{(g\circ f)}(\mathbf{x})\right)^T \\
&= \left( \frac{\partial g}{\partial \mathbf{f}}\frac{\partial \mathbf{f}}{\partial \mathbf{x}} \right)^T \\
&= \left(\frac{\partial \mathbf{f}}{\partial \mathbf{x}}\right)^T \frac{\partial g}{\partial \mathbf{f}}.
\end{aligned}
$$

Hence, $\nabla_{\boldsymbol{\zeta}} \mathcal{L}(\boldsymbol{\zeta})$ can be written using Jacobian notation as
$$
\begin{aligned}
\nabla_{\boldsymbol{\zeta}} \mathcal{L}(\boldsymbol{\zeta}) &= -\frac{1}{N} \sum_{i=1}^{N} \mathbf{J}_{\sigma(z)}^T \, \nabla_{p_i} \, L(p_i) \\
&= -\frac{1}{N} \sum_{i=1}^{N} (J_u \sigma(u) J_z(\boldsymbol{\zeta}))^T \nabla_{p_i} \, L(p_i) \\
&= -\frac{1}{N} \sum_{i=1}^{N} \mathbf{J}^T_z(\boldsymbol{\zeta}) J^T_u \sigma(u) \nabla_{p_i} \, L(p_i).
\end{aligned}
$$
Please note that I have abused my notation, but $u = z(\mathbf{x}, \boldsymbol{\zeta})$ and $p_i = \sigma(u)$. or using matrix calculus notation as
$$
\begin{aligned}
\nabla_{\boldsymbol{\zeta}} \mathcal{L}(\boldsymbol{\zeta}) &= -\frac{1}{N} \sum_{i=1}^{N} \left(\frac{\partial L}{\partial p_i} \frac{\partial \sigma }{\partial u} \frac{\partial z}{\partial \boldsymbol{\zeta}} \right)^T \\
&= -\frac{1}{N} \sum_{i=1}^{N}\left( \frac{\partial z}{\partial \boldsymbol{\zeta}} \right)^T \left( \frac{\partial \sigma }{\partial u} \right)^T \left( \frac{\partial L}{\partial p_i} \right)^T,
\end{aligned}
$$



By compartmentalising these different components, we can rapidly obtain the gradient. Specifically,
$$
\nabla_{p_i}^T \, L(p_i) = \left( \frac{\partial L}{\partial p_i} \right)^T   = \frac{y_i}{p_i} - \frac{1 - y_i}{1-p_i},
$$
$$
J_u \, \sigma(u) = \frac{\partial \sigma }{\partial u} = \frac{e^{-u}}{(1+e^{-u})^2} = \sigma(u) \cdot (1 - \sigma(u)),
$$
and
$$
\mathbf{J}_z(\boldsymbol{\zeta}) = \frac{\partial z}{\partial \boldsymbol{\zeta}}= \begin{bmatrix}
\mathbf{x}_i^T & 1
\end{bmatrix}.
$$

Hence, the full gradient vector is given by
$$
\begin{aligned}
\nabla_{\boldsymbol{\zeta}} \mathcal{L}(\boldsymbol{\zeta}) &= -\frac{1}{N} \sum_{i=1}^{N} \begin{bmatrix} \mathbf{x}_i \\ 1 \end{bmatrix} \sigma(u) \cdot (1 - \sigma(u)) \left(\frac{y_i}{p_i} - \frac{1 - y_i}{1-p_i}\right) \\
&= -\frac{1}{N} \sum_{i=1}^{N} \begin{bmatrix} \mathbf{x}_i \\ 1 \end{bmatrix} \sigma(u) \cdot (1 - \sigma(u)) \left(\frac{y_i}{\sigma(u)} - \frac{1 - y_i}{1-\sigma(u)}\right) \\
&= -\frac{1}{N} \sum_{i=1}^{N} \begin{bmatrix} \mathbf{x}_i \\ 1 \end{bmatrix} \left(y_i - \sigma(u)\right) \\
&= -\frac{1}{N} \sum_{i=1}^{N} \begin{bmatrix} \mathbf{x}_i \\ 1 \end{bmatrix} \left(y_i - \sigma(\mathbf{w}^T\mathbf{x}_i + b)\right) \\
&= \frac{1}{N} \mathbf{X}^T \left(\sigma(\mathbf{X}\boldsymbol{\zeta}) - \mathbf{y} \right) \\
&= \frac{1}{N} \mathbf{X}^T \left(\mathbf{s} - \mathbf{y} \right),
\end{aligned}
$$
where $\mathbf{y} \in \mathbb{R}^N$ is a vector of the expected class labels, $\mathbf{s}=\sigma(\mathbf{X}\boldsymbol{\zeta})\in \mathbb{R}^N$, and $\overline{\mathbf{X}} \in \mathbb{R}^{N \times d + 1}$ is represented as
$$
\overline{\mathbf{X}} = \begin{bmatrix}
\mathbf{x}_1^T & 1 \\
\vdots & \vdots \\
\mathbf{x}_N^T & 1 \\
\end{bmatrix}.
$$
> Note: This is the gradient vector of $\mathcal{L}(\boldsymbol{\zeta})$ and it does not include the regularisation term. To include it, you will just need to add  $\alpha \cdot \nabla_{\boldsymbol{\zeta}}R(\boldsymbol{\zeta})$ to $\nabla_{\boldsymbol{\zeta}} \mathcal{L}(\boldsymbol{\zeta})$.
## 4.2 The Hessian

Certain solvers, e.g., second-order methods such as Newton's method in Optimisation, require Hessian information to perform parameter updates. The Hessian is expressed as the transpose of the Jacobian of the gradient vector, $\mathbf{H}(f(\mathbf{x})) = \mathbf{J}\left( \nabla f(\mathbf{x}) \right)^T$. We can compose this for our use-case as
$$
\mathbf{H}(\boldsymbol{\zeta}) = \left(\frac{\partial}{\partial \boldsymbol{\zeta}}\left( \mathbf{g}(\boldsymbol{\zeta}) \right)\right)^T,
$$
where $\nabla_{\boldsymbol{\zeta}} \mathcal{L}(\boldsymbol{\zeta}) = \mathbf{g}(\boldsymbol{\zeta})$. Then, using chain rule, we can express it as
$$
\mathbf{H}(\boldsymbol{\zeta}) = \left( \frac{\partial \mathbf{g}(\mathbf{s})}{\partial \mathbf{s}} \frac{\partial \mathbf{s}}{\partial\mathbf{z}} \frac{\partial \mathbf{z}(\boldsymbol{\zeta})}{\partial \boldsymbol{\zeta}} \right)^T,
$$

where I have continued to abuse my notation, so for consistency: $s(\mathbf{z})=\sigma(\mathbf{z})$ and $\mathbf{z} = \overline{\mathbf{X}}\boldsymbol{\zeta}$. The required terms are given as:
$$
\begin{aligned}
\frac{\partial \mathbf{g}(\mathbf{s})}{\partial \mathbf{s}} &= \frac{\partial }{\partial \mathbf{s}} \left( \frac{1}{N} \mathbf{X}^T (\mathbf{s} - \mathbf{y}) \right) \\
&= \frac{1}{N}\overline{\mathbf{X}}^T \frac{\partial }{\partial \mathbf{s}} \left(\mathbf{s} - \mathbf{y} \right) \\
&= \frac{1}{N}\overline{\mathbf{X}}^T,
\end{aligned}
$$
where $\mathbf{I}$ is the identity matrix.
$$
\begin{aligned}
\frac{\partial \mathbf{s}}{\partial \mathbf{z}}&=\frac{\partial }{\partial \mathbf{z}} \sigma(\mathbf{z}) \\
&=\text{diag}\left( \sigma(\mathbf{z})(1 - \sigma(\mathbf{z})) \right).
\end{aligned}
$$
$$
\frac{\partial \mathbf{z}(\boldsymbol{\zeta})}{\partial \boldsymbol{\zeta}}=\frac{\partial }{\partial \boldsymbol{\zeta}} \overline{\mathbf{X}}\boldsymbol{\zeta}=\overline{\mathbf{X}}\mathbf{I}=\overline{\mathbf{X}}.
$$
Hence, the Hessian becomes
$$
\begin{aligned}
\mathbf{H}(\boldsymbol{\zeta}) &= \frac{\partial \mathbf{z}(\boldsymbol{\zeta})}{\partial \boldsymbol{\zeta}}^T {\frac{\partial \mathbf{s}}{\partial \mathbf{z}}}^T {\frac{\partial \mathbf{g}(\mathbf{s})}{\partial \mathbf{s}}}^T \\
&= \frac{1}{N} \overline{\mathbf{X}}^T \text{diag}\left[ \sigma(\mathbf{z})(1 - \sigma(\mathbf{z})) \right] \overline{\mathbf{X}}.
\end{aligned}
$$

> Note: This is the Hessian matrix of $\mathcal{L}(\boldsymbol{\zeta})$ and it does not include the regularisation term. To include it, you will just need to add  $\alpha \cdot \mathbf{J}\left( \nabla R(\boldsymbol{\zeta}) \right)^T$ to $\mathbf{H}(\boldsymbol{\zeta})$.


# 5. Multinomial logistic regression

To wrap this discussion up, I will also discuss how multinomial logistic regression can be scaled up to multiple classes. I will not progress further than the model formulation and how to compute the gradient (CONFIRM). The first step is to introduce a mapping to $\mathbf{z} \in \mathcal{R}^K$, where $K$ refers to the number of classes considered. This is achieved using
$$
\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b},
$$
where $\mathbf{W}=[\mathbf{w}_k, \cdots, \mathbf{w}_K]^T$ represents $K$ independent mappings and $\mathbf{b} = [b_k, \cdots, b_K]^T$. To derive the form of $p(y_i=k\vert \mathbf{x})$, there are two prominent forms in the literature, namely via $(i)$ minimal parametrisation and $(ii)$ over-parametrisation of the weight vectors. I will discuss each in turn.

## 5.1 Minimal parametrisation

To derive this formation, the constraint $\sum_{l=1}^{K}p(y=l \vert \mathbf{x})=1$ is key. Specifically, we can write
$$
p(y=K \vert \mathbf{x}) = 1 - \sum_{l=1}^{K-1}p(y=l \vert \mathbf{x})
$$
and then parametrise the log-odds between $p(y=k \vert \mathbf{x})$ and $p(y=K \vert \mathbf{x})$ with the linear mapping to $z_k=\mathbf{w}_k^T \mathbf{x} + b_k$. This is given as
$$
\ln \frac{p(y=k \vert \mathbf{x})}{p(y=K \vert \mathbf{x})} = z_k(\boldsymbol{\zeta}_k, \mathbf{x}) =\mathbf{w}_k^T \mathbf{x} + b_k,
$$
for the log-odds and as
$$
\frac{p(y=k \vert \mathbf{x})}{p(y=K \vert \mathbf{x})} = e^{\mathbf{w}_k^T \mathbf{x} + b_k},
$$
for the odds ratio. Hence, under this formulation, $p(y=k \vert \mathbf{x}) = e^{\mathbf{w}_k^T \mathbf{x} + b_k}p(y=K \vert \mathbf{x})$. Substituting this into the probability sum constraint returns
$$
\begin{aligned}
p(y=K \vert \mathbf{x}) &= 1 - \sum_{l=1}^{K-1}e^{\mathbf{w}_l^T \mathbf{x} + b_l}p(y=K \vert \mathbf{x}) \\
&= \frac{1}{1 + \sum_{l=1}^{K-1}e^{\mathbf{w}_l^T \mathbf{x} + b_l}},
\end{aligned}
$$
which can then be used to obtain
$$
p(y =k \vert \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x} + b_k}}{1 + \sum_{l=1}^{K-1}e^{\mathbf{w}_l^T \mathbf{x} + b_l}}.
$$

In this solution, the fact that one classes probability is fully parametrised by the probabilities of the other classes is key.

## 5.2 Over-parametrisation

In the over-parametrised form, the Gibbs measure is used to define
$$
p(y=k \vert \mathbf{x}) = \frac{1}{Z}\cdot e^{\mathbf{w}_k^T\mathbf{x} + b_k},
$$
where $Z$ is a partition function that can be used to enforce that the Gibbs measure outputs a valid probability. Specifically, the sum over discrete random variables constraint can be used to define $Z$
$$
\begin{aligned}
1 &= \sum_{k=1}^{K} p(y=k \vert \mathbf{x}) \\
1 &= \sum_{k=1}^{K}\frac{1}{Z}\cdot e^{\mathbf{w}_k^T\mathbf{x} + b_k} \\
Z &= \sum_{k=1}^{K} e^{\mathbf{w}_k^T\mathbf{x} + b_k}.
\end{aligned}
$$

Hence, the conditional distribution $p(y=k \vert \mathbf{x})$ is given by
$$
p(y=k \vert \mathbf{x}) = \frac{e^{\mathbf{w}_k^T\mathbf{x} + b_k}}{\sum_{k=1}^{K} e^{\mathbf{w}_k^T\mathbf{x} + b_k}}=\sigma_s^{(k)}(\mathbf{z}),
$$
where $\boldsymbol{\sigma}_s(\mathbf{z}): \mathbb{R}^K \rightarrow \mathbb{R}^K$ is commonly referred to as the softmax function. One problem with this term is that it is invariant to translation, i.e.,  $\sigma_s^{(k)}(\mathbf{z} + \mathbf{c}) = \sigma_s^{(k)}(\mathbf{z})$, and if we set $\mathbf{c} = \boldsymbol{\zeta}_k$, we can recover the minimal parametrisation form, effectively removing the weights for the $K^{th}$ class.

## 5.3 The multinomial objective function

For both forms of $p(y=k\vert \mathbf{x})$, the observed class labels $y_i \in \{1, \cdots, K\}$ for $i=1, \cdots, N$ are considered as samples of categorically distributed random variables $Y_1, \cdots, Y_K$ leading to the probability mass function
$$
p(y|\mathbf{x}_i, \boldsymbol{Z})=\prod_{j=1}^{K} p_{ij}^{[y_i=k]},
$$
where $p_{ij}=p(y=j \vert \mathbf{x}_i)$, $y_i$ is the class label for sample $x_i$, and $[y=k]$ is the Iverson bracket
$$
[U] = \begin{cases}
1 & \text{if } U \text{ is true;} \\
0 & \text{otherwise.} \end{cases}$$
The likelihood function is defined by
$$
l(\boldsymbol{Z}, \mathbf{X}, y) = \prod^{N}_{i=1}\prod_{j=1}^{K} p_{ij}^{[y_i=k]},
$$
which can be used to define an objective function using the empirical risk, i.e., by minimising the sample-normalised NLL function
$$
\begin{aligned}
\mathcal{L}(\boldsymbol{Z})&=-\frac{1}{N} \sum_{i=1}^N\sum_{j=1}^K [y_i=k]\ln p(y=j \vert \mathbf{x}_i) \\
&= -\frac{1}{N} \sum_{i=1}^N\sum_{j=1}^K [y_i=k]\log(\sigma_s^{(k)}(\mathbf{z}_i)).
\end{aligned}
$$

One interesting take-away is that this loss only considers the probability given to the expected class label, and on the surface it appears that the parameter updates will only care about the correct class prediction. To determine if this is truly the case, it is of interest to look at how the parameters are updated. Specifically, iterative parameter updates, e.g., gradient descent,  require the gradient vector, which now relies on the term $\partial \boldsymbol{\sigma} / \partial \mathbf{z}$, i.e., the Jacobian of the softmax function with respect to $\mathbf{z}$. The form of this term is discussed to drive intuition behind the loss.
## 5.4 The Jacobian of the softmax function

Each row in the Jacobian $\mathbf{J}_\boldsymbol{\sigma} \, \boldsymbol{\sigma}(\mathbf{z}) \in \mathbb{R}^{K \times K}$ is given by $\nabla^T\left( \sigma^{(k)}_s(\mathbf{z}) \right)$. For simplicity, lets consider one term in this vector $\frac{\partial \sigma_s^{(i)}(\mathbf{z})}{\partial z_j}$, which represents $J_{\boldsymbol{\sigma}, ij} \, \sigma(\mathbf{z})$. To help this process, the log-derivative trick can be used, which is given by
$$
\frac{\partial}{\partial u} \log f = \frac{1}{f} \cdot \frac{\partial f}{\partial u}.
$$
Hence, this trick yields $\frac{\partial f}{\partial u}= f \frac{\partial}{\partial u} \log f$. Applying this to the considered term yields
$$
\begin{aligned}
\frac{\partial \sigma_s^{(i)}(\mathbf{z})}{\partial z_j} &= \sigma_s^{(i)}(\mathbf{z}) \cdot \frac{\partial}{\partial z_j} \left[ log(\frac{e^{z_i}}{\sum_{k=1}^{K} e^{z_k}}) \right] \\
&= \sigma_s^{(i)}(\mathbf{z}) \cdot \frac{\partial}{\partial z_j} \left[ z_i - log(\sum_{k=1}^{K} e^{z_k}) \right] \\
&- \sigma_s^{(i)}(\mathbf{z}) \cdot \left( [i==j] - \frac{1}{\sum_{k=1}^{K} e^{z_k}}\cdot e^{z_j} \right) \\
&= \sigma_s^{(i)}(\mathbf{z}) \cdot \left( [i==j] - \sigma_s^{(j)}(\mathbf{z}) \right)
\end{aligned}
$$
Hence, the full Jacobian can be written neatly as
$$
\mathbf{J}_\boldsymbol{\sigma} \, \boldsymbol{\sigma}(\mathbf{z}) = \text{diag}(\boldsymbol{\sigma}(\mathbf{z})) - \boldsymbol{\sigma}(\mathbf{z})\boldsymbol{\sigma}(\mathbf{z})^T
$$
This relation indicates that the gradient vector will contain sensitivity information on both the correct class predictions and the incorrect class predictions.
