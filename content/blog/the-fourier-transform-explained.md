---
title: "The Fourier transform: Explained"
publishdate: 2023-01-19T16:18:44+02:00
author: Ryan Balshaw
description: This is an explanation of the Fourier transform for MEV781 students.
draft: false
toc: true
OverviewFig: "animation_real_imag_components.gif"
tags: ["Fourier Transform"]
categories: ["Signal processing"]
_build:
  list: always
  publishResources: true
  render: always
---

Good day 👋

So if you are opening this document, you hopefully have some questions about the Fourier Transform (FT), the Discrete Fourier Transform (DFT) or perhaps you are just curious to see what this document entails. Regardless of your reason for opening, I hope that this document makes the FT and DFT more accessible and intuitive. I have attempted to write this from a basic Mechanical Engineering context, so I hope you enjoy this document! If you have any questions, please feel to contact me at ryanbalshaw81@gmail.com.

Kind regards,

Ryan Balshaw 🦮

P.S. The initial part of this document is largely based off a video from [3blue1brown](https://www.youtube.com/watch?v=spUNpyF58BY), but with more details and the ability to play around with the examples from the video. Feel free to watch this video first and then decide if this document is worth it.

# Introduction

This tutorial will be begin with framing the ideas behind the FT and then go into the more complex details for the DFT, but I will not cover the Fast Fourier Transform (FFT) as it is just the DFT's algorithmic implementation. The first question I would like to ask is: what is your interpretation of a frequency \\(f\\) or \\(\omega\\)?. Hopefully, the following may come to mind:
- It describes some rate, such as the rate of oscillation in sinusoids.

- It is a term used in MEV that has units of \\(Hz\\) or \\( \frac{rad}{s} \\).

- It is the angular velocity around the centre of mass in bodies that undergo some rotation.

All three of these ideas are partially correct and depend on application context, but what is important is that all three of these ideas are key to interpreting the FT. So, without further ado, let's jump right into understanding the FT and the DFT.

The basic idea of the FT is to determine the important frequency components \\( f \\) in a signal, and we can understand this idea by investigating the Fourier series. The Fourier series introduces this idea that a real valued periodic function \\( g(x) \\) can be decomposed into a series of sinusoidal components. This is formally written as

$$
g_N(x) = \frac{a_0}{2} + \sum_{n=1}^N \left[ a_n \cos(\frac{2\pi}{P} n x) + b_n \sin(\frac{2\pi}{P} n x) \right],
$$

where $N$ is the number of components used to decompose  \\( g(x) \\), \\( P \\) is the interval length of \\( g(x) \\) and \\( a_n \\) and \\( b_n \\) are the amplitude weighting components applied to the cosine and sine functions. For those interested, \\( a_0 \\), \\( a_n \\) and \\( b_n \\) are found through

$$
a_n = \frac{2}{P} \int_P g(x) \cdot \cos(\frac{2\pi}{P} n x) dx,
$$

$$
b_n = \frac{2}{P} \int_P g(x) \cdot \sin(\frac{2\pi}{P} n x) dx,
$$

$$
a_0 = \frac{2}{P} \int_P g(x) dx.
$$

However, what if we wish to determine which sinusoidal components are dominant in the signal without knowing its analytical function? To do this, we move away from the Fourier series and introduce the idea behind the Fourier Transform. It is of crucial importance to note that these two methods are *not the same*, but they have strong connections to one another. To show the relation between the two, consider the following Fourier Series formulation of \\( g(x) = x - \lfloor x \rfloor \\) (using the first question from the first assignment, the derivation is on the [github repository](https://github.com/RyanBalshaw/MEV781_Tutorials))

$$
g_N(x) = \frac{1}{2} - \sum_{n=1}^N \frac{1}{\pi n} \sin(2 \pi n x).
$$

This expansion is visualised below, and the relation between the Fourier Series and the Fourier Transform is that the amplitude of the sinusoidal components at a given integer harmonic provides some indication of how the Fourier Transform will look (this is not completely correct, but it is a sufficient proxy at this stage so do not read too much into it).

![image alt text](animation_fourier_series.gif)

To describe the Fourier Transform, I will quote Wikipedia (i.e., the \\(0^{th}\\) iteration of [ChatGPT](https://openai.com/blog/chatgpt)): "In mathematics, a Fourier transform is a mathematical transform that decomposes functions depending on space or time into functions depending on spatial or temporal frequency, such as the expression of a musical chord in terms of the volumes and frequencies of its constituent notes. The term Fourier transform refers to both the frequency domain representation and the mathematical operation that associates the frequency domain representation to a function of space or time."

This may seem intimidating for those less well versed with the Fourier Transform, so let's decompose this into sub-ideas:
1. The Fourier Transform shifts/transforms functions in the time domain to the frequency domain.
2. A mathematical operation is required to facilitate this transformation.

Great, but how does this help us? I will try to explain the intuition behind the mathematical operations in a simple way, with the help of some useful visual plots that one can interact with. This may seem somewhat abstract, but if you bear with me, it will hopefully all make sense at the end.

**Idea 1**

Before I detail the Fourier Transform, let's try the following: let's take a signal \\( g(t) \\) (where the change of independent variable is necessary for time-series data) and wrap it around a 2D axis. To enable this process, let's first consider the meaning of frequency components \\( f \\) and \\( \omega \\): the units for \\( f \\) is typically \\( Hz \\) or \\( cycles/s \\) and the variable \\( \omega \\) has units of \\( rad/s \\). One can obtain \\( \omega \\) from \\( f \\) through \\( \omega = 2 \pi f \\).

Why is this useful? Well in a physics context, \\( \omega \\) describes the rate at which something rotates around an axis. So, to wrap a signal around a 2D axis, we take the signal \\( g(t) \\), its time \\( t \\), a pre-selected \\( f_0 \\) value and now exploit some simple physics relationships, namely:

$$
\theta(t) = \int_{0}^{t}\omega(\tau)d\tau,
$$

where the indefinite integral \\( \int\omega(t)dt \\) can be written in this form if \\( \theta(t = 0) = 0 \\) and we are only concerned with \\( t \geq 0 \\), which is a reasonable assumption to make in this case. If we assume that \\( \omega(t) \\) is a constant (\\( \omega(t) = \omega_0 \\)) then the angular displacement becomes

$$
\theta(t) = \omega_0 t.
$$

This expression allows us to work out where \\( g(t = t_i) \\) will lie in a 2D plane by working out the angular position in the 2D plane and by letting \\( g(t = t_i) \\) be the distance from the origin at some angular position \\( \theta(t=t_i) \\). More realistically, we can just work out the rotation  \\( \theta_i \\) for all time indices \\( t_i \\) and then plot the data using \\( x(t_i) = g(t_i) \cos(\theta_i) \\) and \\( y(t_i) = g(t_i) \sin(\theta_i) \\), where we obtain the rotation \\( \theta \\) through \\( \theta_i = 2 \pi f_0 t_i \\) (note here that I dropped the time index notation  \\( \cdot(t) \\) and replaced it with an index notation \\( \cdot_i \\) as I need to discretise the system to plot it). This effectively shifts the signal from a 2-dimensional Euclidean coordinate system into the polar coordinate system. Let's now go through this process cell by cell.

## Step 1: define a function g(t)

$$
g(t) = \frac{1}{4}\cos(2 \pi f_1 t) + \cos(2 \pi f_2 t) + 1,
$$

where \\( f_1 = 2 \\) and \\( f_2 = 3 \\). This function consists of two cosines addes together with some pre-defined offset. Now, let's try this wrapping around an axis idea! First let's define \\( f \\) and calculate \\( \theta \\).

![function of interest](function.png)

## Step 2: define \\( f \\) and calculate the angular displacement \\( \theta \\).
Let

$$
f_0 = 2.5,
$$

$$
\theta_i =2 \pi f_0 t_i,
$$

where \\( t_i \\) is the \\( i^{th} \\) index in some time vector \\( \mathbf{t} \\). Now, let's wrap the signal around an axis! I have done this for you, in an interactive fashion, but it is simply working out \\([x_i,y_i]\\) using \\([x_i = g(t_i) \cos(\theta_i), y_i = g(t_i) \sin(\theta_i)]\\). I have also plotted the centre of mass of the signal with a red cross, but don't pay too much attention to this yet.

![axis wrapping for f0 = 2.5](animation_fourier_wrapping.gif)

Let's now consider the case where we use different $f_0$ values. In the graphic below, I have visualised this wrapping for
$f_0 \in [0, 1, 2, 3, 4, 5]$. If we inspect the center of mass in each plot (demonstrated by the red cross), we
can see that for \\( f_0=0, 2, \\) and \\( 3 \\) the center of mass is not equal to zero.

![axis wrapping for different f0 values](different_wraps_and_com.png)

Now that we have some idea of how to wrap signals around a 2D axis, let's try and visualise some
information that we can obtain from the 2D plot. For this example, we will consider the centre of mass
of the line \\(\mathbf{r} = [\mu_x, \mu_y]\\), which is given through

$$
\mu_{x} = \frac{1}{N} \sum_i x_i,% = \frac{1}{t_1 - t_0}\int_{t_0}^{t_1} g(t)\sin(\theta(t))dt,
$$

$$
\mu_{y} = \frac{1}{N} \sum_i y_i,% = \frac{1}{t_1 - t_0}\int_{t_0}^{t_1} g(t)\cos(\theta(t))dt.
$$

For those who are interested about how these equations came about, we assume that the discrete points
of the signal sample of the original waveform have equal mass. The centre of mass is then given by
\\( x_{com} = \frac{\sum_{i=1}^N m_i x_i}{M} \\) and \\( y_{com} = \frac{\sum_{i=1}^N m_i y_i}{M} \\).
Since we assume each point has equal mass, \\( m_i \\) can be factored out and \\( \frac{m_i}{M} =
\frac{1}{N} \\).


For visualisation, we can also consider the vector norm of the centre of mass \\(\Vert r \Vert_2 = \sqrt{\mu_{x}^2 + \mu_{y}^2}\\)
and angle \\( \theta = \tan^{-1} \left( \frac{\mu_{y}}{\mu_{x}} \right) \\). Let's visualise how these
four representations of the centre of mass vary for different values of \\( f \\).

![axis wrapping for f0 = 2.5](animation_real_imag_components.gif)

What we can now see is that the centre of mass shows some interesting phenomena, where if \\( f_0 \\) matches some of the
frequencies in our signal, it grows in magnitude. This phenomena is the underlying principle of the Fourier Transform!
To realise this, let's introduce a version of the Fourier Transform mathematics:

$$
\mathcal{X}(\omega) = \frac{1}{T} \int_{0}^{T} g(t) e^{-i \omega t} dt,
$$

where \\( \mathcal{X}(\cdot) \\) is the Fourier transform of the time domain function \\( g(t) \\) at a given frequency \\( \omega \\).
The first thing you will notice is the presence of \\( e^{-i \omega t} \\), which is just a complex domain expression of how we
have organised our 2D plane plot! If this is not clear, take note of the following relationship:

$$
z = x - i y = \Vert \mathbf{z} \Vert_2 (\cos \theta - i \sin \theta) = \Vert z \Vert_2 e^{-i \theta},
$$

where \\( \Vert z \Vert_2 = \sqrt{x^2 + y^2} \\). So for the plots we made previously, we have actually been plotting everything
in the complex domain, which occurs in the Argand (complex) plane. So now that we understand this idea, let's start to expand the
Fourier Transform:

$$
\mathcal{X}(\omega) = \frac{1}{T} \int_{0}^{T} f(t) (\cos(\omega t) - i \sin (\omega t)) dt,
$$

$$
\mathcal{X}(\omega) = \frac{1}{T} [\int_{0}^{T} f(t)\cos(\omega t)dt - i \int_{0}^{T}f(t)\sin (\omega t) dt] ,
$$

$$
\mathcal{X}(f) = \frac{1}{T} [\int_{0}^{T} f(t)\cos(2 \pi f t)dt - i \int_{0}^{T}f(t)\sin (2 \pi f t) dt],
$$

where \\( f = 2\pi \omega \\) is the frequency in Hertz (Hz).

Now, before we continue, let's just take a moment to recall what the definite integral tells us. The definite integral is the area
of a function between the bounds of integration, which can be expressed as the summation of infinitely small areas of a function
weighted by the change in distance along the independent variable axis. For our case, we can immediately see that the expressions
\\( \mu_x \\) and \\( \mu_y \\) could be developed into

$$
\mu_{x} = \frac{1}{N} \sum_i x_i = \frac{1}{N} \sum_i g(t_i) \cos(\theta_i) = \frac{1}{T} \sum_i g(t_i) \cos(\theta_i) \Delta t_i = \frac{1}{t_1 - t_0}\int_{t_0}^{t_1} g(t)\cos(\theta(t))dt,
$$

$$
\mu_{y} = \frac{1}{N} \sum_i y_i = \frac{1}{N} \sum_i g(t_i) \sin(\theta_i) = \frac{1}{T} \sum_i g(t_i) \sin(\theta_i)\Delta t_i = \frac{1}{t_1 - t_0}\int_{t_0}^{t_1} g(t)\sin(\theta(t))dt,
$$

where \\( T = N\Delta t \\). Whichs shows that, ultimately, the Fourier transform for a frequency component \\( f \\) is a
calculation of the centre of mass of a signal that is wound around a 2D axis! This is a pretty amazing result, and the underlying
meaning of the mathematics is to calculate the centre of mass of a signal wound around an axis at some rate. The full definition
one will usually see of the Fourier Transform is

$$
\mathcal{X}(f) = \int_{-\infty}^{\infty} x(t) e^{-i 2 \pi f t}dt,
$$

where \\( x(t) \\) represents a general time domain function, the minus sign in the exponential simply defines the winding/rotation
direction in the complex domain and we drop the division.

# Side comments

1. For those of you who come from a control systems background, you may ask how does the FT relate to the Laplace Transform? The
FT can be considered a special case of the Laplace transform with \\( \alpha = 0 \\). To prove this, consider the Laplace
transform equation:
$$
\mathcal{X}(s) = \int_{-\infty}^{\infty} x(t) e^{st}dt,
$$

where \\( s = \alpha + i \omega \\). Simply zeroing \\( \alpha \\) leaves us with the original Fourier transform. A more in-depth
and intuitive discussion around the relationship between these two transforms can be found [here](https://www.youtube.com/watch?v=n2y7n6jw5d0).

2. What is the offset we see when \\( f=0 \\)? This is simply the DC offset of the signal, otherwise referred to as the signal mean. To prove this, consider the following:

$$
\mathcal{X}(0) = \int_{-\infty}^{\infty} x(t) e^{-i 2 \pi 0 t}dt,
$$

$$
= \int_{-\infty}^{\infty} x(t) e^{0}dt,
$$

$$
= \int_{-\infty}^{\infty} x(t) (1)dt,
$$

$$
= \int_{-\infty}^{\infty} x(t) dt.
$$

3. Does the \\( -\infty \\) and \\( \infty \\) bounds on the integral change affect the process we have gone through?
Well, when we implement the FT, in practice, we often do not know the true signal function \\( x(t) \\) and we can only discretely
sample the signal in some interval. To address the integral bounds, we assume that the sampled signal is continuously recirculated
and thus represents a signal \\( x(t) \\) with period T.

4. You just mentioned that we commonly just sample a signal, so what happens in the FT? As we discretely sample the signal,
we do not implement the FT but rather the DFT. If you follow the MEV781 B1-03 lecture slides, you can clearly see that the
DFT just expands the continuous case to discrete signals with a sampling rate \\( F_s = \frac{1}{\Delta t} \\). The Fast Fourier
Transform (FFT) is just an efficient algorithm used to solve for the DFT coefficients.


# Using the Fourier Transform in practice

To use the FT in practice, we do not implement the FT, but rather its discrete counterpart, the DFT. Fortunately for us,
many commercial software packages have implementations that we can use to quickly implement the DFT. I would
just like to discuss some of the elements of the DFT.

There are two commonly applied versions of the DFT, namely the real and complex DFT. The difference is simply approximating the
Fourier Series expansion versus approximating the Fourier Transform. For the sake of concreteness, I will introduce both
versions to you.

## Real DFT
If we assume that we have a N-long sequence of samples of a time-series signal, which we shall denote as \\(x[n]\\) (\\(n \in \mathbb{Z}\\)), the real DFT gives two \\(\frac{N}{2} + 1\\)-length arrays, which we shall denote as \\(X_1[\cdot]\\) and \\(X_2[\cdot]\\) respectively. The \\(k^{th}\\) index in these arrays is given as

$$
X_1[k] = \frac{2}{N} \sum_{n=0}^{N-1} x[n] \cos \left( \frac{2\pi kn}{N} \right),
$$

$$
X_2[k] = \frac{2}{N} \sum_{n=0}^{N-1} x[n] \sin \left( \frac{2\pi kn}{N} \right),
$$

where \\(k=0,2,\cdots,\frac{N}{2}\\). It is clear to note that this bears a striking resemblence to the Fourier series introduced earlier. We can then "reconstruct" the original time-series signal using

$$
x[n] = \sum_{k=0}^{N/2} X_1[k]\cos \left( \frac{2\pi kn}{N} \right) + X_2[k] \sin \left( \frac{2\pi kn}{N} \right).
$$

## Complex DFT
The complex DFT, as presented in the MEV 781 notes, takes a discrete time-series signal of length \\(N\\) and determines the Fourier Coefficients \\(X[k]\\) using

$$
X[k] = \frac{1}{N} \sum_{n=0}^{N-1} x[n] e^{-j2\pi \frac{kn}{N}},
$$

where there are \\(N\\) complex coefficients. The intuition behind the steps here should be clear. From the previous discussion in the text, it should be clear that \\(X[0]\\) is the DC frequency component, or the discrete signal mean. One can also convert back to the time domain using

$$
x[n] = \sum_{k=0}^{N-1}X[k] e^{j 2 \pi \frac{kn}{N}}.
$$

## How do we use commercial software packages

The FFT is an algorithm that is used to efficiently calculate the complex DFT. For those interested in how the FFT algorithm works,
I recommend watching the following [video](https://www.youtube.com/watch?v=h7apO7q16V0).

Since the FFT is a technique to obtain the DFT coefficients, all that is missing to complete the picture is to relate the \\(N\\)
complex coefficients to frequency components. To do this, we need to convert the integer indices to frequencies, and we need
use the signal sampling frequency \\(F_s\\) for this conversion. Each index of the DFT then spaced using \\(freq_{@X[k]} = k\times
\frac{F_s}{N}\\), \\(k = 0, 1, \cdots, N - 1\\), where \\(\frac{F_s}{N}\\) is known as the frequency resolution (\\(\Delta f\\)). We can also
calculate the frequency resolution if we know the sampling period \\(\Delta t\\): \\(\Delta f = \frac{1}{N\Delta t}\\) as \\(F_s =
\frac{1}{\Delta t}\\).

For visualisation of the DFT coefficients, it is common practice to look at the magnitude and phase of the complex coefficients.
Alternatively, if we wish to visualise the Power spectrum, we just multiply the DFT coefficients with its complex conjugate:
\\(S_k = X[k]\times X^*[k]\\). This is simply looking at the squared magnitude of \\(X[k]\\).

Now, when some of you follow this approach and visualise the DFT, is it expected that the plot have some visual symmetry,
as if the DFT is mirrored about some frequency component. The reason for this is simple to visualise, but is an important aspect
of the DFT. The basic reason can be elaborated through the rotation direction, where changes in rotation direction in a complex
domain cannot be detected in the real domain. To make this clearer, consider the cell below (please run it, observe what happens
and then come back here). Notice that the line in the real-time plane is equivalent and the rotation direction cannot be identified.
However, in the complex-time plane, the lines are out of sync. This is an important result from the FT that also manifests in the DFT.

![axis rotation example](animation_rotation.gif)


To detail why this rotation idea is important, we need to consider the mathematics in the Fourier Transform and the Discrete
Fourier Transform:

$$
\mathcal{X}(f) = \frac{1}{T} [\int_{0}^{T} x(t)\cos(f t)dt - i \int_{0}^{T}x(t)\sin (f t) dt],
$$

$$
= \frac{1}{T} \int_{0}^{T} x(t) \left[\cos(f t) - i \sin (f t) \right] dt.
$$

Then, let's look at the frequency \\(-f\\):

$$
\mathcal{X}(-f) = \frac{1}{T} [\int_{0}^{T} x(t)\cos(f t)dt + i \int_{0}^{T}x(t)\sin (f t) dt],
$$

$$
= \frac{1}{T} \int_{0}^{T} x(t) \left[\cos(f t) + i \sin (f t) \right] dt.
$$

What should be clear is that the Fourier coefficient at \\(f\\) and \\(-f\\) results in a change of sign for the sine component. Given our experience with the visual elements of the Fourier Transform, this would result in a point that is mirrored across the real line (a change in the sign for \\(\mu_y\\)). More formally, \\(X(-f) = F^*(f)\\) where \\(X^*(f)\\) is the complex conjugate of \\(X(f)\\). Now, the question is, how does this manifest in the DFT? Consider any fixed point \\(l\\), where \\(1 \leq l \leq N - 1\\).

$$
X[l] = \frac{1}{N} \sum_{n=0}^{N-1} x[n] e^{-j2\pi \frac{ln}{N}},
$$

$$
X[N - l] = \frac{1}{N} \sum_{n=0}^{N-1} x[n] e^{-j2\pi \frac{(N - l)n}{N}},
$$

$$
 =  \frac{1}{N} \sum_{n=0}^{N-1} x[n] e^{-j2\pi (\frac{(N)n}{N} - \frac{ln}{N})},
$$

$$
 =  \frac{1}{N} \sum_{n=0}^{N-1} x[n] e^{-j2\pi n +j2\pi \frac{ln}{N}},
$$

and since \\(-j2\pi n \\) simply moves the point \\(n\\) rotations around the domain we can effectively drop it from the equation. Thus:

$$
X[N - l] = \frac{1}{N} \sum_{n=0}^{N-1} x[n] e^{j2\pi \frac{ln}{N}},
$$

where it is now clear to see that \\( X[l] = X^{\ast}[N - l] \\). Furthermore, if \\( l=\frac{N}{2}\\), we get
\\( X[\frac{N}{2}] = X^{\ast}[\frac{N}{2}] \\).

To summarise the above, the \\(l^{th}\\) index in the DFT of a _real-valued_ discrete signal has the same magnitude as the
\\((N - l)^{th}\\) index, but the two are complex conjugates of one another (mirrored over the real line). The emphasis
on _real-valued_ is because this does not hold if your signal has complex coefficients, however this does not apply when
we take the DFT of time-series signals.

So, what does this all mean? Well, the largest frequency component that the DFT can effectively determine is
\\(freq_{@X[N/2]} = \frac{N}{2} \times \frac{F_s}{N} = \frac{F_s}{2}\\). This frequency component is commonly referred to as the
Nyquist frequency. After this point, all of the DFT indices are simply the complex conjugates of previously determined indices.
Hence, it is common to plot the one-sided magnitude of the DFT. However, if you do so, please multiply the one-sided magnitude
by two to account for the fact that you removed half of the information from the DFT.

Finally, there are some key differences in terminology that you can slip up on, and to make sure you are aware of them please
refer to this [document](https://www.analog.com/media/en/technical-documentation/dsp-book/dsp_book_Ch31.pdf).

If you managed to get this far, thank you for reading this! I hope that you found it mildly enlightening.

As always, thanks for reading! 👨🏼‍💻
