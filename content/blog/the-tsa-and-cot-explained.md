---
title: "Time synchronous averaging and computed order tracking: Explained"
publishdate: 2023-04-14T16:14:08+02:00
author: Ryan Balshaw
description: This is a simple explanation of the TSA and COT.
draft: false
toc: true
tags: ["tag1", "tag2", "tag3"]
categories: ["category1"]
_build:
  list: always
  publishResources: true
  render: always
---

Good day 👋

The purpose of this explanation is to detail a high-level overview of the time-synchronous average (TSA) and Computed Order Tracking (COT). This will require a basic understanding of the objectives of TSA and COT, and I will try to provide some explanations and intuition to help those who do not have a good grasp on the concepts. At the end of each section, I provide a step-by-step guide for implementing these methods. If you have any questions, please feel to contact me at ryanbalshaw81@gmail.com.

Kind regards,

Ryan Balshaw 🦮

## Introduction

The TSA and COT are two methods that are especially useful in vibration analysis. The TSA allows on to visualise the expected vibration signal from one rotation, while COT allows one to perform an order spectrum analysis on the order tracked signal to visualise the amplitude and/or the phase of the signal with respect to the shaft frequency. The TSA allows one to visualise how the vibration signal looks in one rotation and the averaging is necessary to reduce the effect of noise, which can easily interfere with any analyses we do in the time domain otherwise. COT allows us to identify signal components that are a direct function of the shaft speed, which is useful for identifying mesh frequencies or bearing faults, as these are a function of the shaft speed.

Now, some of you may be wondering: Why are the TSA and COT included in the same section of discussion? Aren't these completely different ideas? To a certain degree, yes, these techniques are quite different. However, in implementation, they share several similar elements and only differ in the latter stages of the implementation. I prefer to think of TSA as a per revolution average of a COT signal. However, we will not focus on this right now, but I will try to allude to this fact at the end of this write-up.

## Time Synchronous Averaging

Time synchronous averaging, conceptually, is a straightforward procedure:
- Determine time points/array indices of tachometer pulses to indicate when a rotation begins or ends.
- Break a signal into \\( N_r \\) segments where a segment contains a shaft rotation and \\( r \\) indicates the rotation number, \\( r = 1, 2, etc \\)
- Construct an average signal by averaging the \\( N_r \\) segments.

Mathematically, this can be given as:

$$
x_{TSA}[n] = \frac{1}{N_r} \sum_{i=0}^{N_r - 1} x[n + iN_s], \text{ where } 1 \leq n \leq Ns,
$$

where \\( N_r \\) is the number of shaft rotations captured by the tachometer and \\( N_s \\) is the number of points per rotation segment. Note that there are two implications that are implicit here: _1)_ the averaging is performed synchronous to a shaft rotation or shaft period, and _2)_ the number of indices per rotation is constant for all rotations. The implication of the former is that we know, a priori, which rotation period is of interest and can extract signal segments that correspond to this period. The implication of the latter is that the shaft speed is perfectly constant, i.e. it does not change through time.

We can make this process tangible by testing it out. For example, consider a noisy sinusoidal signal (\\( F_{sinusoid}= 1Hz \\)) obtained under a sampling frequency of \\( F_s = 10000 Hz \\). In this example, we can assume that \\( F_{sinusoid} = F_{shaft} \\) for simplicity. Thus,  this implies that \\( N_r = 100 \\) (as it takes 1 second to complete a rotation and the signal is 100 seconds long) and \\( N_s = F_s \\). This signal is shown as

![Toy signal example](toy_signal.png)

We can visualise how the TSA changes as a function of the number of averaged rotations. This is shown as

![Toy TSA example](averaging_process.png)

_However_, there is a problem here. If the shaft speed is varying or there is perhaps some natural speed drift or jitter, we are not guaranteed to have the same number of points in the rotation segments. This implies that \\(N_s\\) may not be consistent for each of the \\(N_r\\) segments, which is a different problem to asynchronous averaging.

So, what do we do? The simplest solution is to perform linear interpolation for a set number  of samples \\(N_s\\) for each rotation, thereby ensuring that there are a consistent number of samples for each segment that is representative of a shaft rotation. This process will ensure that we can efficiently determine the TSA of a signal.

Finally, the general TSA process can be given as:
1. Obtain a tachometer signal.
2. Using the tachometer signal, determine the time-point/array indices of a tachometer pulse.
3. If the tachometer has multiple pulses per revolution, take every \\(PPR^{th}\\) time-point/array index to ensure that we have the time-points/array indices that correspond to a rotation.
4. For a signal between rotations \\(r_i\\) and \\(r_{i+1}\\), linearly interpolate a signal for \\(N_s\\) samples to ensure that each of the \\(N_r\\) segments have the same number of points.
5. Average the \\(N_r\\) segments.

_Note: You may be wondering why I keep emphasising that the tachometer pulse points are either given by time-points or an array indices._ The reason for this is: a tachometer is not always guaranteed to be the sampled at the signal sampling rate, thus the array indices of the pulse point may not align between the vibration signal and the tachometer signal. In this case, we need to determine the rotation start and end points using time values, which is simple, and then find the corresponding array index from the vibration signal. *I feel that it is important to be wary of this if you use the TSA in future.*

## Computed Order Tracking

Computed order tracking, in implementation, requires one to do the following:
- Determine time points/array indices of tachometer pulses to indicate when a rotation begins or ends.
- Resample a vibration signal between each rotation to obtain a signal that is sampled with respect to the shaft speed (or as I like to think about it \\( N_s \\) times per revolution). This resampling procedure ensures that we take a sample at constant angular increments.
- Transform the signal using the DFT to obtain the order spectrum (where the sampling frequency \\( F_s \\) is now \\( N_s \\) orders) and inspect the Fourier coefficient magnitude and/or phase.

In the work of Lin and Zhao [1], the following figure is given:

![Lin and Zhao COT example](COT_process.png)

Which gives the example of a sinusoidal signal with what appears to be a linearly increasing frequency. We can think of this in a practical setting as a signal that comes from a machine with a linearly increasing shaft speed and provides an indication of when the tachometer is pulsing. It is simple enough to recreate this signal and process, so let us do exactly that. I will consider a signal with a linearly increasing frequency between \\( 1Hz \\) and \\( 5Hz \\) for a time of 10 seconds. This process gives the following figure

![Chirp example](COT_signal.png)

with the following shaft speed and angular displacement

![angular speed and displacement](COT_speeds.png)

Now, by using the 'tachometer' information (which we can infer from the angular displacement of the shaft), we can begin the COT process. By interpolating the segments for each rotation, the following COT signal can be developed

![COT chirp example](COT_resampled_signal.png)

Note, I also plotted the COT signal with a different time axis, one where I set \\( N_s \\) to \\( F_s \\), to show what the effect of resampling has on the signal. Finally (:grinning_face_with_sweat:), to demonstrate how the frequency spectrum and the order spectrum differ, we can look at both spectrums. This becomes

![COT example spectra](COT_spectrum.png)

Here, we can see that the smearing in the frequency spectrum, and we can see a single order component in the signal at one order, which we expected as we designed to signal to vary with the shaft frequency.

Finally, the general COT process can be given as:
1. Obtain a tachometer signal.
2. Using the tachometer signal, determine the time-point/array indices of a tachometer pulse.
3. If the tachometer has multiple pulses per revolution, take every \\( PPR^{th} \\) time-point/array index to ensure that we have the time-points/array indices that correspond to a rotation.
4. For a signal between rotations \\( r_i \\) and \\( r_{i+1} \\), linearly interpolate a signal for $N_s$ samples to ensure that each of the $N_r$ segments have the same number of points.
5. Investigate the order spectrum.

## The relationship between TSA and COT

If one re-examines the steps for TSA and COT, it is clear that step 5 is the only general step that is different. This conceptual difference is why I stated in the beginning that they share a number of similar steps and only differ in the latter stages of the implementation. I urge you, if you are still confused, to try and re-implement these simple problems by yourself and perhaps label the steps.

For those of you who have reached the end of this document, I hope that you have found it useful. All the best with your exams, I hope that they go well!

As always, thanks for reading! 👨🏼‍💻

## References

[1.] Lin J, Zhao M (2015) A review and strategy for the diagnosis of speed-varying machinery. 2014 Int Conf Progn Heal Manag PHM 2014. https://doi.org/10.1109/ICPHM.2014.7036368
