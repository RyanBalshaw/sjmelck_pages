---
title: "Setting Up My First Post"
date: 2022-08-15T14:05:51+02:00
description: ""
draft: false
---

# How to get really nice plots

## Minimal working example:

```python
from matplotlib import pyplot as plt
import numpy as np


def fix(ax=None, minor_flag=True, flag_3d=True):
    if ax is None:
        ax = plt.gca()

    fig = ax.get_figure()
    # Force the figure to be drawn

    fig.canvas.draw()

    if minor_flag:
        # Remove '\mathdefault' from all minor tick labels
        labels = [
            label.get_text()
            .replace("\\times", "}$.$\\mathdefault{")
            .replace("-", "$\\mathdefault{-}$")
            .replace("−", "$\\mathdefault{-}$")
            for label in ax.get_xminorticklabels()
        ]
        ax.set_xticklabels(labels, minor=True)

        labels = [
            label.get_text()
            .replace("\\times", "}$.$\\mathdefault{")
            .replace("-", "$\\mathdefault{-}$")
            .replace("−", "$\\mathdefault{-}$")
            for label in ax.get_yminorticklabels()
        ]
        ax.set_yticklabels(labels, minor=True)

        if flag_3d:
            labels = [
                label.get_text()
                .replace("\\times", "}$.$\\mathdefault{")
                .replace("-", "$\\mathdefault{-}$")
                .replace("−", "$\\mathdefault{-}$")
                for label in ax.get_zminorticklabels()
            ]
            ax.set_zticklabels(labels, minor=True)

    else:
        # Remove '\mathdefault' from all minor tick labels
        labels = [
            label.get_text()
            .replace("\\times", "}$.$\\mathdefault{")
            .replace("-", "$\\mathdefault{-}$")
            .replace("−", "$\\mathdefault{-}$")
            for label in ax.get_xmajorticklabels()
        ]
        ax.set_xticklabels(labels, minor=False)

        labels = [
            label.get_text()
            .replace("\\times", "}$.$\\mathdefault{")
            .replace("-", "$\\mathdefault{-}$")
            .replace("−", "$\\mathdefault{-}$")
            for label in ax.get_ymajorticklabels()
        ]
        ax.set_yticklabels(labels, minor=False)

        if flag_3d:
            labels = [
                label.get_text()
                .replace("\\times", "}$.$\\mathdefault{")
                .replace("-", "$\\mathdefault{-}$")
                .replace("−", "$\\mathdefault{-}$")
                for label in ax.get_zmajorticklabels()
            ]
            ax.set_zticklabels(labels, minor=False)


fig, ax = plt.subplots(figsize = (10, 6))

x_range = np.linspace(-10, 10, 1000)
ax.plot(x_range, x_range**2, label = r"$y=x^2$")
ax.set_title("Base example (no nice plots)")

ax.legend()
ax.set_xlabel("Record number")
ax.set_ylabel("$y$")

plt.show(block=False)

#Override parameters
plt.rc("mathtext", fontset="cm")
plt.rc("font", family="serif", size=12, serif="cmr10")

fig, ax = plt.subplots(1, 2, figsize = (10, 6))
fig.suptitle("The ideal plot (with and without fix)")
ax = ax.flatten()

x_range = np.linspace(-10, 10, 1000)
ax[0].plot(x_range, x_range**2, label = r"$y=x^2$")
ax[1].plot(x_range, x_range**2, label = r"$y=x^2$")

ax[0].set_title("Figure without fix")
ax[1].set_title("Figure with fix")

for axs in ax:
    axs.legend()
    axs.set_xlabel("$x$")
    axs.set_ylabel("$y$")

#Fix the second figure
fix(ax[1], minor_flag=False, flag_3d=False)

plt.show(block=True)
```

$$ y = x^2 $$ - Markdown does not work (SAD)!

https://www.morch.com/posts/2021-07-24-mathjax-in-hugo/
