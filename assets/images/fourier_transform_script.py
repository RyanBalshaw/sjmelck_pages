# Define imports

import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Setup the save directory
tmp_dir_path = os.path.join(os.getcwd(), "the-fourier-transform-explained")

########################################################################################
# Setup figure 1
########################################################################################
t_sawtooth = np.arange(0, 1, 1 / 100)
g_sawtooth = t_sawtooth - np.floor(t_sawtooth)

N = 8
vis_mat = np.zeros(
    (N + 1, len(t_sawtooth))
)  # plus 1 to make sure N is the number of sinusoids
vis_mat[0, :] = 1 / 2

for i in range(1, N + 1, 1):
    vis_mat[i, :] = -1 / (np.pi * i) * np.sin(2 * np.pi * i * t_sawtooth)

# Frequency
freq_mag = np.arange(0, N + 1, 1 / 100)
amp_mag = np.zeros_like(freq_mag)
amp_mag[range(0, len(freq_mag), 100)] = np.max(vis_mat, axis=1)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

ax.plot(
    t_sawtooth,
    np.ones_like(t_sawtooth) * -1,
    np.sum(vis_mat, axis=0),
    color="#003f5c",
    label="approximated signal",
)
ax.plot(
    np.ones_like(freq_mag) * 3,
    freq_mag,
    amp_mag,
    color="#ffa600",
    label="signal amplitude",
)

for cnt in range(1, N + 2, 1):
    i = cnt - 1
    ax.plot(
        t_sawtooth,
        np.ones_like(t_sawtooth) * (i),
        vis_mat[i, :],
        alpha=0.8,
        color="#bc5090",
    )

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.set_xlabel("Time")
ax.set_ylabel("Integer component amplitude")
ax.set_zlabel("signal magnitude")

ax.legend()


def update(i):
    ax.view_init(elev=20, azim=i)


anim = FuncAnimation(
    fig,
    update,
    frames=np.arange(0, 360 + 1, 1),
    interval=50,
    cache_frame_data=False,
    repeat=False,
)

writergif = PillowWriter(fps=20)
anim.save(os.path.join(tmp_dir_path, "animation_fourier_series.gif"), writergif)

plt.show()
########################################################################################
# Figure 2
########################################################################################
############
# ASSUMPTIONS
############

t1 = 0  # start time
t2 = 4  # end time
Fs = 1000  # Sampling frequency
offset = 1  # DC offset

freq1 = 2
freq2 = 3

t = np.arange(t1, t2, 1 / int(t2 * Fs))
g_function = (
    1 / 4 * np.cos(2 * np.pi * freq1 * t) + np.cos(freq2 * 2 * np.pi * t) + offset
)

plt.figure()
plt.title("The function g(t)")
plt.plot(t, g_function)
plt.xlabel("Time (s)")
plt.grid()
plt.savefig(os.path.join(tmp_dir_path, "function.png"))
plt.show()

########################################################################################
# Figure 3
########################################################################################

# create your own definition of 1 revolution

########################
# MANUALLY CHOSEN FEATURE (you can edit f to be whatever you want)
########################
f0 = 2.5
########################
theta_range = f0 * 2 * np.pi * t
fig2, ax2 = plt.subplots()

x = g_function * np.cos(theta_range)
y = g_function * np.sin(theta_range)

ax2.set_title(r"Sequential wrapping: $f_0$ = {}".format(f0))
ax2.set_xlim([-2.5, 2.5])
ax2.set_ylim([-2.5, 2.5])
ax2.grid()
ax2.set_xlabel("x")
ax2.set_xlabel("y")

(line,) = ax2.plot(x, y, zorder=10, color="b", linestyle="--")
line_com = ax2.scatter(0, 0, marker="x", color="r", zorder=20, label="center of mass")
ax2.legend()

annotation = ax2.annotate(
    "", xy=(0, 0), xytext=(x[0], y[0]), arrowprops={"arrowstyle": "<-"}
)


def update2(i, line, annotation):
    line.set_data(x[:i], y[:i])
    annotation.set_position((x[i], y[i]))

    line_com.set_offsets([np.mean(x[: i + 1]), np.mean(y[: i + 1])])

    # return line, annotation,


anim2 = FuncAnimation(
    fig2,
    update2,
    frames=range(1, len(theta_range) - 1, 100),
    fargs=[line, annotation],
    interval=0.1,
    blit=False,
    cache_frame_data=False,
    repeat=False,
)

writergif2 = PillowWriter(fps=200)
anim2.save(os.path.join(tmp_dir_path, "animation_fourier_wrapping.gif"), writergif2)

plt.show()

########################################################################################
# Figure 4
########################################################################################
plt.close("all")
# Crucial to use widget for ipywidgets (in every cell you use the magic line, crashes if
# you use %matplotlib notebook or anything like that)
# plot as magnitude and phase
fig, ax = plt.subplots(2, 3, figsize=(10, 8))
ax = ax.flatten()

f0_vals = [0, 1, 2, 3, 4, 5]
# define figure 1 - For cartesian

lines = []
coms = []

for cnt, axs in enumerate(ax):
    axs.set_xlim((-2.5, 2.5))
    axs.set_ylim((-2.5, 2.5))
    axs.grid()
    axs.set_title(f"$f_0 = {f0_vals[cnt]}$")
    axs.set_xlabel("x")
    axs.set_ylabel("y")

    # plot a unit circle
    axs.plot(np.linspace(-1, 1, 50), np.sqrt(1 - np.linspace(-1, 1, 50) ** 2), "k--")
    axs.plot(
        np.linspace(-1, 1, 50), -1 * np.sqrt(1 - np.linspace(-1, 1, 50) ** 2), "k--"
    )

    x = g_function * np.cos(2 * np.pi * f0_vals[cnt] * t)
    y = g_function * np.sin(2 * np.pi * f0_vals[cnt] * t)

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    axs.plot(x, y, zorder=10, color="b", linestyle="--")
    axs.scatter(
        [mean_x], [mean_y], marker="x", color="r", label="center of mass", zorder=20
    )
    axs.legend()

fig.tight_layout()
plt.savefig(os.path.join(tmp_dir_path, "different_wraps_and_com.png"))
plt.show()

########################################################################################
# Figure 5
########################################################################################
plt.close("all")

real = []
imag = []
radius = []
angle = []

f_start = -5
f_end = 5
granularity = 1000
f_range = np.linspace(f_start, f_end, granularity)

for f0 in f_range:
    theta_range = 2 * np.pi * f0 * t

    x = g_function * np.cos(theta_range)
    y = g_function * np.sin(theta_range)

    mu_x = np.mean(x)
    mu_y = np.mean(y)

    real.append(mu_x)
    imag.append(mu_y)

    radius.append(np.sqrt(mu_x**2 + mu_y**2))
    angle.append(np.arctan2(mu_y, mu_x))

fig3, ax3 = plt.subplots(2, 2, figsize=(10, 10))
ax3 = ax3.flatten()

ax3[0].set_title("x component")
ax3[1].set_title("y component")
ax3[2].set_title("Vector magnitude")
ax3[3].set_title("vector angle")

for axs in ax3:
    axs.set_xlabel("Frequency (Hz)")

(line1,) = ax3[0].plot([], [], color="k")
(line2,) = ax3[1].plot([], [], color="k")
(line3,) = ax3[2].plot([], [], color="k")
(line4,) = ax3[3].plot([], [], color="k")

data_tup = (real, imag, radius, angle)
line_tup = (line1, line2, line3, line4)


def animate_test(num, f_range, data_tup, line_tup):
    for i in range(len(data_tup)):
        line_tup[i].set_data(f_range[:num], data_tup[i][:num])
        line_tup[i].axes.axis(
            [f_start, f_end, np.min(data_tup[i]), np.max(data_tup[i])]
        )

    return (line_tup,)


fig.tight_layout()

anim3 = FuncAnimation(
    fig3,
    animate_test,
    range(0, len(f_range), 10),
    fargs=[f_range, data_tup, line_tup],
    interval=1.5,
    blit=False,
    repeat=False,
)
# lowering interval will speed up the animation

writergif3 = PillowWriter(fps=20)
anim3.save(os.path.join(tmp_dir_path, "animation_real_imag_components.gif"), writergif3)

plt.show()

########################################################################################
# Figure 6
########################################################################################
f = 5
data_forward = np.exp(1j * 2 * np.pi * f * t)
data_backward = np.exp(-1j * 2 * np.pi * f * t)

ax1_x = np.real(data_forward)
ax1_y = np.imag(data_forward)

ax2_x = np.real(data_backward)
ax2_y = np.imag(data_backward)

fig4 = plt.figure(figsize=(9, 9))
ax4 = fig4.add_subplot(1, 2, 1, projection="3d")
ax4.set_title("anti-clockwise rotation")
ax4.set_xlabel("Time")
ax4.set_ylabel("Real")
ax4.set_zlabel("Imaginary")
ax4.set_xticks([])
ax4.set_yticks([])
ax4.set_zticks([])
(line1_1,) = ax4.plot(
    t, ax1_x, ax1_y, zorder=30, color="#003f5c", label="complex representation"
)
(line1_2,) = ax4.plot(
    t, ax1_x, [0] * len(t), zorder=20, color="#bc5090", label="real representation"
)
(line1_3,) = ax4.plot(
    t, [0] * len(t), ax1_y, zorder=10, color="#ffa600", label="imaginary representation"
)
ax4.legend()

ax5 = fig4.add_subplot(1, 2, 2, projection="3d")
ax5.set_title("clockwise rotation")
ax5.set_xlabel("Time")
ax5.set_ylabel("Real")
ax5.set_zlabel("Imaginary")
ax5.set_xticks([])
ax5.set_yticks([])
ax5.set_zticks([])
(line2_1,) = ax5.plot(t, ax2_x, ax2_y, zorder=30, color="#003f5c")
(line2_2,) = ax5.plot(t, ax2_x, [0] * len(t), zorder=20, color="#bc5090")
(line2_3,) = ax5.plot(t, [0] * len(t), ax2_y, zorder=10, color="#ffa600")

fig4.tight_layout(w_pad=4)


def animate_f_differences(num, line1_1, line1_2, line1_3, line2_1, line2_2, line2_3):
    line1_1.set_data(t[:num], ax1_x[:num])
    line1_1.set_3d_properties(ax1_y[:num])

    line1_2.set_data(t[:num], ax1_x[:num])
    line1_2.set_3d_properties([0] * num)

    line1_3.set_data(t[:num], [0] * num)
    line1_3.set_3d_properties(ax1_y[:num])

    line2_1.set_data(t[:num], ax2_x[:num])
    line2_1.set_3d_properties(ax2_y[:num])

    line2_2.set_data(t[:num], ax2_x[:num])
    line2_2.set_3d_properties([0] * num)

    line2_3.set_data(t[:num], [0] * num)
    line2_3.set_3d_properties(ax2_y[:num])

    return (
        line1_1,
        line1_2,
        line1_3,
        line2_1,
        line2_2,
        line2_3,
    )


anim4 = FuncAnimation(
    fig4,
    animate_f_differences,
    range(0, len(t), 100),
    fargs=[line1_1, line1_2, line1_3, line2_1, line2_2, line2_3],
    interval=0.01,
    blit=False,
    repeat=False,
)

writergif4 = PillowWriter(fps=200)
anim4.save(os.path.join(tmp_dir_path, "animation_rotation.gif"), writergif4)

plt.show()
