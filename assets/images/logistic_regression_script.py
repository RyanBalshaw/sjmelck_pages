import os

import numpy as np
from logistic_regression_utils import LogisticRegression, RegType
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs  # , make_classification


def simple_problem():
    # Setup the save directory
    tmp_dir_path = os.path.join(os.getcwd(), "logistic-regression")

    # Linear and sigmoid
    def linear(x, b0=1, b1=1):
        return b0 + b1 * x

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    np.random.seed(1)

    # Sample points for class 0 and class 1
    num_points = 5
    b0 = 1
    b1 = 1
    x_intercept = -b0 / b1
    x_samples_0 = np.random.uniform(x_intercept - 3.5, x_intercept, num_points)
    x_samples_1 = np.random.uniform(x_intercept, x_intercept + 3.5, num_points)
    z_samples_0 = linear(x_samples_0)
    z_samples_1 = linear(x_samples_1)

    x_min = x_intercept - 3.5 - 0.5
    x_max = x_intercept + 3.5 + 0.5

    x = np.linspace(x_min, x_max, 200)
    z = linear(x)
    sig = sigmoid(z)

    color_class_0 = "#FF9999"
    color_class_1 = "#66B2FF"

    fig, axs = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={'width_ratios': [1, 1.3]})
    fig.suptitle("Logistic Regression", fontsize=16)

    # Draw arrow
    fig.text(0.5, 0.85, r'$\Longrightarrow$', fontsize=40, ha='center', va='center')

    # Sub-plot 1: Show linear mapping
    axs[0].set_title("Linear mapping")
    axs[0].plot(x, z, label='$z = b_0 + b_1 x$', lw=2, color='navy')

    # Show points on linear mapping
    axs[0].scatter(x_samples_0, z_samples_0, color="k", marker = "x", s=40, alpha=0.6)
    axs[0].scatter(x_samples_1, z_samples_1, color="k", marker = "x", s=40, alpha=0.6)

    # Plot vertical lines up to linear mapping
    for cnt, (xs_0, xs_1, zs_0, zs_1) in enumerate(zip(x_samples_0, x_samples_1, z_samples_0, z_samples_1)):
        axs[0].plot([xs_0, xs_0], [0, zs_0], linestyle = "--", color = "k")
        axs[0].plot([xs_1, xs_1], [0, zs_1], linestyle="--", color="k")

    # Plot scatter of x points for two classes
    axs[0].scatter(x_samples_0, np.zeros_like(x_samples_0), color=color_class_0, s=55, label='Class 0 samples')
    axs[0].scatter(x_samples_1, np.zeros_like(x_samples_0), color=color_class_1, s=55, label='Class 1 samples')

    axs[0].set_xlabel(r'$x$', fontsize=14)
    axs[0].set_ylabel(r'$z$', fontsize=14)
    axs[0].set_xlim([x_min, x_max])
    axs[0].set_ylim([-4, 4])

    axs[0].axhline(0, color='black', lw=1)
    # axs[0].axvline(0, color='black', lw=1)

    axs[0].spines['right'].set_visible(False)
    axs[0].spines['top'].set_visible(False)
    # axs[0].spines['left'].set_position('zero')
    axs[0].spines['bottom'].set_position('zero')
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    axs[0].legend(loc='upper left', fontsize=10, frameon=False)

    # Subplot 2: Show distribution mapping
    fig.subplots_adjust(wspace=0.12)
    axs[1].set_title("Probability mass function")

    # ========== RIGHT: Sigmoid + Classes ==========
    axs[1].plot(x, sig, lw=2, color='m')

    # Plot samples
    axs[1].scatter(x_samples_0, np.zeros_like(x_samples_0), color=color_class_0, s=55, label='Class 0 samples')
    axs[1].scatter(x_samples_1, np.zeros_like(x_samples_0), color=color_class_1, s=55, label='Class 1 samples')

    # Shade class areas
    axs[1].fill_between(x, 0, sig, where=(sig < 0.5), color=color_class_0, alpha=0.16)
    axs[1].fill_between(x, sig, 1, where=(sig > 0.5), color=color_class_1, alpha=0.16)

    # Annotate class areas
    axs[1].text(-4, 0.25, 'Class 0 region', color=color_class_0, fontsize=12)
    axs[1].text(1.5, 0.85, 'Class 1 region', color=color_class_1, fontsize=12)

    axs[1].set_xlabel(r'$x$', fontsize=14)
    axs[1].set_ylabel(r'', fontsize=14)
    axs[1].set_xlim([x_min, x_max])
    axs[1].set_ylim([-0.1, 1.1])
    axs[1].axhline(0, color='black', lw=1)
    axs[1].axhline(1, color='black', lw=1)
    # axs[1].axvline(0, color='black', lw=1)
    axs[1].set_xticks([])
    axs[1].set_yticks([0, 0.5, 1])

    axs[1].spines['right'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    # axs[1].spines['left'].set_position('zero')
    axs[1].spines['bottom'].set_position('zero')

    # [right plot]: Add a dashed line at decision boundary
    decision_boundary_x = x_intercept
    axs[1].axvline(decision_boundary_x, color='grey', lw=1, ls='--', alpha=0.6)
    axs[1].axhline(0.5, color='grey', lw=1, ls='--', alpha=0.6)
    axs[1].text(decision_boundary_x + 0.35, 0.525, 'threshold', color="grey", fontsize=9)

    # [y-axis label, right plot]
    axs[1].set_ylabel(r'$p(y=1|x)$', fontsize=14)

    plt.tight_layout(rect=[0.01, 0, 1, 1])
    plt.savefig(os.path.join(tmp_dir_path, "summary_figure.png"))
    plt.show()

    # X, y = make_classification(n_features=2, n_informative=2, n_redundant=0)
    X, y = make_blobs(
        n_samples=400, n_features=2, centers=2, cluster_std=1, random_state=3
    )
    # X, y = make_circles(n_samples=1000,noise=0.01, random_state=0)

    # plot 1
    plt.figure(figsize=(10, 6))
    colors = ListedColormap([colors_class_0, colors_class_1])

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=colors, edgecolors="k", s=50, alpha=0.8)
    plt.title("Binary Classification Dataset (make_blobs)", fontsize=14)
    plt.xlabel("Feature 1", fontsize=12)
    plt.ylabel("Feature 2", fontsize=12)
    plt.grid(alpha=0.3)

    for class_id, class_name in enumerate(["Class 0", "Class 1"]):
        plt.scatter([], [], c=colors(class_id), label=class_name, edgecolors="k", s=50)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(tmp_dir_path, "dataset.png"))
    plt.show()

    model = LogisticRegression(
        learning_rate=0.1,
        max_iter=400,
        regulariser_type=RegType.L1,
        alpha=0.001,
        optimiser="gradient_descent",  #
        tol=1e-4,
    )
    model.fit(X, y)

    # Plot 2
    plt.figure(figsize=(10, 5))
    plt.plot(model._loss_values, color="#1f77b4", linewidth=2)
    plt.title("Loss Convergence During Training", fontsize=14)
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel("Loss Value", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(tmp_dir_path, "training_loss.png"))
    plt.show()

    X_grid, Y_grid = np.meshgrid(
        np.linspace(X[:, 0].min(), X[:, 0].max(), 100),
        np.linspace(X[:, 1].min(), X[:, 1].max(), 100),
    )
    X_test = np.hstack([X_grid.ravel().reshape(-1, 1), Y_grid.ravel().reshape(-1, 1)])

    D = model.predict_proba(X_test)[:, 0]
    Z = model.predict_log_odds(X_test)

    # Plot 3
    fig = plt.figure(figsize=(20, 9))

    # First subplot for Log-Odds
    ax1 = fig.add_subplot(121, projection="3d")  # 1 row, 2 cols, position 1

    surf1 = ax1.plot_surface(
        X_grid,
        Y_grid,
        Z.reshape(100, 100),
        cmap="viridis",
        alpha=0.8,
        linewidth=0,
        antialiased=True,
    )

    for i, (x_i, y_i) in enumerate(zip(X[:, 0], X[:, 1])):
        if y[i] == 0:
            ax1.scatter([x_i], [y_i], [0], color="blue", s=30, edgecolor="k")
        else:
            ax1.scatter([x_i], [y_i], [0], color="red", s=30, edgecolor="k")

    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5, label="Log-Odds")

    ax1.set_xlabel("Feature 1", fontsize=14)
    ax1.set_ylabel("Feature 2", fontsize=14)
    ax1.set_zlabel("Log-Odds Value", fontsize=14)
    ax1.set_title("Log-Odds Surface with Decision Boundary", fontsize=16)

    # Second subplot for Probability (D)
    ax2 = fig.add_subplot(122, projection="3d")

    surf2 = ax2.plot_surface(
        X_grid,
        Y_grid,
        D.reshape(100, 100),
        cmap="RdYlBu_r",
        alpha=0.8,
        linewidth=0,
        antialiased=True,
    )

    # Add data points - project them onto z=0 or z=1 based on their class
    for i, (x_i, y_i) in enumerate(zip(X[:, 0], X[:, 1])):
        z_i = 0 if y[i] == 0 else 1  # Project to bottom or top
        if y[i] == 0:
            ax2.scatter([x_i], [y_i], [z_i], color="blue", s=30, edgecolor="k")
        else:
            ax2.scatter([x_i], [y_i], [z_i], color="red", s=30, edgecolor="k")

    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5, label="Probability of Class 1")

    ax2.set_xlabel("Feature 1", fontsize=14)
    ax2.set_ylabel("Feature 2", fontsize=14)
    ax2.set_zlabel("Probability", fontsize=14)
    ax2.set_title("Probability Surface with Decision Boundary", fontsize=16)
    ax2.set_zlim(0, 1)

    plt.tight_layout(pad=4.0)

    def update(i):
        ax1.view_init(elev=None, azim=i)
        ax2.view_init(elev=None, azim=i)

    anim = FuncAnimation(
        fig,
        update,
        frames=np.arange(0, 360 + 1, 1),
        interval=50,
        cache_frame_data=False,
        repeat=False,
    )

    writergif = PillowWriter(fps=20)
    anim.save(os.path.join(tmp_dir_path, "animation_prob_class_1.gif"), writergif)

    plt.show()

    # gradient_error, hessian_error = model.check_gradients_and_hessian(X, y)


if __name__ == "__main__":
    simple_problem()
