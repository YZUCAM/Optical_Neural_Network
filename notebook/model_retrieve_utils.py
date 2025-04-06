import typing
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_loss_curves(results):
    """plot training curves"""
    loss = results["train_loss"]
    # test_loss = results["test_loss"]
    accuracy = results["train_acc"]
    # test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))
    # plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    # plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    # plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()



def plot_complex_image(
    image: np.ndarray,
    title: str = None,
    show: bool = True,
    filename: str = None,
    angle_twopi: bool = True,
    dx: float = None,
    return_fig: bool = False,
):

    titles = ["Intensity", "Angle", "Real", "Imaginary"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    field_plot = [np.abs(image), np.angle(image), np.real(image), np.imag(image)]

    if dx is not None:
        grid_length = dx * (image.size(-1) - 1)
        extent = [-grid_length / 2, grid_length / 2, -grid_length / 2, grid_length / 2]
    else:
        extent = None

    for i, ax in enumerate(axes.flat):
        im = ax.imshow(field_plot[i], extent=extent)
        ax.set_title(titles[i])

        # Create a divider for the existing axes instance
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        # Add colorbar to the created axes
        fig.colorbar(im, cax=cax, orientation="vertical")

    # Setting the main title for the figure
    fig.suptitle(title, y=0.92)  # Adjust y to move the title down

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.4, hspace=0)  # adjust only the width spacing

    if angle_twopi:
        axes[0, 1].get_images()[0].set_clim(-np.pi, np.pi)
        axes[0, 1].get_images()[0].set_cmap("twilight_shifted")
        axes[0, 1].get_images()[0].set_interpolation("none")

    if filename is not None:
        if not os.path.splitext(filename)[1]:
            filename += ".pdf"
        fig.savefig(filename, bbox_inches="tight", pad_inches=0.02, dpi=500)

    if show:
        plt.show()

    if return_fig:
        return fig


# create dummy sample
def create_dummy_sample(dataloader: torch.utils.data.DataLoader, num_of_data: int) -> torch.Tensor:
    X, y = next(iter(dataloader))
    if not num_of_data > len(X):    
        X_dummy = X[:num_of_data]
        y_dummy = y[:num_of_data]
    else:
        print(f"number of data selected are out of size, decrease the number of data.")
    return X_dummy, y_dummy