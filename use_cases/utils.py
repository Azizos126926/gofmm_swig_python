import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image, offsetbox

def plot_embedding(X, y, digits, title=None):
    """Scale and visualize the embedding vectors.

    X
        Data set of images of shape ( problem_size, 784)
    y
        Target labels of shape ( problem_size, )
    title
        Title of the plot
    
    """
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure(figsize=[10, 10])
    ax = plt.subplot(111)

    for i in range(X.shape[0]):
        plt.text(
            X[i, 0],
            X[i, 1],
            str(y[i]),
            color=plt.cm.Set1(y[i] / 10.0),
            fontdict={"weight": "bold", "size": 9},
        )

    if hasattr(offsetbox, "AnnotationBbox"):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1.0, 1.0]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits[i], cmap=plt.cm.gray_r), X[i]
            )
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])

    if title is not None:
        plt.title(title)
    plt.savefig("digits.png")

def sort_eigen_pairs( evals, evecs, basis_change_matrix ):
    """Sorts eigen values in descending order and
    orders eigen vectors in the corresponding order.
    Also does math similar to datafold methods.

    evals
        Eigen values to be sorted
    evecs
        Eigen vectors to be sorted
    basis_change_matrix
        The changed basis obtained from datafold
    
    """
    sort_order = np.argsort( evals )
    sort_order = sort_order[::-1]
    evals[:] = evals[sort_order]
    evecs[:,:] = evecs[:,sort_order]

    evecs[:,:] = basis_change_matrix @ evecs
    evecs[:,:] /= np.linalg.norm(evecs, axis=0)[np.newaxis, :]

