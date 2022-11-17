import sys
import time

import logging
import os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import gzip
import numpy as np
from matplotlib import image, offsetbox
import scipy.sparse.linalg
from scipy.linalg import eig, eigh
import struct

import datafold.pcfold as pfold
from datafold.dynfold import DiffusionMaps
from datafold.utils.plot import plot_pairwise_eigenvector
from datafold.pcfold.kernels import PCManifoldKernel

import full_matrix

def plot_embedding(X, y, digits, title=None):
    """Scale and visualize the embedding vectors"""
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
    plt.savefig('mnist_digits.png')

executable = "./test_gofmm"
problem_size = 1024
max_leaf_node_size = 512
num_of_neighbors = 0
max_off_diagonal_ranks = 512
num_rhs = 1
user_tolerance = 1E-7
computation_budget = 0.00
distance_type = "kernel"
matrix_type = "dense"
kernel_type = "gaussian"

num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") #/ 255

# Make sure images have shape (28, 28, 1)
#x_train = np.expand_dims(x_train, -1)
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print(x_train.shape[0], "train samples")

images = x_train[0:problem_size,:,:]
y = y_train[0:problem_size]

X = images
print("X before reshape", X.shape)
X = images.reshape(problem_size, 784)
print("X sfter reshape", X)
print("X shape:", X.shape)
print("y shape:", y.shape)

X_pcm = pfold.PCManifold(X)
X_pcm.optimize_parameters(result_scaling=2)

print(f"epsilon={X_pcm.kernel.epsilon}, cut-off={X_pcm.cut_off}")

t0 = time.time()
dmap = DiffusionMaps(
    kernel=pfold.GaussianKernel(epsilon=X_pcm.kernel.epsilon),
    n_eigenpairs=6,
    dist_kwargs=dict(cut_off=X_pcm.cut_off),
)

dmap = dmap.fit(X_pcm)
dmap = dmap.set_target_coords([1, 2])
X_dmap = dmap.transform(X_pcm)

# Mapping of diffusion maps
plot_embedding(
    X_dmap,
    y,
    images,
    title="Diffusion map embedding of the digits (time %.2fs)" % (time.time() - t0),
)

dmap = DiffusionMaps(
    kernel=pfold.GaussianKernel(epsilon=X_pcm.kernel.epsilon),
    n_eigenpairs=6,
    dist_kwargs=dict(cut_off=X_pcm.cut_off),
)
dmap = dmap.fit(X_pcm)
plot_pairwise_eigenvector(
    eigenvectors=dmap.eigenvectors_[:, 1:],
    n=0,
    idx_start=1,
    fig_params=dict(figsize=(10, 10)),
    scatter_params=dict(c=y),
)

plt.savefig('mnist_digits_dmap.png')

pcm = pfold.PCManifold(X, 
                        kernel=pfold.DmapKernelFixed(internal_kernel=pfold.GaussianKernel(epsilon=24.44322087308319), is_stochastic=True, alpha=1, symmetrize_kernel=True),
                        dist_kwargs=dict(cut_off=21.219348907470703, kmin=0, backend= "guess_optimal"))

kernel_output = pcm.compute_kernel_matrix()
( kernel_matrix, cdist_kwargs, ret_extra, ) = PCManifoldKernel.read_kernel_output(kernel_output=kernel_output)


kernel_matrix_sparse = kernel_matrix.copy()
kernel_matrix_sparse = kernel_matrix_sparse.asfptype()
kernel_matrix = kernel_matrix.todense()
kernel_matrix = kernel_matrix.astype("float32")
#kernel_matrix.tofile("KernelMatrix_32768.bin")
weights = np.ones((problem_size, num_rhs))      


kernel_matrix_OP = full_matrix.FullMatrix( executable, problem_size, max_leaf_node_size,
                            num_of_neighbors, max_off_diagonal_ranks, num_rhs, user_tolerance, computation_budget,
                            distance_type, matrix_type, kernel_type, kernel_matrix, weights, dtype=np.float32 )
print("weights shape",weights.shape)
print("K shape",kernel_matrix.shape)

n_eigenpairs = 6
solver_kwargs = {
    "k": n_eigenpairs,
    "which": "LM",
    "v0": np.ones(problem_size),
    "tol": 1e-14,
    "sigma": 1.1, 
    "mode": "normal"
}

basis_change_matrix = ret_extra['basis_change_matrix']
inv_basis_change_matrix = scipy.sparse.diags(np.reciprocal(basis_change_matrix.data.ravel()))

evals_all, evecs_all = scipy.sparse.linalg.eigsh(kernel_matrix_sparse, **solver_kwargs)
evals_large, evecs_large = scipy.sparse.linalg.eigsh(kernel_matrix_OP, **solver_kwargs)

sort_scipy = np.argsort( evals_all )
sort_scipy = sort_scipy[::-1]
sorted_scipy_evals = evals_all[sort_scipy]
sorted_scipy_evecs = evecs_all[:,sort_scipy]

sort_gofmm = np.argsort( evals_large )
sort_gofmm = sort_gofmm[::-1]
sorted_gofmm_evals = evals_large[sort_gofmm]
sorted_gofmm_evecs = evecs_large[:,sort_gofmm]

sorted_gofmm_evecs = basis_change_matrix @ sorted_gofmm_evecs
sorted_gofmm_evecs /= np.linalg.norm(sorted_gofmm_evecs, axis=0)[np.newaxis, :]

sorted_scipy_evecs = basis_change_matrix @ sorted_scipy_evecs
sorted_scipy_evecs /= np.linalg.norm(sorted_scipy_evecs, axis=0)[np.newaxis, :]

print("eigenvalues of gofmm")
print(sorted_gofmm_evals)
print("eigenvectors of gofmm sorted")
print(sorted_gofmm_evecs)
print("eigenvalues of scipy")
print(sorted_scipy_evals)
print("eigenvectors of scipy")
print(sorted_scipy_evecs)
print("eigenvalues of datafold")
print(dmap.eigenvalues_)
print("eigenvectors of datafold")
print(dmap.eigenvectors_)

plot_pairwise_eigenvector(
    eigenvectors=sorted_scipy_evecs[:, 1:],
    n=0,
    idx_start=1,
    fig_params=dict(figsize=(10, 10)),
    scatter_params=dict(c=y),
)
plt.savefig('mnist_digits_scipy.png')
plot_pairwise_eigenvector(
    eigenvectors=sorted_gofmm_evecs[:, 1:],
    n=0,
    idx_start=1,
    fig_params=dict(figsize=(10, 10)),
    scatter_params=dict(c=y),
)
plt.savefig('mnist_digits_gofmm.png')
