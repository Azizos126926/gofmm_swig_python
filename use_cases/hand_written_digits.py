import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import scipy.sparse.linalg
from scipy.linalg import eig, eigh
from sklearn.model_selection import train_test_split

import datafold.pcfold as pfold
from datafold.dynfold import DiffusionMaps
from datafold.utils.plot import plot_pairwise_eigenvector
from datafold.pcfold.kernels import PCManifoldKernel

import full_matrix
from utils import plot_embedding
from utils import sort_eigen_pairs

# Source code taken and adapted from https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html

"""Input variables required for the instantiation of FullMatrix"""
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

"""Loading the hand written digits from sklearn"""
digits = datasets.load_digits(n_class=6)
X = digits.data[0:problem_size,:]
y = digits.target[0:problem_size]
images = digits.images[0:problem_size,:,:]

X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(
    X, y, images, train_size=2 / 3, test_size=1 / 3
)

""""Instantiation of point cloud data and find the manifold using DiffusionMaps"""
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

dmap = DiffusionMaps(
    kernel=pfold.GaussianKernel(epsilon=X_pcm.kernel.epsilon),
    n_eigenpairs=6,
    dist_kwargs=dict(cut_off=X_pcm.cut_off),
)
dmap = dmap.fit(X_pcm)

"""Compute the same kernel matrix with the same optimized datafold parameters, for the instantiation of FullMatrix"""
pcm = pfold.PCManifold(X, 
                        kernel=pfold.DmapKernelFixed(internal_kernel=pfold.GaussianKernel(epsilon=378.0533464967807), is_stochastic=True, alpha=1, symmetrize_kernel=True),
                        dist_kwargs=dict(cut_off=83.45058418010026, kmin=0, backend= "guess_optimal"))

kernel_output = pcm.compute_kernel_matrix()
( kernel_matrix, cdist_kwargs, ret_extra, ) = PCManifoldKernel.read_kernel_output(kernel_output=kernel_output)

"""Convert the kernel matrix to dense matrix type"""
kernel_matrix_sparse = kernel_matrix.copy()
kernel_matrix_sparse = kernel_matrix_sparse.asfptype()
kernel_matrix = kernel_matrix.todense()
kernel_matrix = kernel_matrix.astype("float32")
#kernel_matrix.tofile("KernelMatrix_32768.bin")
weights = np.ones((problem_size, num_rhs))      


"""Instantiation of FullMatrix"""
kernel_matrix_OP = full_matrix.FullMatrix( executable, problem_size, max_leaf_node_size,
                            num_of_neighbors, max_off_diagonal_ranks, num_rhs, user_tolerance, computation_budget,
                            distance_type, matrix_type, kernel_type, kernel_matrix, weights, dtype=np.float32 )

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

evals_all, evecs_all = scipy.sparse.linalg.eigsh(kernel_matrix_sparse, **solver_kwargs)
evals_large, evecs_large = scipy.sparse.linalg.eigsh(kernel_matrix_OP, **solver_kwargs)

sort_eigen_pairs( evals_all, evecs_all, basis_change_matrix )
sort_eigen_pairs( evals_large, evecs_large, basis_change_matrix )

"""Print eigen pairs and plot hand written digits, eigen vector comparisons"""
print("eigenvalues of gofmm")
print(evals_large)
print("eigenvectors of gofmm sorted")
print(evecs_large)
print("eigenvalues of scipy")
print(evals_all)
print("eigenvectors of scipy")
print(evecs_all)
print("eigenvalues of datafold")
print(dmap.eigenvalues_)
print("eigenvectors of datafold")
print(dmap.eigenvectors_)

plot_pairwise_eigenvector(
    eigenvectors=evecs_all[:, 1:],
    n=0,
    idx_start=1,
    fig_params=dict(figsize=(10, 10)),
    scatter_params=dict(c=y),
)
plt.savefig('hr_digits_scipy.png')
plot_pairwise_eigenvector(
    eigenvectors=evecs_large[:, 1:],
    n=0,
    idx_start=1,
    fig_params=dict(figsize=(10, 10)),
    scatter_params=dict(c=y),
)
plt.savefig('hr_digits_gofmm.png')

plot_pairwise_eigenvector(
    eigenvectors=dmap.eigenvectors_[:, 1:],
    n=0,
    idx_start=1,
    fig_params=dict(figsize=(10, 10)),
    scatter_params=dict(c=y),
)
plt.savefig('hr_digits_dmap.png')

plot_embedding(
    X_dmap,
    y,
    images,
    title="Diffusion map embedding of the digits (time %.2fs)" % (time.time() - t0),
)


