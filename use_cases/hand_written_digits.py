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
from plot_embedding import plot_embedding

# Source code taken and adapted from https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html

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

digits = datasets.load_digits(n_class=6)
X = digits.data[0:problem_size,:]
y = digits.target[0:problem_size]
images = digits.images[0:problem_size,:,:]
print("X",X)
print("images",images)
print("images shape",images.shape)
print("target",y)

X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(
    X, y, images, train_size=2 / 3, test_size=1 / 3
)

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

plt.savefig('hr_digits_dmap.png')

pcm = pfold.PCManifold(X, 
                        kernel=pfold.DmapKernelFixed(internal_kernel=pfold.GaussianKernel(epsilon=378.0533464967807), is_stochastic=True, alpha=1, symmetrize_kernel=True),
                        dist_kwargs=dict(cut_off=83.45058418010026, kmin=0, backend= "guess_optimal"))

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
plt.savefig('hr_digits_scipy.png')
plot_pairwise_eigenvector(
    eigenvectors=sorted_gofmm_evecs[:, 1:],
    n=0,
    idx_start=1,
    fig_params=dict(figsize=(10, 10)),
    scatter_params=dict(c=y),
)
plt.savefig('hr_digits_gofmm.png')
