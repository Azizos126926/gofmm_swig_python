import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg
from scipy.linalg import eig, eigh
from sklearn.datasets import make_s_curve
import sys
sys.path.insert(1, '../python')

# NOTE: make sure "path/to/datafold" is in sys.path or PYTHONPATH if not ,!installed
import datafold.dynfold as dfold
import datafold.pcfold as pfold
from datafold.dynfold import LocalRegressionSelection
from datafold.utils.plot import plot_pairwise_eigenvector
from datafold.pcfold.kernels import PCManifoldKernel

import full_matrix


#datafold stuff
random_state = 42
rng = np.random.default_rng( random_state )
number_of_samples = 2048
number_of_samples_to_plot = 1024
idx_plot = np.random.permutation(number_of_samples)[0:number_of_samples_to_plot]
# generate point cloud
X, X_color = make_s_curve(number_of_samples, random_state=42, noise=0)

X_pcm = pfold.PCManifold(X, 
                        kernel=pfold.DmapKernelFixed(internal_kernel=pfold.GaussianKernel(epsilon=0.7305584457602416), is_stochastic=True, alpha=1, symmetrize_kernel=True),
                        dist_kwargs=dict(cut_off=3.6684307127363676, kmin=0, backend= "guess_optimal"))

kernel_output = X_pcm.compute_kernel_matrix()
( kernel_matrix, cdist_kwargs, ret_extra, ) = PCManifoldKernel.read_kernel_output(kernel_output=kernel_output)

pcm = pfold.PCManifold(X)
pcm.optimize_parameters()

dmap = dfold.DiffusionMaps(
    kernel=pfold.GaussianKernel(pcm.kernel.epsilon),
    n_eigenpairs=9,
    dist_kwargs=dict(cut_off=pcm.cut_off),
)
dmap = dmap.fit(pcm, store_kernel_matrix=True)
evecs, evals = dmap.eigenvectors_, dmap.eigenvalues_


basis_change_matrix = ret_extra['basis_change_matrix']
inv_basis_change_matrix = scipy.sparse.diags(np.reciprocal(basis_change_matrix.data.ravel()))

executable = "./test_gofmm"
problem_size = 2048
max_leaf_node_size = 1024
num_of_neighbors = 0
max_off_diagonal_ranks = 1024
num_rhs = 1
user_tolerance = 1E-7
computation_budget = 0.00
distance_type = "kernel"
matrix_type = "dense"
kernel_type = "gaussian"

kernel_matrix_sparse = kernel_matrix.copy()
kernel_matrix_sparse = kernel_matrix_sparse.asfptype()
kernel_matrix = kernel_matrix.todense()
kernel_matrix = kernel_matrix.astype("float32")
#kernel_matrix.tofile("KernelMatrix_32768.bin")
weights = np.ones((problem_size, num_rhs))      


kernel_matrix_OP = full_matrix.FullMatrix( executable, problem_size, max_leaf_node_size,
                            num_of_neighbors, max_off_diagonal_ranks, num_rhs, user_tolerance, computation_budget,
                            distance_type, matrix_type, kernel_type, kernel_matrix, weights, dtype=np.float32 )

n_eigenpairs = 9
solver_kwargs = {
    "k": n_eigenpairs,
    "which": "LM",
    "v0": np.ones(problem_size),
    "tol": 1e-14,
    "sigma": 1.1, 
    "mode": "normal"
}

plot_pairwise_eigenvector(
    eigenvectors=dmap.eigenvectors_[idx_plot, :],
    n=1,
    fig_params=dict(figsize=[15, 15]),
    scatter_params=dict(cmap=plt.cm.Spectral, c=X_color[idx_plot]),
)

plt.savefig('dmap.png')

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

# plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    X[idx_plot, 0],
    X[idx_plot, 1],
    X[idx_plot, 2],
    c=X_color[idx_plot],
    cmap=plt.cm.Spectral,
)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("point cloud on S-shaped manifold")
ax.view_init(10, 70)
plt.savefig('scurve.png')

absolute_errors = sorted_scipy_evals - sorted_gofmm_evals
print( " Norm error: ", np.linalg.norm( absolute_errors )/n_eigenpairs)

plot_pairwise_eigenvector(
    eigenvectors=sorted_gofmm_evecs[idx_plot, :],
    n=1,
    fig_params=dict(figsize=[15, 15]),
    scatter_params=dict(cmap=plt.cm.Spectral, c=X_color[idx_plot]),
)
plt.savefig('gofmm.png')

plot_pairwise_eigenvector(
    eigenvectors=sorted_scipy_evecs[idx_plot, :],
    n=1,
    fig_params=dict(figsize=[15, 15]),
    scatter_params=dict(cmap=plt.cm.Spectral, c=X_color[idx_plot]),
)
plt.savefig('scipy.png')

selection = LocalRegressionSelection(
    intrinsic_dim=2, n_subsample=1024, strategy="dim"
).fit(sorted_scipy_evecs)
print(f"Found parsimonious eigenvectors (indices): {selection.evec_indices_}")

target_mapping = selection.transform(sorted_scipy_evecs)

f, ax = plt.subplots(figsize=(15, 9))
ax.scatter(
    target_mapping[idx_plot, 0],
    target_mapping[idx_plot, 1],
    c=X_color[idx_plot],
    cmap=plt.cm.Spectral,
)
plt.savefig('unfolded.png')
