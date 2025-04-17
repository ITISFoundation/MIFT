import numpy as np
from .utils import compute_svd_jax_numpy, regularize, check_regularization


def efficient_unstructured_FFT_1D(
    u1d_mask, z1d, kz, N, mask1d, SVD=True, max_reg_iterations=100
):
    """
    This function computes Fourier-based coefficients for 1D masked data. The inputs are as follows:
        - u1d_mask: the masked input data
        - z1d: the space coordinate for the data
        - kz: the reduced k_0 for the data which is q*2*np.pi/(np.max(z1d)-np.min(z1d)), q is used for padding
        - N: number of Fourier-basis
        - mask1d: the mask, one where the data is available and zero where the data deos not exist
        - SVD: determines whether or not the user wants/must use SVD for solving [AA][xx] = [BB]
        - max_reg_iterations: maximum number of iterations for regularization
    The outputs are [AA], [BB], and [xx], in which [xx] are the Fourier-based coefficients.
    """

    i = np.linspace(-2 * N, 2 * N, 4 * N + 1)  # n
    # precompute exponentials
    kzarray = np.exp(1j * 1 * kz * np.outer(i, z1d))

    # ompute Si
    mask1d_z = np.zeros((4 * N + 1), dtype=np.complex64)
    for i in range(0, 4 * N + 1):
        mask1d_z[i] = np.tensordot(mask1d, kzarray[i, :], axes=([0], [0]))

    # compute pre-requsite for [BB]
    u1dx_z = np.zeros((2 * N + 1), dtype=np.complex64)
    for i in range(0, 2 * N + 1):
        u1dx_z[i] = np.tensordot(u1d_mask, kzarray[i + 1 * N, :], axes=([0], [0]))

    # compute A([A]) and b ([B])
    A = np.zeros((2 * (2 * N + 1), 2 * (2 * N + 1)))
    b = np.zeros(2 * (2 * N + 1))

    # a helper variable for vectorization
    a = np.linspace(-1 * N, 1 * N, 2 * N + 1)

    # compute [B]
    kz_indt = a.astype(int)
    b[::2] = np.real(u1dx_z[kz_indt + 1 * N])
    b[1::2] = np.imag(u1dx_z[kz_indt + 1 * N])

    # compute [A]
    kz_ind = np.kron(a, np.ones(((2 * N + 1)))).astype(int)
    kpz_ind = np.kron(np.ones(((2 * N + 1),)), a).astype(int)
    Sp = mask1d_z[kz_ind + kpz_ind + 2 * N]
    Sm = mask1d_z[kz_ind - kpz_ind + 2 * N]
    A[::2, ::2] = np.real(Sp + Sm).reshape(((2 * N + 1), (2 * N + 1)))
    A[::2, 1::2] = -np.imag(Sp - Sm).reshape(((2 * N + 1), (2 * N + 1)))
    A[1::2, ::2] = np.imag(Sp + Sm).reshape(((2 * N + 1), (2 * N + 1)))
    A[1::2, 1::2] = np.real(Sp - Sm).reshape(((2 * N + 1), (2 * N + 1)))

    # compute [AA] and [BB]
    AA = A[: (2 * N + 1), : (2 * N + 1)]
    bb = b[: (2 * N + 1)] / 2

    if SVD:
        # compute SVD using jax numpy
        u, s, vh = compute_svd_jax_numpy(AA)

        # regularization if needed, the thresholds can be modified
        sval_nums = regularize(s)

        # check if regualizarion is fine, otherwise and if we have high error, we will increase number of singular values that we use
        coef = check_regularization(max_reg_iterations, AA, bb, u, s, vh, sval_nums)
    else:
        coef = np.linalg.solve(AA, bb)
        # Check if AA @ AA_inv ≈ Identity
        normalized_residual = np.linalg.norm(AA @ np.linalg.inv(AA) - np.eye(AA.shape[0]), ord='fro')/np.linalg.norm(np.eye(AA.shape[0]), ord='fro')
        print(f"normalized residual = {normalized_residual}")
        # Optional: Print or raise warning
        if normalized_residual > 1e-2:
            print("Warning: [AA] might be ill-conditioned. Consider enabling SVD regularization.")
        else:
            print(f"residual = {normalized_residual}, [AA] seems well-conditioned.")


    return (AA, bb, coef)


def efficient_unstructured_IFFT_point_1D(coef, z1d, kz, N):
    """
    This function reconstructs the data in space domain from the Fourier-based coefficients for 1D data.
    The inputs are as follows:
        - coef: Fourier-based coefficients
        - z1d: the space coordinate for the data you want ro reconstruct
        - kz: the reduced k_0 for the data which is q*2*np.pi/(np.max(z1d)-np.min(z1d)), q is used for padding
        - N: number of Fourier-basis
    The outputs are reconstructed data.
    """
    # compute full coef (i. e. [x]) from [xx]
    coefs_new = np.zeros(2 * (2 * N + 1))
    for i in np.arange(0, len(coef) - 1, 2):
        coefs_new[i : i + 2] = coef[i : i + 2]
        coefs_new[len(coefs_new) - 2 - i] = coef[i]
        coefs_new[len(coefs_new) - i - 1] = -coef[i + 1]
    coefs_new[len(coef) - 1] = 2 * coef[len(coef) - 1]
    coefs_complex = np.zeros(((2 * N + 1)), dtype=np.complex64)
    coefs_complex = coefs_new[::2] + 1j * coefs_new[1::2]

    i = np.linspace(-1 * N, 1 * N, 2 * N + 1)  # n
    # precompute exponentials
    kzarray = np.exp(1j * 1 * kz * np.outer(i, z1d))

    # recinstruct the data
    cnt = 0
    recon = 0
    for kz_ind in range(-N, N + 1):
        recon += kzarray[kz_ind + N, :] * coefs_complex[cnt]
        cnt += 1
    return 2 * np.real(recon)
