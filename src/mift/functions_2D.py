import numpy as np
from .utils import compute_svd_jax_numpy, regularize, check_regularization


def efficient_unstructured_FFT_2D(
    u2d_mask, y1d, z1d, ky, kz, Ny, Nz, mask2d, SVD=True, max_reg_iterations=100
):
    """
    This function computes Fourier-based coefficients for 2D masked data. The inputs are as follows:
        - u2d_mask: the masked input data
        - y1d: the space coordinate for the data in y-axis (1D)
        - z1d: the space coordinate for the data in z-axis (1D)
        - ky: the reduced k_0 for the data which is q*2*np.pi/(np.max(y1d)-np.min(y1d)), q is used for padding
        - kz: similar to ky
        - Ny and Nz: number of Fourier-basis in y and z axis
        - mask2d: the mask, one where the data is available and zero where the data deos not exist
        - SVD: determines whether or not the user wants/must use SVD for solving [AA][xx] = [BB]
        - max_reg_iterations: maximum number of iterations for regularization
    The outputs are [AA], [BB], and [xx], in which [xx] are the Fourier-based coefficients.
    """
    iy = np.linspace(-2 * Ny, 2 * Ny, 4 * Ny + 1)  # ny
    iz = np.linspace(-2 * Nz, 2 * Nz, 4 * Nz + 1)  # nz
    # precompute exponentials
    kyarray = np.exp(1j * 1 * ky * np.outer(iy, y1d))
    kzarray = np.exp(1j * 1 * kz * np.outer(iz, z1d))

    # ompute Si
    mask2d_z = np.zeros((len(y1d), 4 * Nz + 1), dtype=np.complex64)
    for i in range(0, 4 * Nz + 1):
        mask2d_z[:, i] = np.tensordot(mask2d, kzarray[i, :], axes=([1], [0]))
    mask2d_zy = np.zeros((4 * Ny + 1, 4 * Nz + 1), dtype=np.complex64)
    for i in range(0, 4 * Ny + 1):
        mask2d_zy[i, :] = np.tensordot(mask2d_z, kyarray[i, :], axes=([0], [0]))
    # mask2d_zy is the sum over all points fot exp(jkyy+jkzz)

    # compute pre-requsite for [BB]
    u2dx_z = np.zeros((len(y1d), 2 * Nz + 1), dtype=np.complex64)
    for i in range(0, 2 * Nz + 1):
        u2dx_z[:, i] = np.tensordot(
            np.complex64(u2d_mask), kzarray[i + 1 * Nz, :], axes=([1], [0])
        )
    u2dx_zy = np.zeros((2 * Ny + 1, 2 * Nz + 1), dtype=np.complex64)
    for i in range(0, 2 * Ny + 1):
        u2dx_zy[i, :] = np.tensordot(u2dx_z, kyarray[i + 1 * Ny, :], axes=([0], [0]))
    # u2dx_zy is sum over all points for fx*exp(jkyy+jkzz)

    # compute A([A]) and b ([B])
    A = np.zeros((2 * (2 * Ny + 1) * (2 * Nz + 1), 2 * (2 * Ny + 1) * (2 * Nz + 1)))
    b = np.zeros(2 * (2 * Ny + 1) * (2 * Nz + 1))

    # a helper variable for vectorization
    ay = np.linspace(-1 * Ny, 1 * Ny, 2 * Ny + 1)
    az = np.linspace(-1 * Nz, 1 * Nz, 2 * Nz + 1)

    # compute [B]
    ky_indt = np.kron(ay, np.ones(az.shape)).astype(int)
    kz_indt = np.kron(np.ones(ay.shape), az).astype(int)
    b[::2] = np.real(u2dx_zy[ky_indt + 1 * Ny, kz_indt + 1 * Nz])
    b[1::2] = np.imag(u2dx_zy[ky_indt + 1 * Ny, kz_indt + 1 * Nz])

    # compute [A]
    ky_ind = np.kron(ay, np.ones((((2 * Nz + 1) ** 2) * (2 * Ny + 1)))).astype(int)
    kz_ind = np.kron(
        np.ones((2 * Ny + 1,)), np.kron(az, np.ones(((2 * Ny + 1) * (2 * Nz + 1))))
    ).astype(int)
    kpy_ind = np.kron(ay, np.ones(az.shape))
    kpy_ind = np.kron(np.ones(((2 * Nz + 1) * (2 * Ny + 1),)), kpy_ind).astype(int)
    kpz_ind = np.kron(np.ones(ay.shape), az)
    kpz_ind = np.kron(np.ones(((2 * Nz + 1) * (2 * Ny + 1),)), kpz_ind).astype(int)
    Sp = mask2d_zy[ky_ind + kpy_ind + 2 * Ny, kz_ind + kpz_ind + 2 * Nz]
    Sm = mask2d_zy[ky_ind - kpy_ind + 2 * Ny, kz_ind - kpz_ind + 2 * Nz]
    A[::2, ::2] = np.real(Sp + Sm).reshape(
        ((2 * Ny + 1) * (2 * Nz + 1), (2 * Ny + 1) * (2 * Nz + 1))
    )
    A[::2, 1::2] = -np.imag(Sp - Sm).reshape(
        ((2 * Ny + 1) * (2 * Nz + 1), (2 * Ny + 1) * (2 * Nz + 1))
    )
    A[1::2, ::2] = np.imag(Sp + Sm).reshape(
        ((2 * Ny + 1) * (2 * Nz + 1), (2 * Ny + 1) * (2 * Nz + 1))
    )
    A[1::2, 1::2] = np.real(Sp - Sm).reshape(
        ((2 * Ny + 1) * (2 * Nz + 1), (2 * Ny + 1) * (2 * Nz + 1))
    )

    # compute [AA] and [BB]
    AA = A[: (2 * Ny + 1) * (2 * Nz + 1), : (2 * Ny + 1) * (2 * Nz + 1)]
    bb = b[: (2 * Ny + 1) * (2 * Nz + 1)] / 2

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
        residual = np.linalg.norm(AA @ np.linalg.inv(AA) - np.eye(AA.shape[0]), ord='fro')

        # Optional: Print or raise warning
        if residual > 1e-1:
            print("Warning: [AA] might be ill-conditioned. Consider enabling SVD regularization.")


    return (AA, bb, coef)


# this is for recpnstructing whole 2D space
def efficient_unstructured_IFFT_2D(coef, y1d, z1d, ky, kz, Ny, Nz):
    """
    This function reconstructs the data in space domain from the Fourier-based coefficients for 2D data.
    The inputs are as follows:
        - coef: Fourier-based coefficients
        - y1d: the space coordinate for the data you want to reconstruct in y-axis (1D)
        - z1d: the space coordinate for the data you want to reconstruct in z-axis (1D)
        - ky: the reduced k_0 for the data which is q*2*np.pi/(np.max(y1d)-np.min(y1d)), q is used for padding
        - kz: similar to ky
        - Ny and Nz: number of Fourier-basis in y and z axis
    The outputs are reconstructed data.
    """
    # compute full coef (i. e. [x]) from [xx]
    coefs_new = np.zeros(2 * (2 * Ny + 1) * (2 * Nz + 1))
    for i in np.arange(0, len(coef) - 1, 2):
        coefs_new[i : i + 2] = coef[i : i + 2]
        coefs_new[len(coefs_new) - 2 - i] = coef[i]
        coefs_new[len(coefs_new) - i - 1] = -coef[i + 1]
    coefs_new[len(coef) - 1] = 2 * coef[len(coef) - 1]
    coefs_complex = np.zeros(((2 * Ny + 1) * (2 * Nz + 1)), dtype=np.complex64)
    coefs_complex = coefs_new[::2] + 1j * coefs_new[1::2]

    coefs2D = np.reshape(coefs_complex, (2 * Ny + 1, 2 * Nz + 1))

    # precompute exponentials
    iy = np.linspace(-1 * Ny, 1 * Ny, 2 * Ny + 1)
    iz = np.linspace(-1 * Nz, 1 * Nz, 2 * Nz + 1)
    kyarray = np.exp(1j * 1 * ky * np.outer(iy, y1d))
    kzarray = np.exp(1j * 1 * kz * np.outer(iz, z1d))

    # reconstruct the data
    recon_z = np.zeros((2 * Ny + 1, len(z1d)), dtype=np.complex64)
    for i in range(0, len(z1d)):
        recon_z[:, i] = np.tensordot(coefs2D, kzarray[:, i], axes=([1], [0]))
    recon_zy = np.zeros((len(y1d), len(z1d)), dtype=np.complex64)
    for i in range(0, len(y1d)):
        recon_zy[i, :] = np.tensordot(recon_z, kyarray[:, i], axes=([0], [0]))

    return 2 * np.real(recon_zy)
