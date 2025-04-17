import numpy as np
from .utils import (
    compute_svd_jax_numpy,
    regularize,
    check_regularization,
    check_regularization_3D,
)


def efficient_unstructured_FFT_3D(
    u3d_mask, x1d, y1d, z1d, kx, ky, kz, Nx, Ny, Nz, mask3d, SVD=True, iterations=100
):
    """
    This function computes Fourier-based coefficients for 3D masked data. The inputs are as follows:
        - u3d_mask: the masked input data
        - x1d: the space coordinate for the data in x-axis (1D)
        - y1d: the space coordinate for the data in y-axis (1D)
        - z1d: the space coordinate for the data in z-axis (1D)
        - kx: the reduced k_0 for the data which is q*2*np.pi/(np.max(x1d)-np.min(x1d)), q is used for padding
        - ky: similar to kx
        - kz: similar to kx
        - Nx, Ny and Nz: number of Fourier-basis in x, y and z axis
        - mask3d: the mask, one where the data is available and zero where the data deos not exist
        - SVD: determines whether or not the user wants/must use SVD for solving [AA][xx] = [BB]
        - iterations: maximum number of iterations for regularization
    The outputs are [AA], [BB], and [xx], in which [xx] are the Fourier-based coefficients.
    """
    ix = np.linspace(-2 * Nx, 2 * Nx, 4 * Nx + 1)
    iy = np.linspace(-2 * Ny, 2 * Ny, 4 * Ny + 1)
    iz = np.linspace(-2 * Nz, 2 * Nz, 4 * Nz + 1)
    # precompute exponentials
    kxarray = np.exp(1j * 1 * kx * np.outer(ix, x1d))
    kyarray = np.exp(1j * 1 * ky * np.outer(iy, y1d))
    kzarray = np.exp(1j * 1 * kz * np.outer(iz, z1d))

    # ompute Si
    mask3d_z = np.zeros((len(x1d), len(y1d), 4 * Nz + 1), dtype=np.complex64)
    for i in range(0, 4 * Nz + 1):
        mask3d_z[:, :, i] = np.tensordot(mask3d * 1.0, kzarray[i, :], axes=([2], [0]))
    mask3d_zy = np.zeros((len(x1d), 4 * Ny + 1, 4 * Nz + 1), dtype=np.complex64)
    for i in range(0, 4 * Ny + 1):
        mask3d_zy[:, i, :] = np.tensordot(mask3d_z, kyarray[i, :], axes=([1], [0]))
    mask3d_zyx = np.zeros((4 * Nx + 1, 4 * Ny + 1, 4 * Nz + 1), dtype=np.complex64)
    for i in range(0, 4 * Nx + 1):
        mask3d_zyx[i, :, :] = np.tensordot(mask3d_zy, kxarray[i, :], axes=([0], [0]))
    # mask3d_zyx is the sum over all points fot exp(jkxx+jkyy+jkzz)

    # compute pre-requsite for [BB]
    u3dx_z = np.zeros((len(x1d), len(y1d), 2 * Nz + 1), dtype=np.complex64)
    for i in range(0, 2 * Nz + 1):
        u3dx_z[:, :, i] = np.tensordot(
            u3d_mask, kzarray[i + 1 * Nz, :], axes=([2], [0])
        )
    u3dx_zy = np.zeros((len(x1d), 2 * Ny + 1, 2 * Nz + 1), dtype=np.complex64)
    for i in range(0, 2 * Ny + 1):
        u3dx_zy[:, i, :] = np.tensordot(u3dx_z, kyarray[i + 1 * Ny, :], axes=([1], [0]))
    u3dx_zyx = np.zeros((2 * Nx + 1, 2 * Ny + 1, 2 * Nz + 1), dtype=np.complex64)
    for i in range(0, 2 * Nx + 1):
        u3dx_zyx[i, :, :] = np.tensordot(
            u3dx_zy, kxarray[i + 1 * Nx, :], axes=([0], [0])
        )
    # u2dx_zyx is sum over all points for fx*exp(jkxx+jkyy+jkzz)

    # compute A([A]) and b ([B])
    A = np.zeros(
        (
            2 * (2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1),
            2 * (2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1),
        )
    )
    b = np.zeros(2 * (2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1))

    # a helper variable for vectorization
    ax = np.linspace(-1 * Nx, 1 * Nx, 2 * Nx + 1)
    ay = np.linspace(-1 * Ny, 1 * Ny, 2 * Ny + 1)
    az = np.linspace(-1 * Nz, 1 * Nz, 2 * Nz + 1)

    # compute [B]
    kx_indt = np.kron(ax, np.ones(((2 * Ny + 1) * (2 * Nz + 1),))).astype(int)
    ky_indt = np.kron(ay, np.ones(az.shape))
    ky_indt = np.kron(np.ones(((2 * Nx + 1),)), ky_indt).astype(int)
    kz_indt = np.kron(np.ones(((2 * Nx + 1) * (2 * Ny + 1),)), az).astype(int)
    b[::2] = np.real(u3dx_zyx[kx_indt + 1 * Nx, ky_indt + 1 * Ny, kz_indt + 1 * Nz])
    b[1::2] = np.imag(u3dx_zyx[kx_indt + 1 * Nx, ky_indt + 1 * Ny, kz_indt + 1 * Nz])

    # compute [A]
    kx_ind = np.kron(
        ax,
        np.ones(
            ((2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1) * (2 * Ny + 1) * (2 * Nz + 1))
        ),
    ).astype(int)
    ky_ind = np.kron(
        np.ones((2 * Nx + 1,)),
        np.kron(
            ay, np.ones(((2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1) * (2 * Nz + 1)))
        ),
    ).astype(int)
    kz_ind = np.kron(
        np.ones(((2 * Nx + 1) * (2 * Ny + 1),)),
        np.kron(az, np.ones(((2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1)))),
    ).astype(int)
    kpx_ind = np.kron(ax, np.ones(((2 * Ny + 1) * (2 * Nz + 1),)))
    kpx_ind = np.kron(
        np.ones(((2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1),)), kpx_ind
    ).astype(int)
    kpy_ind = np.kron(ay, np.ones(az.shape))
    kpy_ind = np.kron(np.ones(((2 * Nx + 1),)), kpy_ind)
    kpy_ind = np.kron(
        np.ones(((2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1),)), kpy_ind
    ).astype(int)
    kpz_ind = np.kron(np.ones(((2 * Nx + 1) * (2 * Ny + 1),)), az)
    kpz_ind = np.kron(
        np.ones(((2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1),)), kpz_ind
    ).astype(int)
    Sp = mask3d_zyx[
        kx_ind + kpx_ind + 2 * Nx, ky_ind + kpy_ind + 2 * Ny, kz_ind + kpz_ind + 2 * Nz
    ]
    Sm = mask3d_zyx[
        kx_ind - kpx_ind + 2 * Nx, ky_ind - kpy_ind + 2 * Ny, kz_ind - kpz_ind + 2 * Nz
    ]
    A[::2, ::2] = np.real(Sp + Sm).reshape(
        (
            (2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1),
            (2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1),
        )
    )
    A[::2, 1::2] = -np.imag(Sp - Sm).reshape(
        (
            (2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1),
            (2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1),
        )
    )
    A[1::2, ::2] = np.imag(Sp + Sm).reshape(
        (
            (2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1),
            (2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1),
        )
    )
    A[1::2, 1::2] = np.real(Sp - Sm).reshape(
        (
            (2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1),
            (2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1),
        )
    )

    # compute [AA] and [BB]
    AA = A[
        : (2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1),
        : (2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1),
    ]
    bb = b[: (2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1)] / 2

    if SVD:
        # compute SVD using jax numpy
        u, s, vh = compute_svd_jax_numpy(AA)

        # regularization if needed, the thresholds can be modified
        sval_nums = regularize(s)

        # check if regualizarion is fine, otherwise and if we have high error, we will increase number of singular values that we use
        coef = check_regularization_3D(iterations, AA, bb, u, s, vh, sval_nums)

    else:
        coef = np.linalg.solve(AA, bb)
        # Check if AA @ AA_inv ≈ Identity
        normalized_residual = np.linalg.norm(AA @ np.linalg.inv(AA) - np.eye(AA.shape[0]), ord='fro')/np.linalg.norm(np.eye(AA.shape[0]), ord='fro')

        # Optional: Print or raise warning
        if normalized_residual > 1e-2:
            print("Warning: [AA] might be ill-conditioned. Consider enabling SVD regularization.")
        else:
            print(f"normalized residual = {normalized_residual}, [AA] seems well-conditioned.")

    return (AA, bb, coef)


# this is for recpnstructing whole 3D space
def efficient_unstructured_IFFT_3D(coef, x1d, y1d, z1d, kx, ky, kz, Nx, Ny, Nz):
    """
    This function reconstructs the data in space domain from the Fourier-based coefficients for 3D data.
    The inputs are as follows:
        - coef: Fourier-based coefficients
        - x1d: the space coordinate for the data you want to reconstruct in x-axis (1D)
        - y1d: the space coordinate for the data you want to reconstruct in y-axis (1D)
        - z1d: the space coordinate for the data you want to reconstruct in z-axis (1D)
        - kx: the reduced k_0 for the data which is q*2*np.pi/(np.max(x1d)-np.min(x1d)), q is used for padding
        - ky: similar to kx
        - kz: similar to kx
        - Nx, Ny and Nz: number of Fourier-basis in x, y and z axis
    The outputs are reconstructed data.
    """
    # compute full coef (i. e. [x]) from [xx]
    coefs_new = np.zeros(2 * (2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1))
    for i in np.arange(0, len(coef) - 1, 2):
        coefs_new[i : i + 2] = coef[i : i + 2]
        coefs_new[len(coefs_new) - 2 - i] = coef[i]
        coefs_new[len(coefs_new) - i - 1] = -coef[i + 1]
    coefs_new[len(coef) - 1] = 2 * coef[len(coef) - 1]
    coefs_complex = np.zeros(
        ((2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1)), dtype=np.complex64
    )
    coefs_complex = coefs_new[::2] + 1j * coefs_new[1::2]

    coefs3D = np.reshape(coefs_complex, (2 * Nx + 1, 2 * Ny + 1, 2 * Nz + 1))

    # precompute exponentials
    ix = np.linspace(-1 * Nx, 1 * Nx, 2 * Nx + 1)
    iy = np.linspace(-1 * Ny, 1 * Ny, 2 * Ny + 1)
    iz = np.linspace(-1 * Nz, 1 * Nz, 2 * Nz + 1)
    kxarray = np.exp(1j * 1 * kx * np.outer(ix, x1d))
    kyarray = np.exp(1j * 1 * ky * np.outer(iy, y1d))
    kzarray = np.exp(1j * 1 * kz * np.outer(iz, z1d))

    # reconstruct the data
    recon_z = np.zeros((2 * Nx + 1, 2 * Ny + 1, len(z1d)), dtype=np.complex64)
    for i in range(0, len(z1d)):
        recon_z[:, :, i] = np.tensordot(coefs3D, kzarray[:, i], axes=([2], [0]))
    recon_zy = np.zeros((2 * Nx + 1, len(y1d), len(z1d)), dtype=np.complex64)
    for i in range(0, len(y1d)):
        recon_zy[:, i, :] = np.tensordot(recon_z, kyarray[:, i], axes=([1], [0]))
    recon_zyx = np.zeros((len(x1d), len(y1d), len(z1d)), dtype=np.complex64)
    for i in range(0, len(x1d)):
        recon_zyx[i, :, :] = np.tensordot(recon_zy, kxarray[:, i], axes=([0], [0]))
    return 2 * np.real(recon_zyx)


# this is for reconstructing specific points
def efficient_unstructured_IFFT_point(coef, x1d, y1d, z1d, kx, ky, kz, Nx, Ny, Nz):
    """
    This function reconstructs the data in space domain from the Fourier-based coefficients for 3D data in specific points.
    The difference between this function and "efficient_unstructured_IFFT_3D" is that, this function reconctructs the data on specific points,
    but "efficient_unstructured_IFFT_3D", reconctructs the data on whole 3D domain.
    The inputs are as follows:
        - coef: Fourier-based coefficients
        - x1d: the x coordinate for the data you want to reconstruct
        - y1d: the y coordinate for the data you want to reconstruct
        - z1d: the z coordinate for the data you want to reconstruct
        - kx: the reduced k_0 for the data which is q*2*np.pi/(np.max(x1d)-np.min(x1d)), q is used for padding
        - ky: similar to kx
        - kz: similar to kx
        - Nx, Ny and Nz: number of Fourier-basis in x, y and z axis
    The outputs are reconstructed data.
    """
    # compute full coef (i. e. [x]) from [xx]
    coefs_new = np.zeros(2 * (2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1))
    for i in np.arange(0, len(coef) - 1, 2):
        coefs_new[i : i + 2] = coef[i : i + 2]
        coefs_new[len(coefs_new) - 2 - i] = coef[i]
        coefs_new[len(coefs_new) - i - 1] = -coef[i + 1]
    coefs_new[len(coef) - 1] = 2 * coef[len(coef) - 1]
    coefs_complex = np.zeros(
        ((2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1)), dtype=np.complex64
    )
    coefs_complex = coefs_new[::2] + 1j * coefs_new[1::2]

    coefs3D = np.reshape(coefs_complex, (2 * Nx + 1, 2 * Ny + 1, 2 * Nz + 1))

    # precompute exponentials
    ix = np.linspace(-1 * Nx, 1 * Nx, 2 * Nx + 1)
    iy = np.linspace(-1 * Ny, 1 * Ny, 2 * Ny + 1)
    iz = np.linspace(-1 * Nz, 1 * Nz, 2 * Nz + 1)
    kxarray = np.exp(1j * 1 * kx * np.outer(ix, x1d))
    kyarray = np.exp(1j * 1 * ky * np.outer(iy, y1d))
    kzarray = np.exp(1j * 1 * kz * np.outer(iz, z1d))

    # reconstruct the data
    recon = 0
    cnt = 0
    for kx_ind in range(-Nx, Nx + 1):
        for ky_ind in range(-Ny, Ny + 1):
            for kz_ind in range(-Nz, Nz + 1):
                recon += (
                    kxarray[kx_ind + Nx, :]
                    * kyarray[ky_ind + Ny, :]
                    * kzarray[kz_ind + Nz, :]
                    * coefs_complex[cnt]
                )
                cnt += 1
    return 2 * np.real(recon)


def efficient_unstructured_FFT_3D_oneTime(
    x1d, y1d, z1d, kx, ky, kz, Nx, Ny, Nz, mask3d, SVD=True
):
    """
    This function computes [AA]  matrix and its SVD for 3D masked data. This functions alongside with "efficient_unstructured_FFT_3D_eachTime" should be used
    instead of "efficient_unstructured_FFT_3D" when there are several time-step in the dataset and they all have the same mask.
    In this case, the [AA] is the same for all time-steps which will be done using this function only once.
    The inputs are as follows:
        - u3d_mask: the masked input data
        - x1d: the space coordinate for the data in x-axis (1D)
        - y1d: the space coordinate for the data in y-axis (1D)
        - z1d: the space coordinate for the data in z-axis (1D)
        - kx: the reduced k_0 for the data which is q*2*np.pi/(np.max(x1d)-np.min(x1d)), q is used for padding
        - ky: similar to kx
        - kz: similar to kx
        - Nx, Ny and Nz: number of Fourier-basis in x, y and z axis
        - mask3d: the mask, one where the data is available and zero where the data deos not exist
        - SVD: determines whether or not the user wants/must use SVD for solving [AA][xx] = [BB]
    The outputs are [AA], its inversion and its SVD. .
    """
    ix = np.linspace(-2 * Nx, 2 * Nx, 4 * Nx + 1)
    iy = np.linspace(-2 * Ny, 2 * Ny, 4 * Ny + 1)
    iz = np.linspace(-2 * Nz, 2 * Nz, 4 * Nz + 1)
    # precompute exponentials
    kxarray = np.exp(1j * 1 * kx * np.outer(ix, x1d))
    kyarray = np.exp(1j * 1 * ky * np.outer(iy, y1d))
    kzarray = np.exp(1j * 1 * kz * np.outer(iz, z1d))

    # ompute Si
    mask3d_z = np.zeros((len(x1d), len(y1d), 4 * Nz + 1), dtype=np.complex64)
    for i in range(0, 4 * Nz + 1):
        mask3d_z[:, :, i] = np.tensordot(mask3d * 1.0, kzarray[i, :], axes=([2], [0]))
    mask3d_zy = np.zeros((len(x1d), 4 * Ny + 1, 4 * Nz + 1), dtype=np.complex64)
    for i in range(0, 4 * Ny + 1):
        mask3d_zy[:, i, :] = np.tensordot(mask3d_z, kyarray[i, :], axes=([1], [0]))
    mask3d_zyx = np.zeros((4 * Nx + 1, 4 * Ny + 1, 4 * Nz + 1), dtype=np.complex64)
    for i in range(0, 4 * Nx + 1):
        mask3d_zyx[i, :, :] = np.tensordot(mask3d_zy, kxarray[i, :], axes=([0], [0]))
    # mask3d_zyx is the sum over all points fot exp(jkxx+jkyy+jkzz)

    # compute A([A])
    A = np.zeros(
        (
            2 * (2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1),
            2 * (2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1),
        )
    )

    # a helper variable for vectorization
    ax = np.linspace(-1 * Nx, 1 * Nx, 2 * Nx + 1)
    ay = np.linspace(-1 * Ny, 1 * Ny, 2 * Ny + 1)
    az = np.linspace(-1 * Nz, 1 * Nz, 2 * Nz + 1)

    # compute [A]
    kx_ind = np.kron(
        ax,
        np.ones(
            ((2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1) * (2 * Ny + 1) * (2 * Nz + 1))
        ),
    ).astype(int)
    ky_ind = np.kron(
        np.ones((2 * Nx + 1,)),
        np.kron(
            ay, np.ones(((2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1) * (2 * Nz + 1)))
        ),
    ).astype(int)
    kz_ind = np.kron(
        np.ones(((2 * Nx + 1) * (2 * Ny + 1),)),
        np.kron(az, np.ones(((2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1)))),
    ).astype(int)
    kpx_ind = np.kron(ax, np.ones(((2 * Ny + 1) * (2 * Nz + 1),)))
    kpx_ind = np.kron(
        np.ones(((2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1),)), kpx_ind
    ).astype(int)
    kpy_ind = np.kron(ay, np.ones(az.shape))
    kpy_ind = np.kron(np.ones(((2 * Nx + 1),)), kpy_ind)
    kpy_ind = np.kron(
        np.ones(((2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1),)), kpy_ind
    ).astype(int)
    kpz_ind = np.kron(np.ones(((2 * Nx + 1) * (2 * Ny + 1),)), az)
    kpz_ind = np.kron(
        np.ones(((2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1),)), kpz_ind
    ).astype(int)
    Sp = mask3d_zyx[
        kx_ind + kpx_ind + 2 * Nx, ky_ind + kpy_ind + 2 * Ny, kz_ind + kpz_ind + 2 * Nz
    ]
    Sm = mask3d_zyx[
        kx_ind - kpx_ind + 2 * Nx, ky_ind - kpy_ind + 2 * Ny, kz_ind - kpz_ind + 2 * Nz
    ]
    A[::2, ::2] = np.real(Sp + Sm).reshape(
        (
            (2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1),
            (2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1),
        )
    )
    A[::2, 1::2] = -np.imag(Sp - Sm).reshape(
        (
            (2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1),
            (2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1),
        )
    )
    A[1::2, ::2] = np.imag(Sp + Sm).reshape(
        (
            (2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1),
            (2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1),
        )
    )
    A[1::2, 1::2] = np.real(Sp - Sm).reshape(
        (
            (2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1),
            (2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1),
        )
    )

    # compute [AA]
    AA = A[
        : (2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1),
        : (2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1),
    ]

    if SVD:
        # compute SVD using jax numpy
        u, s, vh = compute_svd_jax_numpy(AA)

        # regularization if needed, the thresholds can be modified
        sval_nums = regularize(s)

        invAA = np.matmul(
            vh[0:sval_nums, :].transpose(),
            np.matmul(np.diag(1 / s[0:sval_nums]), u[:, 0:sval_nums].transpose()),
        )
    else:
        invAA = np.linalg.inv(AA)
        (u, s, vh) = (np.nan, np.nan, np.nan)

    return (AA, invAA, u, s, vh)


def efficient_unstructured_FFT_3D_eachTime(
    u3d_mask,
    x1d,
    y1d,
    z1d,
    kx,
    ky,
    kz,
    Nx,
    Ny,
    Nz,
    AA,
    invAA,
    u,
    s,
    vh,
    SVD=True,
    max_reg_iterations=100,
):
    """
    This function computes [bb] and [xx] matrix for 3D masked data. This functions alongside with "efficient_unstructured_FFT_3D_OneTime" should be used
    instead of "efficient_unstructured_FFT_3D" when there are several time-step in the dataset and they all have the same mask.
    In this case, the [AA] is the same for all time-steps which will be done using "efficient_unstructured_FFT_3D_oneTime" function only once.
    then [BB] and [xx] needs to be computed for each time step using this fucntion.
    The inputs are as follows:
        - u3d_mask: the masked input data
        - x1d: the space coordinate for the data in x-axis (1D)
        - y1d: the space coordinate for the data in y-axis (1D)
        - z1d: the space coordinate for the data in z-axis (1D)
        - kx: the reduced k_0 for the data which is q*2*np.pi/(np.max(x1d)-np.min(x1d)), q is used for padding
        - ky: similar to kx
        - kz: similar to kx
        - Nx, Ny and Nz: number of Fourier-basis in x, y and z axis
        - [AA]: AA matrix which is computed using "efficient_unstructured_FFT_3D_oneTime" function
        - u, s, vh: SVD of [AA] which is computed using "efficient_unstructured_FFT_3D_oneTime" function
        - SVD: determines whether or not the user wants/must use SVD for solving [AA][xx] = [BB]
        - max_reg_iterations: maximum number of iterations for regularization
    The outputs are [BB] and [xx].
    """
    ix = np.linspace(-2 * Nx, 2 * Nx, 4 * Nx + 1)
    iy = np.linspace(-2 * Ny, 2 * Ny, 4 * Ny + 1)
    iz = np.linspace(-2 * Nz, 2 * Nz, 4 * Nz + 1)
    # precompute exponentials
    kxarray = np.exp(1j * 1 * kx * np.outer(ix, x1d))
    kyarray = np.exp(1j * 1 * ky * np.outer(iy, y1d))
    kzarray = np.exp(1j * 1 * kz * np.outer(iz, z1d))

    # compute pre-requsite for [BB]
    u3dx_z = np.zeros((len(x1d), len(y1d), 2 * Nz + 1), dtype=np.complex64)
    for i in range(0, 2 * Nz + 1):
        u3dx_z[:, :, i] = np.tensordot(
            u3d_mask, kzarray[i + 1 * Nz, :], axes=([2], [0])
        )
    u3dx_zy = np.zeros((len(x1d), 2 * Ny + 1, 2 * Nz + 1), dtype=np.complex64)
    for i in range(0, 2 * Ny + 1):
        u3dx_zy[:, i, :] = np.tensordot(u3dx_z, kyarray[i + 1 * Ny, :], axes=([1], [0]))
    u3dx_zyx = np.zeros((2 * Nx + 1, 2 * Ny + 1, 2 * Nz + 1), dtype=np.complex64)
    for i in range(0, 2 * Nx + 1):
        u3dx_zyx[i, :, :] = np.tensordot(
            u3dx_zy, kxarray[i + 1 * Nx, :], axes=([0], [0])
        )
    # u2dx_zyx is sum over all points for fx*exp(jkxx+jkyy+jkzz)

    # compute A([A]) and b ([B])
    b = np.zeros(2 * (2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1))

    # a helper variable for vectorization
    ax = np.linspace(-1 * Nx, 1 * Nx, 2 * Nx + 1)
    ay = np.linspace(-1 * Ny, 1 * Ny, 2 * Ny + 1)
    az = np.linspace(-1 * Nz, 1 * Nz, 2 * Nz + 1)

    # compute [B]
    kx_indt = np.kron(ax, np.ones(((2 * Ny + 1) * (2 * Nz + 1),))).astype(int)
    ky_indt = np.kron(ay, np.ones(az.shape))
    ky_indt = np.kron(np.ones(((2 * Nx + 1),)), ky_indt).astype(int)
    kz_indt = np.kron(np.ones(((2 * Nx + 1) * (2 * Ny + 1),)), az).astype(int)
    b[::2] = np.real(u3dx_zyx[kx_indt + 1 * Nx, ky_indt + 1 * Ny, kz_indt + 1 * Nz])
    b[1::2] = np.imag(u3dx_zyx[kx_indt + 1 * Nx, ky_indt + 1 * Ny, kz_indt + 1 * Nz])

    # compute [AA] and [BB]
    bb = b[: (2 * Nx + 1) * (2 * Ny + 1) * (2 * Nz + 1)] / 2

    if SVD:
        # regularization if needed, the thresholds can be modified
        sval_nums = regularize(s)

        # check if regualizarion is fine, otherwise and if we have high error, we will increase number of singular values that we use
        coef = check_regularization(max_reg_iterations, AA, bb, u, s, vh, sval_nums)

    else:
        coef = np.matmul(invAA, bb)

    return (bb, coef)
