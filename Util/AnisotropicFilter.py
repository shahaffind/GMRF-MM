import numpy as np


def create_D(Nx, Ny):
    diff = np.vstack([np.eye(Nx, Nx, k=0) - np.eye(Nx, Nx, k=-1), np.hstack([np.zeros(Nx - 1), -1])])
    D = np.vstack([np.kron(np.eye(Ny), diff), np.kron(diff, np.eye(Nx))])

    return D


def laplacian(Nx, Ny):
    D = create_D(Nx, Ny)
    return D.T.dot(D)
