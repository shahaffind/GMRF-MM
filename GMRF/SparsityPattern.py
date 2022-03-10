import numpy as np


def get_patch_sparsity(patch_dims, blanket_dims):
    n_dims = len(patch_dims)
    total_dim = 1
    for i in range(n_dims):
        total_dim *= patch_dims[i]

    C = np.zeros([total_dim, total_dim])

    iterate_over_dim(patch_dims, blanket_dims, [], 0, 0, C)

    return C


def iterate_over_dim(patch_dims, blanket_dims, loc, pos, d, C):
    if d >= len(patch_dims):
        set_row(patch_dims, blanket_dims, loc, pos, C)
    else:
        pos *= patch_dims[d]
        for i in range(patch_dims[d]):
            iterate_over_dim(patch_dims, blanket_dims, loc + [i], pos + i, d + 1, C)


def set_row(patch_dims, blanket_dims, loc, pos, C):
    set_row_over_dim(patch_dims, blanket_dims, loc, pos, 0, 0, C)


def set_row_over_dim(patch_dims, blanket_dims, center_loc, center_pos, pos, d, C):
    if d >= len(patch_dims):
        C[center_pos, pos] = 1
    else:
        delta = int((blanket_dims[d]-1)/2)
        i_low, i_high = max(center_loc[d] - delta, 0), min(center_loc[d] + delta + 1, patch_dims[d])

        pos *= patch_dims[d]
        for i in range(i_low, i_high):
            set_row_over_dim(patch_dims, blanket_dims, center_loc, center_pos, pos + i, d + 1, C)
