import numpy as np


def rotate90(imgs):
    """
    Rotate the image (or images) 90 degrees
    """
    return np.rot90(imgs) if imgs.ndim == 3 else np.rot90(imgs, axes=(1, 2))


def rotate180(imgs):
    """
    Rotate the image (or images) 180 degrees
    """
    return np.rot90(imgs, k=2) if imgs.ndim == 3 else np.rot90(imgs, axes=(1, 2), k=2)


def __salt_n_pepper(imgs, amount=0.01, salt_vs_pepper=0.5, seperate_channels=False):
    amount *= imgs.size if seperate_channels else imgs[..., 0].size

    num_salt = np.ceil(amount * salt_vs_pepper)
    num_pepper = np.ceil(amount * (1.0 - salt_vs_pepper))

    dims = imgs.shape if seperate_channels else imgs.shape[:-1]
    salt = [np.random.randint(0, i, int(num_salt)) for i in dims]
    pepper = [np.random.randint(0, i, int(num_pepper)) for i in dims]

    imgs[tuple(s for s in salt)] = 1
    imgs[tuple(p for p in pepper)] = 0

    return imgs


def salt_n_pepper(amount=0.01, salt_vs_pepper=0.5, seperate_channels=False):
    """
    Add salt and pepper noise to an image or image series.

    Args:
        amount (float, optional): How much noise to add. Defaults to 0.01.
        salt_vs_pepper (float, optional): The ratio between salt and pepper noise. Defaults to 0.5.
        seperate_channels (bool, optional): Whether to apply noise to each channel seperately. Defaults to False.
    """
    return lambda imgs: __salt_n_pepper(imgs, amount, salt_vs_pepper, seperate_channels)