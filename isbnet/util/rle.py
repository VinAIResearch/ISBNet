# Modify from https://www.kaggle.com/paulorzp/run-length-encode-and-decode
import numpy as np
import torch


def rle_encode(mask):
    """Encode RLE (Run-length-encode) from 1D binary mask.

    Args:
        mask (np.ndarray): 1D binary mask
    Returns:
        rle (dict): encoded RLE
    """
    length = mask.shape[0]
    mask = np.concatenate([[0], mask, [0]])
    runs = np.where(mask[1:] != mask[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    counts = " ".join(str(x) for x in runs)
    rle = dict(length=length, counts=counts)
    return rle


def rle_encode_gpu(mask):
    """Encode RLE (Run-length-encode) from 1D binary mask.

    Args:
        mask (np.ndarray): 1D binary mask
    Returns:
        rle (dict): encoded RLE
    """
    length = mask.shape[0]
    # mask = np.concatenate([[0], mask, [0]])
    zeros_tensor = torch.tensor([0], dtype=torch.bool, device=mask.device)
    mask = torch.cat([zeros_tensor, mask, zeros_tensor])

    runs = torch.nonzero(mask[1:] != mask[:-1]).view(-1) + 1
    runs[1::2] -= runs[::2]
    # runs = np.where(mask[1:] != mask[:-1])[0] + 1
    # runs[1::2] -= runs[::2]
    # counts = " ".join(str(x) for x in runs)
    counts = runs.cpu().numpy()
    # breakpoint()
    rle = dict(length=length, counts=counts)
    return rle


def rle_encode_gpu_batch(masks):
    """Encode RLE (Run-length-encode) from 1D binary mask.

    Args:
        mask (np.ndarray): 1D binary mask
    Returns:
        rle (dict): encoded RLE
    """
    n_inst, length = masks.shape[:2]
    zeros_tensor = torch.zeros((n_inst, 1), dtype=torch.bool, device=masks.device)
    masks = torch.cat([zeros_tensor, masks, zeros_tensor], dim=1)

    rles = []
    for i in range(n_inst):
        mask = masks[i]
        runs = torch.nonzero(mask[1:] != mask[:-1]).view(-1) + 1

        runs[1::2] -= runs[::2]

        counts = runs.cpu().numpy()
        rle = dict(length=length, counts=counts)
        rles.append(rle)
    return rles


def rle_decode(rle):
    """Decode rle to get binary mask.

    Args:
        rle (dict): rle of encoded mask
    Returns:
        mask (np.ndarray): decoded mask
    """
    length = rle["length"]
    s = rle["counts"]

    starts, nums = [np.asarray(x, dtype=np.int32) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + nums
    mask = np.zeros(length, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask
