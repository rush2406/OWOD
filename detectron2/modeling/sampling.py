# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import numpy as np
import heapq

from detectron2.layers import nonzero_tuple

__all__ = ["subsample_labels","subsample_labels_iou","random_choice", "sample_via_interval"]

def random_choice(gallery, num):
    """Random select some elements from the gallery.
    If `gallery` is a Tensor, the returned indices will be a Tensor;
    If `gallery` is a ndarray or list, the returned indices will be a
    ndarray.
    Args:
        gallery (Tensor | ndarray | list): indices pool.
        num (int): expected sample num.
    Returns:
        Tensor or ndarray: sampled indices.
    """
    assert len(gallery) >= num

    is_tensor = isinstance(gallery, torch.Tensor)
    if not is_tensor:
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
        else:
            device = 'cpu'
        gallery = torch.tensor(gallery, dtype=torch.long, device=device)
    # This is a temporary fix. We can revert the following code
    # when PyTorch fixes the abnormal return of torch.randperm.
    # See: https://github.com/open-mmlab/mmdetection/pull/5014
    perm = torch.randperm(gallery.numel())[:num].to(device=gallery.device)
    rand_inds = gallery[perm]
    if not is_tensor:
        rand_inds = rand_inds.cpu().numpy()
    return rand_inds

def sample_via_interval(max_overlaps, full_set, num_expected,floor_thr,num_bins):

    """Sample according to the iou interval.
    Args:
        max_overlaps (torch.Tensor): IoU between bounding boxes and ground
            truth boxes.
        full_set (set(int)): A full set of indices of boxes。
        num_expected (int): Number of expected samples。
    Returns:
        np.ndarray: Indices  of samples
    """
    max_iou = max_overlaps.max()
    iou_interval = (max_iou - floor_thr) / num_bins
    per_num_expected = int(num_expected / num_bins)

    sampled_inds = []
    for i in range(num_bins):
        start_iou = floor_thr + i * iou_interval
        end_iou = floor_thr + (i + 1) * iou_interval
        tmp_set = set(
            np.where(
                np.logical_and(max_overlaps >= start_iou,
                               max_overlaps < end_iou))[0])
        tmp_inds = list(tmp_set & full_set)
        if len(tmp_inds) > per_num_expected:
            tmp_sampled_set = random_choice(tmp_inds,
                                                 per_num_expected)
        else:
            tmp_sampled_set = np.array(tmp_inds, dtype=np.int)
        sampled_inds.append(tmp_sampled_set)

    sampled_inds = np.concatenate(sampled_inds)
    if len(sampled_inds) < num_expected:
        num_extra = num_expected - len(sampled_inds)
        extra_inds = np.array(list(full_set - set(sampled_inds)))
        if len(extra_inds) > num_extra:
            extra_inds = random_choice(extra_inds, num_extra)
        sampled_inds = np.concatenate([sampled_inds, extra_inds])

    return sampled_inds

def subsample_labels(
    labels: torch.Tensor, num_samples: int, positive_fraction: float, bg_label: int, objectness_logits: torch.Tensor = None, ohem=None
):
    """
    Return `num_samples` (or fewer, if not enough found)
    random samples from `labels` which is a mixture of positives & negatives.
    It will try to return as many positives as possible without
    exceeding `positive_fraction * num_samples`, and then try to
    fill the remaining slots with negatives.

    Args:
        labels (Tensor): (N, ) label vector with values:
            * -1: ignore
            * bg_label: background ("negative") class
            * otherwise: one or more foreground ("positive") classes
        num_samples (int): The total number of labels with value >= 0 to return.
            Values that are not sampled will be filled with -1 (ignore).
        positive_fraction (float): The number of subsampled labels with values > 0
            is `min(num_positives, int(positive_fraction * num_samples))`. The number
            of negatives sampled is `min(num_negatives, num_samples - num_positives_sampled)`.
            In order words, if there are not enough positives, the sample is filled with
            negatives. If there are also not enough negatives, then as many elements are
            sampled as is possible.
        bg_label (int): label index of background ("negative") class.

    Returns:
        pos_idx, neg_idx (Tensor):
            1D vector of indices. The total length of both is `num_samples` or fewer.
    """
    positive = nonzero_tuple((labels != -1) & (labels != bg_label))[0]
    negative = nonzero_tuple(labels == bg_label)[0]

    num_pos = int(num_samples * positive_fraction)
    # protect against not enough positive examples
    num_pos = min(positive.numel(), num_pos)
    num_neg = num_samples - num_pos
    # protect against not enough negative examples
    num_neg = min(negative.numel(), num_neg)

    # randomly select positive and negative examples
    perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]

    #getting the instances with high objectness scores from backgrounds
    #perm2 = list(zip(*heapq.nlargest(num_neg, enumerate(objectness_logits), key=operator.itemgetter(1))))[0]
    perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

    #pos_idx = positive[perm1]
    
    if(ohem is None):
        neg_idx = negative[perm2]
        pos_idx = positive[perm1]
    else:
        neg_idx = negative[torch.randperm(negative.numel(), device=negative.device)]
        pos_idx = positive[torch.randperm(positive.numel(), device=positive.device)]

    return pos_idx, neg_idx,num_neg


def subsample_labels_iou(
    labels: torch.Tensor, num_samples: int, positive_fraction: float, bg_label: int, floor_thr, num_bins, floor_fraction,match_quality_matrix, objectness_logits: torch.Tensor = None
):
    """
    Return `num_samples` (or fewer, if not enough found)
    random samples from `labels` which is a mixture of positives & negatives.
    It will try to return as many positives as possible without
    exceeding `positive_fraction * num_samples`, and then try to
    fill the remaining slots with negatives.

    Args:
        labels (Tensor): (N, ) label vector with values:
            * -1: ignore
            * bg_label: background ("negative") class
            * otherwise: one or more foreground ("positive") classes
        num_samples (int): The total number of labels with value >= 0 to return.
            Values that are not sampled will be filled with -1 (ignore).
        positive_fraction (float): The number of subsampled labels with values > 0
            is `min(num_positives, int(positive_fraction * num_samples))`. The number
            of negatives sampled is `min(num_negatives, num_samples - num_positives_sampled)`.
            In order words, if there are not enough positives, the sample is filled with
            negatives. If there are also not enough negatives, then as many elements are
            sampled as is possible.
        bg_label (int): label index of background ("negative") class.

    Returns:
        pos_idx, neg_idx (Tensor):
            1D vector of indices. The total length of both is `num_samples` or fewer.
    """

    positive = nonzero_tuple((labels != -1) & (labels != bg_label))[0]
    negative = nonzero_tuple(labels == bg_label)[0]

    num_pos = int(num_samples * positive_fraction)
    # protect against not enough positive examples
    num_pos = min(positive.numel(), num_pos)
    num_neg = num_samples - num_pos
    # protect against not enough negative examples
    num_neg = min(negative.numel(), num_neg)

    # randomly select positive and negative examples
    perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]

    #getting the instances with high objectness scores from backgrounds
    #perm2 = list(zip(*heapq.nlargest(num_neg, enumerate(objectness_logits), key=operator.itemgetter(1))))[0]
    perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

    pos_idx = positive[perm1]

    max_overlaps = match_quality_matrix.cpu().numpy()
    # balance sampling for negative samples
    neg_set = set(negative.cpu().numpy())

    if floor_thr > 0:
        floor_set = set(
            np.where(
                np.logical_and(max_overlaps >= 0,
                               max_overlaps < floor_thr))[0])
        iou_sampling_set = set(
            np.where(max_overlaps >= floor_thr)[0])
    elif floor_thr == 0:
        floor_set = set(np.where(max_overlaps == 0)[0])
        iou_sampling_set = set(
            np.where(max_overlaps > floor_thr)[0])
    else:
        floor_set = set()
        iou_sampling_set = set(
            np.where(max_overlaps > floor_thr)[0])
        # for sampling interval calculation
        floor_thr = 0


    floor_neg_inds = list(floor_set & neg_set)
    iou_sampling_neg_inds = list(iou_sampling_set & neg_set)
    num_expected = num_neg
    num_expected_iou_sampling = int(num_expected *
                                    (1 - floor_fraction))

    if len(iou_sampling_neg_inds) > num_expected_iou_sampling:
        if num_bins >= 2:
            iou_sampled_inds = sample_via_interval(
                max_overlaps, set(iou_sampling_neg_inds),
                num_expected_iou_sampling)
        else:
            iou_sampled_inds = random_choice(
                iou_sampling_neg_inds, num_expected_iou_sampling)
    else:
        iou_sampled_inds = np.array(
            iou_sampling_neg_inds, dtype=np.int)
    num_expected_floor = num_expected - len(iou_sampled_inds)
    if len(floor_neg_inds) > num_expected_floor:
        sampled_floor_inds = random_choice(
            floor_neg_inds, num_expected_floor)
    else:
        sampled_floor_inds = np.array(floor_neg_inds, dtype=np.int)
    sampled_inds = np.concatenate(
        (sampled_floor_inds, iou_sampled_inds))
    if len(sampled_inds) < num_expected:
        num_extra = num_expected - len(sampled_inds)
        extra_inds = np.array(list(neg_set - set(sampled_inds)))
        if len(extra_inds) > num_extra:
            extra_inds = random_choice(extra_inds, num_extra)
        sampled_inds = np.concatenate((sampled_inds, extra_inds))
    sampled_inds = torch.from_numpy(sampled_inds).long().to(labels.device)


    return pos_idx, sampled_inds, num_neg