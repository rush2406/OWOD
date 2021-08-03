# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import heapq

from detectron2.layers import nonzero_tuple

__all__ = ["subsample_labels","subsample_negative_labels"]


def subsample_negative_labels(
    labels: torch.Tensor, num_samples: int):
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


    perm = torch.randperm(negative.numel(), device=negative.device)[:num_samples]

    neg_idx = negative[perm]

    return neg_idx

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
