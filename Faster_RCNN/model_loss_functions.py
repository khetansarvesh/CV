import torch as t
from torch.nn import functional as F

def rpn_class_loss_f(predicted_scores, y_true):
  """
  Computes RPN class loss.

  Parameters
  ----------
  predicted_scores : torch.Tensor
    A tensor of shape (batch_size, height, width, num_anchors) containing
    objectness scores (0 = background, 1 = object).
  y_true : torch.Tensor
    Ground truth tensor of shape (batch_size, height, width, num_anchors, 6).

  Returns
  -------
  torch.Tensor
    Scalar loss.
  """

  epsilon = 1e-7

  # y_true_class: (batch_size, height, width, num_anchors), same as predicted_scores
  y_true_class = y_true[:,:,:,:,1].reshape(predicted_scores.shape)
  y_predicted_class = predicted_scores

  # y_mask: y_true[:,:,:,0] is 1.0 for anchors included in the mini-batch
  y_mask = y_true[:,:,:,:,0].reshape(predicted_scores.shape)

  # Compute how many anchors are actually used in the mini-batch (e.g.,
  # typically 256)
  N_cls = t.count_nonzero(y_mask) + epsilon

  # Compute element-wise loss for all anchors
  loss_all_anchors = F.binary_cross_entropy(input = y_predicted_class, target = y_true_class, reduction = "none")

  # Zero out the ones which should not have been included
  relevant_loss_terms = y_mask * loss_all_anchors

  # Sum the total loss and normalize by the number of anchors used
  return t.sum(relevant_loss_terms) / N_cls

def rpn_regression_loss_f(predicted_box_deltas, y_true):
  """
  Computes RPN box delta regression loss.

  Parameters
  ----------
  predicted_box_deltas : torch.Tensor
    A tensor of shape (batch_size, height, width, num_anchors * 4) containing
    RoI box delta regressions for each anchor, stored as: ty, tx, th, tw.
  y_true : torch.Tensor
    Ground truth tensor of shape (batch_size, height, width, num_anchors, 6).

  Returns
  -------
  torch.Tensor
    Scalar loss.
  """
  epsilon = 1e-7
  scale_factor = 1.0  # hyper-parameter that controls magnitude of regression loss and is chosen to make regression term comparable to class term
  sigma = 3.0         # see: https://github.com/rbgirshick/py-faster-rcnn/issues/89
  sigma_squared = sigma * sigma

  y_predicted_regression = predicted_box_deltas
  y_true_regression = y_true[:,:,:,:,2:6].reshape(y_predicted_regression.shape)

  # Include only anchors that are used in the mini-batch and which correspond
  # to objects (positive samples)
  y_included = y_true[:,:,:,:,0].reshape(y_true.shape[0:4]) # trainable anchors map: (batch_size, height, width, num_anchors)
  y_positive = y_true[:,:,:,:,1].reshape(y_true.shape[0:4]) # positive anchors
  y_mask = y_included * y_positive

  # y_mask is of the wrong shape. We have one value per (y,x,k) position but in
  # fact need to have 4 values (one for each of the regression variables). For
  # example, y_predicted might be (1,37,50,36) and y_mask will be (1,37,50,9).
  # We need to repeat the last dimension 4 times.
  y_mask = y_mask.repeat_interleave(repeats = 4, dim = 3)

  # The paper normalizes by dividing by a quantity called N_reg, which is equal
  # to the total number of anchors (~2400) and then multiplying by lambda=10.
  # This does not make sense to me because we are summing over a mini-batch at
  # most, so we use N_cls here. I might be misunderstanding what is going on
  # but 10/2400 = 1/240 which is pretty close to 1/256 and the paper mentions
  # that training is relatively insensitve to choice of normalization.
  N_cls = t.count_nonzero(y_included) + epsilon

  # Compute element-wise loss using robust L1 function for all 4 regression
  # components
  x = y_true_regression - y_predicted_regression
  x_abs = t.abs(x)
  is_negative_branch = (x_abs < (1.0 / sigma_squared)).float()
  R_negative_branch = 0.5 * x * x * sigma_squared
  R_positive_branch = x_abs - 0.5 / sigma_squared
  loss_all_anchors = is_negative_branch * R_negative_branch + (1.0 - is_negative_branch) * R_positive_branch

  # Zero out the ones which should not have been included
  relevant_loss_terms = y_mask * loss_all_anchors
  return scale_factor * t.sum(relevant_loss_terms) / N_cls

def detector_class_loss_f(predicted_classes, y_true):
  """
  Computes detector class loss.

  Parameters
  ----------
  predicted_classes : torch.Tensor
    RoI predicted classes as categorical vectors, (N, num_classes).
  y_true : torch.Tensor
    RoI class labels as categorical vectors, (N, num_classes).

  Returns
  -------
  torch.Tensor
    Scalar loss.
  """
  epsilon = 1e-7
  scale_factor = 1.0
  cross_entropy_per_row = -(y_true * t.log(predicted_classes + epsilon)).sum(dim = 1)
  N = cross_entropy_per_row.shape[0] + epsilon
  cross_entropy = t.sum(cross_entropy_per_row) / N
  return scale_factor * cross_entropy

def detector_regression_loss_f(predicted_box_deltas, y_true):
  """
  Computes detector regression loss.

  Parameters
  ----------
  predicted_box_deltas : torch.Tensor
    RoI predicted box delta regressions, (N, 4*(num_classes-1)). The background
    class is excluded and only the non-background classes are included. Each
    set of box deltas is stored in parameterized form as (ty, tx, th, tw).
  y_true : torch.Tensor
    RoI box delta regression ground truth labels, (N, 2, 4*(num_classes-1)).
    These are stored as mask values (1 or 0) in (:,0,:) and regression
    parameters in (:,1,:). Note that it is important to mask off the predicted
    and ground truth values because they may be set to invalid values.

  Returns
  -------
  torch.Tensor
    Scalar loss.
  """
  epsilon = 1e-7
  scale_factor = 1.0
  sigma = 1.0
  sigma_squared = sigma * sigma

  # We want to unpack the regression targets and the mask of valid targets into
  # tensors each of the same shape as the predicted:
  #   (num_proposals, 4*(num_classes-1))
  # y_true has shape:
  #   (num_proposals, 2, 4*(num_classes-1))
  y_mask = y_true[:,0,:]
  y_true_targets = y_true[:,1,:]

  # Compute element-wise loss using robust L1 function for all 4 regression
  # targets
  x = y_true_targets - predicted_box_deltas
  x_abs = t.abs(x)
  is_negative_branch = (x_abs < (1.0 / sigma_squared)).float()
  R_negative_branch = 0.5 * x * x * sigma_squared
  R_positive_branch = x_abs - 0.5 / sigma_squared
  losses = is_negative_branch * R_negative_branch + (1.0 - is_negative_branch) * R_positive_branch

  # Normalize to number of proposals (e.g., 128). Although this may not be
  # what the paper does, it seems to work. Other implemetnations do this.
  # Using e.g., the number of positive proposals will cause the loss to
  # behave erratically because sometimes N will become very small.
  N = y_true.shape[0] + epsilon
  relevant_loss_terms = y_mask * losses
  return scale_factor * t.sum(relevant_loss_terms) / N
