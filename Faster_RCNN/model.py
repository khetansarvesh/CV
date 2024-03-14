from CV.Faster_RCNN.util_math import t_intersection_over_union, convert_deltas_to_boxes, t_convert_deltas_to_boxes

import numpy as np
import torch as t
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms
from torchvision.ops import RoIPool
from dataclasses import dataclass
import random

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





# PyTorch implementation of the RPN (region proposal network) stage of
# Faster R-CNN. Given a feature map (i.e., the output of the VGG-16
# convolutional layers), generates objectness scores for each anchor box, and
# boxes in the form of modifications to anchor center points and dimensions.
#
# Unlike the original Faster R-CNN implementation (and many subsequent re-
# implementations), which used two outputs per anchor (object and background)
# and a softmax activation, this implementation uses only a single output and
# sigmoid activation, which is simpler but equivalent. A value of < 0.5 is
# background and >= 0.5 is an object.
#
# The RPN class and box regression losses are defined here.

class RegionProposalNetwork(nn.Module):
  def __init__(self, feature_map_channels, allow_edge_proposals = False):
    super().__init__()

    # Constants
    self._allow_edge_proposals = allow_edge_proposals

    # Layers
    num_anchors = 9
    channels = feature_map_channels
    self._rpn_conv1 = nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size = (3, 3), stride = 1, padding = "same")
    self._rpn_class = nn.Conv2d(in_channels = channels, out_channels = num_anchors, kernel_size = (1, 1), stride = 1, padding = "same")
    self._rpn_boxes = nn.Conv2d(in_channels = channels, out_channels = num_anchors * 4, kernel_size = (1, 1), stride = 1, padding = "same")

    # Initialize weights
    self._rpn_conv1.weight.data.normal_(mean = 0.0, std = 0.01)
    self._rpn_conv1.bias.data.zero_()
    self._rpn_class.weight.data.normal_(mean = 0.0, std = 0.01)
    self._rpn_class.bias.data.zero_()
    self._rpn_boxes.weight.data.normal_(mean = 0.0, std = 0.01)
    self._rpn_boxes.bias.data.zero_()

  def forward(self, feature_map, image_shape, anchor_map, anchor_valid_map, max_proposals_pre_nms, max_proposals_post_nms):
    """
    Predict objectness scores and regress region-of-interest box proposals on
    an input feature map.

    Parameters
    ----------
    feature_map : torch.Tensor
      Feature map of shape (batch_size, feature_map_channels, height, width).
    image_shape : Tuple[int, int, int]
      Shapes of each image in pixels: (num_channels, height, width).
    anchor_map : np.ndarray
      Map of anchors, shaped (height, width, num_anchors * 4). The last
      dimension contains the anchor boxes specified as a 4-tuple of
      (center_y, center_x, height, width), repeated for all anchors at that
      coordinate of the feature map.
    anchor_valid_map : np.ndarray
      Map indicating which anchors are valid (do not intersect image bounds),
      shaped (height, width, num_anchors).
    max_proposals_pre_nms : int
      How many of the best proposals (sorted by objectness score) to extract
      before applying non-maximum suppression.
    max_proposals_post_nms : int
      How many of the best proposals (sorted by objectness score) to keep after
      non-maximum suppression.

    Returns
    -------
    torch.Tensor, torch.Tensor, torch.Tensor
      - Objectness scores (batch_size, height, width, num_anchors)
      - Box regressions (batch_size, height, width, num_anchors * 4), as box
        deltas (that is, (ty, tx, th, tw) for each anchor)
      - Proposals (N, 4) -- all corresponding proposal box corners stored as
        (y1, x1, y2, x2).
    """

    # Pass through the network
    y = F.relu(self._rpn_conv1(feature_map))
    objectness_score_map = t.sigmoid(self._rpn_class(y))
    box_deltas_map = self._rpn_boxes(y)

    # Transpose shapes to be more convenient:
    #   objectness_score_map -> (batch_size, height, width, num_anchors)
    #   box_deltas_map       -> (batch_size, height, width, num_anchors * 4)
    objectness_score_map = objectness_score_map.permute(0, 2, 3, 1).contiguous()
    box_deltas_map = box_deltas_map.permute(0, 2, 3, 1).contiguous()

    # Extract box deltas and anchors as (N,4) tensors and scores as (N,) list
    anchors, objectness_scores, box_deltas = self._extract_valid(
      anchor_map = anchor_map,
      anchor_valid_map = anchor_valid_map,
      objectness_score_map = objectness_score_map,
      box_deltas_map = box_deltas_map
    )

    # Detach from graph to avoid backprop. According to my understanding, this
    # should be redundant here because we later take care to detach the
    # proposals (in FasterRCNNModel). However, there is a memory leak involving
    # t_convert_deltas_to_boxes() if this is not done here. Ultimately, the
    # numerical results are not affected. Proposals returned from this function
    # are supposed to be constant and are fed into the detector stage. See any
    # commit prior to 209141c for an earlier version of the code here that
    # performed all operations on CPU using NumPy, which was slightly slower
    # but equivalent.
    box_deltas = box_deltas.detach()

    # Convert regressions to box corners
    proposals = t_convert_deltas_to_boxes(
      box_deltas = box_deltas,
      anchors = t.from_numpy(anchors).cuda(),
      box_delta_means = t.tensor([0, 0, 0, 0], dtype = t.float32, device = "cuda"),
      box_delta_stds = t.tensor([1, 1, 1, 1], dtype = t.float32, device = "cuda")
    )

    # Keep only the top-N scores. Note that we do not care whether the
    # proposals were labeled as objects (score > 0.5) and peform a simple
    # ranking among all of them. Restricting them has a strong adverse impact
    # on training performance.
    sorted_indices = t.argsort(objectness_scores)                   # sort in ascending order of objectness score
    sorted_indices = sorted_indices.flip(dims = (0,))               # descending order of score
    proposals = proposals[sorted_indices][0:max_proposals_pre_nms]  # grab the top-N best proposals
    objectness_scores = objectness_scores[sorted_indices][0:max_proposals_pre_nms]  # corresponding scores

    # Clip to image boundaries
    proposals[:,0:2] = t.clamp(proposals[:,0:2], min = 0)
    proposals[:,2] = t.clamp(proposals[:,2], max = image_shape[1])
    proposals[:,3] = t.clamp(proposals[:,3], max = image_shape[2])

    # Remove anything less than 16 pixels on a side
    height = proposals[:,2] - proposals[:,0]
    width = proposals[:,3] - proposals[:,1]
    idxs = t.where((height >= 16) & (width >= 16))[0]
    proposals = proposals[idxs]
    objectness_scores = objectness_scores[idxs]

    # Perform NMS
    idxs = nms(
      boxes = proposals,
      scores = objectness_scores,
      iou_threshold = 0.7
    )
    idxs = idxs[0:max_proposals_post_nms]
    proposals = proposals[idxs]

    # Return network outputs as PyTorch tensors and extracted object proposals
    return objectness_score_map, box_deltas_map, proposals

  def _extract_valid(self, anchor_map, anchor_valid_map, objectness_score_map, box_deltas_map):
    assert objectness_score_map.shape[0] == 1 # only batch size of 1 supported for now

    height, width, num_anchors = anchor_valid_map.shape
    anchors = anchor_map.reshape((height * width * num_anchors, 4))             # [N,4] all anchors
    anchors_valid = anchor_valid_map.reshape((height * width * num_anchors))    # [N,] whether anchors are valid (i.e., do not cross image boundaries)
    scores = objectness_score_map.reshape((height * width * num_anchors))       # [N,] prediced objectness scores
    box_deltas = box_deltas_map.reshape((height * width * num_anchors, 4))      # [N,4] predicted box delta regression targets

    if self._allow_edge_proposals:
      # Use all proposals
      return anchors, scores, box_deltas
    else:
      # Filter out those proposals generated at invalid anchors
      idxs = anchors_valid > 0
      return anchors[idxs], scores[idxs], box_deltas[idxs]




# PyTorch implementation of the final detector stage of Faster R-CNN. As input,
# takes a series of proposals (or RoIs) and produces classifications and boxes.
# The boxes are parameterized as modifications to the original incoming
# proposal boxes. That is, the proposal boxes are exactly analogous to the
# anchors that the RPN stage uses.

class DetectorNetwork(nn.Module):
  def __init__(self, num_classes, backbone):
    super().__init__()

    self._input_features = 7 * 7 * backbone.feature_map_channels

    # Define network
    self._roi_pool = RoIPool(output_size = (7, 7), spatial_scale = 1.0 / backbone.feature_pixels)
    self._pool_to_feature_vector = backbone.pool_to_feature_vector
    self._classifier = nn.Linear(in_features = backbone.feature_vector_size, out_features = num_classes)
    self._regressor = nn.Linear(in_features = backbone.feature_vector_size, out_features = (num_classes - 1) * 4)

    # Initialize weights
    self._classifier.weight.data.normal_(mean = 0.0, std = 0.01)
    self._classifier.bias.data.zero_()
    self._regressor.weight.data.normal_(mean = 0.0, std = 0.001)
    self._regressor.bias.data.zero_()

  def forward(self, feature_map, proposals):
    """
    Predict final class and box delta regressions for region-of-interest
    proposals. The proposals serve as "anchors" for the box deltas, which
    refine the proposals into final boxes.

    Parameters
    ----------
    feature_map : torch.Tensor
      Feature map of shape (batch_size, feature_map_channels, height, width).
    proposals : torch.Tensor
      Region-of-interest box proposals that are likely to contain objects.
      Has shape (N, 4), where N is the number of proposals, with each box given
      as (y1, x1, y2, x2) in pixel coordinates.

    Returns
    -------
    torch.Tensor, torch.Tensor
      Predicted classes, (N, num_classes), encoded as a one-hot vector, and
      predicted box delta regressions, (N, 4*(num_classes-1)), where the deltas
      are expressed as (ty, tx, th, tw) and are relative to each corresponding
      proposal box. Because there is no box for the background class 0, it is
      excluded entirely and only (num_classes-1) sets of box delta targets are
      computed.
    """
    # Batch size of one for now, so no need to associate proposals with batches
    assert feature_map.shape[0] == 1, "Batch size must be 1"
    batch_idxs = t.zeros((proposals.shape[0], 1)).cuda()

    # (N, 5) tensor of (batch_idx, x1, y1, x2, y2)
    indexed_proposals = t.cat([ batch_idxs, proposals ], dim = 1)
    indexed_proposals = indexed_proposals[:, [ 0, 2, 1, 4, 3 ]] # each row, (batch_idx, y1, x1, y2, x2) -> (batch_idx, x1, y1, x2, y2)

    # RoI pooling: (N, feature_map_channels, 7, 7)
    rois = self._roi_pool(feature_map, indexed_proposals)

    # Forward propagate
    y = self._pool_to_feature_vector(rois = rois)
    classes_raw = self._classifier(y)
    classes = F.softmax(classes_raw, dim = 1)
    box_deltas = self._regressor(y)

    return classes, box_deltas





# PyTorch implementation of Faster R-CNN training and inference models. Here,
# all stages of Faster R-CNN are instantiated, RPN mini-batches are sampled,
# ground truth labels from RPN proposal boxes (RoIs) for the detector stage are
# generated, and  proposals are sampled.


class FasterRCNNModel(nn.Module):
  @dataclass
  class Loss:
    rpn_class:            float
    rpn_regression:       float
    detector_class:       float
    detector_regression:  float
    total:                float

  def __init__(self, num_classes, backbone, rpn_minibatch_size = 256, proposal_batch_size = 128, allow_edge_proposals = True):
    """
    Parameters
    ----------
    num_classes : int
      Number of output classes.
    backbone : models.Backbone
      Backbone network for feature extraction and pooled feature vector
      construction (for input to detector heads).
    rpn_minibatch_size : int
      Size of the RPN mini-batch. The number of ground truth anchors sampled
      for training at each step.
    proposal_batch_size : int
      Number of region proposals to sample at each training step.
    allow_edge_proposals : bool
      Whether to use proposals generated at invalid anchors (those that
      straddle image edges). Invalid anchors are excluded from RPN training, as
      explicitly stated in the literature, but Faster R-CNN implementations
      tend to still pass proposals generated at invalid anchors to the
      detector.
    """
    super().__init__()

    # Constants
    self._num_classes = num_classes
    self._rpn_minibatch_size = rpn_minibatch_size
    self._proposal_batch_size = proposal_batch_size
    self._detector_box_delta_means = [ 0, 0, 0, 0 ]
    self._detector_box_delta_stds = [ 0.1, 0.1, 0.2, 0.2 ]

    # Backbone
    self.backbone = backbone

    # Network stages
    self._stage1_feature_extractor = backbone.feature_extractor
    self._stage2_region_proposal_network = RegionProposalNetwork( feature_map_channels = backbone.feature_map_channels, allow_edge_proposals = allow_edge_proposals)
    self._stage3_detector_network = DetectorNetwork(num_classes = num_classes, backbone = backbone)

  def forward(self, image_data, anchor_map = None, anchor_valid_map = None):
    """
    Forward inference. Use for test and evaluation only.

    Parameters
    ----------
    image_data : torch.Tensor
      A tensor of shape (batch_size, channels, height, width) representing
      images normalized using the VGG-16 convention (BGR, ImageNet channel-wise
      mean-centered).
    anchor_map : torch.Tensor
      Map of anchors, shaped (height, width, num_anchors * 4). The last
      dimension contains the anchor boxes specified as a 4-tuple of
      (center_y, center_x, height, width), repeated for all anchors at that
      coordinate of the feature map. If this or anchor_valid_map is not
      provided, both will be computed here.
    anchor_valid_map : torch.Tensor
      Map indicating which anchors are valid (do not intersect image bounds),
      shaped (height, width). If this or anchor_map is not provided, both will
      be computed here.

    Returns
    -------
    np.ndarray, torch.Tensor, torch.Tensor
      - Proposals (N, 4) from region proposal network
      - Classes (M, num_classes) from detector network
      - Box delta regressions (M, (num_classes - 1) * 4) from detector network
    """
    assert image_data.shape[0] == 1, "Batch size must be 1"
    image_shape = image_data.shape[1:]  # (batch_index, channels, height, width) -> (channels, height, width)

    # Anchor maps can be pre-computed and passed in explicitly (for performance
    # reasons) but if they are missing, we compute them on-the-fly here
    if anchor_map is None or anchor_valid_map is None:
      feature_map_shape = self.backbone.compute_feature_map_shape(image_shape = image_shape)
      anchor_map, anchor_valid_map = anchors.generate_anchor_maps(image_shape = image_shape, feature_map_shape = feature_map_shape, feature_pixels = self.backbone.feature_pixels)

    # Run each stage
    feature_map = self._stage1_feature_extractor(image_data = image_data)
    objectness_score_map, box_deltas_map, proposals = self._stage2_region_proposal_network(
      feature_map = feature_map,
      image_shape = image_shape,
      anchor_map = anchor_map,
      anchor_valid_map = anchor_valid_map,
      max_proposals_pre_nms = 6000, # test time values
      max_proposals_post_nms = 300
    )
    classes, box_deltas = self._stage3_detector_network(
      feature_map = feature_map,
      proposals = proposals
    )

    return proposals, classes, box_deltas

  # @utils.no_grad
  def predict(self, image_data, score_threshold, anchor_map = None, anchor_valid_map = None):
    """
    Performs inference on an image and obtains the final detected boxes.

    Parameters
    ----------
    image_data : torch.Tensor
      A tensor of shape (batch_size, channels, height, width) representing
      images normalized using the VGG-16 convention (BGR, ImageNet channel-wise
      mean-centered).
    score_threshold : float
      Minimum required score threshold (applied per class) for a detection to
      be considered. Set this higher for visualization to minimize extraneous
      boxes.
    anchor_map : torch.Tensor
      Map of anchors, shaped (height, width, num_anchors * 4). The last
      dimension contains the anchor boxes specified as a 4-tuple of
      (center_y, center_x, height, width), repeated for all anchors at that
      coordinate of the feature map. If this or anchor_valid_map is not
      provided, both will be computed here.
    anchor_valid_map : torch.Tensor
      Map indicating which anchors are valid (do not intersect image bounds),
      shaped (height, width). If this or anchor_map is not provided, both will
      be computed here.

    Returns
    -------
    Dict[int, np.ndarray]
      Scored boxes, (N, 5) tensor of box corners and class score,
      (y1, x1, y2, x2, score), indexed by class index.
    """
    self.eval()
    assert image_data.shape[0] == 1, "Batch size must be 1"

    # Forward inference
    proposals, classes, box_deltas = self(
      image_data = image_data,
      anchor_map = anchor_map,
      anchor_valid_map = anchor_valid_map
    )
    proposals = proposals.cpu().numpy()
    classes = classes.cpu().numpy()
    box_deltas = box_deltas.cpu().numpy()

    # Convert proposal boxes -> center point and size
    proposal_anchors = np.empty(proposals.shape)
    proposal_anchors[:,0] = 0.5 * (proposals[:,0] + proposals[:,2]) # center_y
    proposal_anchors[:,1] = 0.5 * (proposals[:,1] + proposals[:,3]) # center_x
    proposal_anchors[:,2:4] = proposals[:,2:4] - proposals[:,0:2]   # height, width

    # Separate out results per class: class_idx -> (y1, x1, y2, x2, score)
    boxes_and_scores_by_class_idx = {}
    for class_idx in range(1, classes.shape[1]):  # skip class 0 (background)
      # Get the box deltas (ty, tx, th, tw) corresponding to this class, for
      # all proposals
      box_delta_idx = (class_idx - 1) * 4
      box_delta_params = box_deltas[:, (box_delta_idx + 0) : (box_delta_idx + 4)] # (N, 4)
      proposal_boxes_this_class = convert_deltas_to_boxes(
        box_deltas = box_delta_params,
        anchors = proposal_anchors,
        box_delta_means = self._detector_box_delta_means,
        box_delta_stds = self._detector_box_delta_stds
      )

      # Clip to image boundaries
      proposal_boxes_this_class[:,0::2] = np.clip(proposal_boxes_this_class[:,0::2], 0, image_data.shape[2] - 1)  # clip y1 and y2 to [0,height)
      proposal_boxes_this_class[:,1::2] = np.clip(proposal_boxes_this_class[:,1::2], 0, image_data.shape[3] - 1)  # clip x1 and x2 to [0,width)

      # Get the scores for this class. The class scores are returned in
      # normalized categorical form. Each row corresponds to a class.
      scores_this_class = classes[:,class_idx]

      # Keep only those scoring high enough
      sufficiently_scoring_idxs = np.where(scores_this_class > score_threshold)[0]
      proposal_boxes_this_class = proposal_boxes_this_class[sufficiently_scoring_idxs]
      scores_this_class = scores_this_class[sufficiently_scoring_idxs]
      boxes_and_scores_by_class_idx[class_idx] = (proposal_boxes_this_class, scores_this_class)

    # Perform NMS per class
    scored_boxes_by_class_idx = {}
    for class_idx, (boxes, scores) in boxes_and_scores_by_class_idx.items():
      idxs = nms(
        boxes = t.from_numpy(boxes).cuda(),
        scores = t.from_numpy(scores).cuda(),
        iou_threshold = 0.3 #TODO: unsure about this. Paper seems to imply 0.5 but https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/fast_rcnn/config.py has 0.3 for test NMS
      ).cpu().numpy()
      boxes = boxes[idxs]
      scores = np.expand_dims(scores[idxs], axis = 0) # (N,) -> (N,1)
      scored_boxes = np.hstack([ boxes, scores.T ])   # (N,5), with each row: (y1, x1, y2, x2, score)
      scored_boxes_by_class_idx[class_idx] = scored_boxes

    return scored_boxes_by_class_idx

  def train_step(self, optimizer, image_data, anchor_map, anchor_valid_map, gt_rpn_map, gt_rpn_object_indices, gt_rpn_background_indices, gt_boxes):
    """
    Performs one training step on a sample of data.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
      Optimizer.
    image_data : torch.Tensor
      A tensor of shape (batch_size, channels, height, width) representing
      images normalized using the VGG-16 convention (BGR, ImageNet channel-wise
      mean-centered).
    anchor_map : torch.Tensor
      Map of anchors, shaped (height, width, num_anchors * 4). The last
      dimension contains the anchor boxes specified as a 4-tuple of
      (center_y, center_x, height, width), repeated for all anchors at that
      coordinate of the feature map. If this or anchor_valid_map is not
      provided, both will be computed here.
    anchor_valid_map : torch.Tensor
      Map indicating which anchors are valid (do not intersect image bounds),
      shaped (height, width). If this or anchor_map is not provided, both will
      be computed here.
    gt_rpn_map : torch.Tensor
      Ground truth RPN map of shape
      (batch_size, height, width, num_anchors, 6), where height and width are
      the feature map dimensions, not the input image dimensions. The final
      dimension contains:
       - 0: Trainable anchor (1) or not (0). Only valid and non-neutral (that
            is, definitely positive or negative) anchors are trainable. This is
            the same as anchor_valid_map with additional invalid anchors caused
            by neutral samples
       - 1: For trainable anchors, whether the anchor is an object anchor (1)
            or background anchor (0). For non-trainable anchors, will be 0.
       - 2: Regression target for box center, ty.
       - 3: Regression target for box center, tx.
       - 4: Regression target for box size, th.
       - 5: Regression target for box size, tw.
    gt_rpn_object_indices : List[np.ndarray]
      For each image in the batch, a map of shape (N, 3) of indices (y, x, k)
      of all N object anchors in the RPN ground truth map.
    gt_rpn_background_indices : List[np.ndarray]
      For each image in the batch, a map of shape (M, 3) of indices of all M
      background anchors in the RPN ground truth map.
    gt_boxes : List[List[datasets.training_sample.Box]]
      For each image in the batch, a list of ground truth object boxes.

    Returns
    -------
    Loss
      Loss (a dataclass with class and regression losses for both the RPN and
      detector states).
    """
    self.train()

    # Clear accumulated gradient
    optimizer.zero_grad()

    # For now, we only support a batch size of 1
    assert image_data.shape[0] == 1, "Batch size must be 1"
    assert len(gt_rpn_map.shape) == 5 and gt_rpn_map.shape[0] == 1, "Batch size must be 1"
    assert len(gt_rpn_object_indices) == 1, "Batch size must be 1"
    assert len(gt_rpn_background_indices) == 1, "Batch size must be 1"
    assert len(gt_boxes) == 1, "Batch size must be 1"
    image_shape = image_data.shape[1:]

    # Stage 1: Extract features
    feature_map = self._stage1_feature_extractor(image_data = image_data)

    # Stage 2: Generate object proposals using RPN
    rpn_score_map, rpn_box_deltas_map, proposals = self._stage2_region_proposal_network(
      feature_map = feature_map,
      image_shape = image_shape,  # each image in batch has identical shape: (num_channels, height, width)
      anchor_map = anchor_map,
      anchor_valid_map = anchor_valid_map,
      max_proposals_pre_nms = 12000,
      max_proposals_post_nms = 2000
    )

    # Sample random mini-batch of anchors (for RPN training)
    gt_rpn_minibatch_map = self._sample_rpn_minibatch(
      rpn_map = gt_rpn_map,
      object_indices = gt_rpn_object_indices,
      background_indices = gt_rpn_background_indices
    )

    # Assign labels to proposals and take random sample (for detector training)
    proposals, gt_classes, gt_box_deltas = self._label_proposals(
      proposals = proposals,
      gt_boxes = gt_boxes[0], # for now, batch size of 1
      min_background_iou_threshold = 0.0,
      min_object_iou_threshold = 0.5
    )
    proposals, gt_classes, gt_box_deltas = self._sample_proposals(
      proposals = proposals,
      gt_classes = gt_classes,
      gt_box_deltas = gt_box_deltas,
      max_proposals = self._proposal_batch_size,
      positive_fraction = 0.25
    )

    # Make sure RoI proposals and ground truths are detached from computational
    # graph so that gradients are not propagated through them. They are treated
    # as constant inputs into the detector stage.
    proposals = proposals.detach()
    gt_classes = gt_classes.detach()
    gt_box_deltas = gt_box_deltas.detach()

    # Stage 3: Detector
    detector_classes, detector_box_deltas = self._stage3_detector_network(
      feature_map = feature_map,
      proposals = proposals
    )

    # Compute losses
    rpn_class_loss = rpn_class_loss_f(predicted_scores = rpn_score_map, y_true = gt_rpn_minibatch_map)
    rpn_regression_loss = rpn_regression_loss_f(predicted_box_deltas = rpn_box_deltas_map, y_true = gt_rpn_minibatch_map)
    detector_class_loss = detector_class_loss_f(predicted_classes = detector_classes, y_true = gt_classes)
    detector_regression_loss = detector_regression_loss_f(predicted_box_deltas = detector_box_deltas, y_true = gt_box_deltas)
    total_loss = rpn_class_loss + rpn_regression_loss + detector_class_loss + detector_regression_loss
    loss = FasterRCNNModel.Loss(
      rpn_class = rpn_class_loss.detach().cpu().item(),
      rpn_regression = rpn_regression_loss.detach().cpu().item(),
      detector_class = detector_class_loss.detach().cpu().item(),
      detector_regression = detector_regression_loss.detach().cpu().item(),
      total = total_loss.detach().cpu().item()
    )

    # Backprop
    total_loss.backward()

    # Optimizer step
    optimizer.step()

    # Return losses and data useful for computing statistics
    return loss

  def _sample_rpn_minibatch(self, rpn_map, object_indices, background_indices):
    """
    Selects anchors for training and produces a copy of the RPN ground truth
    map with only those anchors marked as trainable.

    Parameters
    ----------
    rpn_map : np.ndarray
      RPN ground truth map of shape
      (batch_size, height, width, num_anchors, 6).
    object_indices : List[np.ndarray]
      For each image in the batch, a map of shape (N, 3) of indices (y, x, k)
      of all N object anchors in the RPN ground truth map.
    background_indices : List[np.ndarray]
      For each image in the batch, a map of shape (M, 3) of indices of all M
      background anchors in the RPN ground truth map.

    Returns
    -------
    np.ndarray
      A copy of the RPN ground truth map with index 0 of the last dimension
      recomputed to include only anchors in the minibatch.
    """
    assert rpn_map.shape[0] == 1, "Batch size must be 1"
    assert len(object_indices) == 1, "Batch size must be 1"
    assert len(background_indices) == 1, "Batch size must be 1"
    positive_anchors = object_indices[0]
    negative_anchors = background_indices[0]
    assert len(positive_anchors) + len(negative_anchors) >= self._rpn_minibatch_size, "Image has insufficient anchors for RPN minibatch size of %d" % self._rpn_minibatch_size
    assert len(positive_anchors) > 0, "Image does not have any positive anchors"
    assert self._rpn_minibatch_size % 2 == 0, "RPN minibatch size must be evenly divisible"

    # Sample, producing indices into the index maps
    num_positive_anchors = len(positive_anchors)
    num_negative_anchors = len(negative_anchors)
    num_positive_samples = min(self._rpn_minibatch_size // 2, num_positive_anchors) # up to half the samples should be positive, if possible
    num_negative_samples = self._rpn_minibatch_size - num_positive_samples          # the rest should be negative
    positive_anchor_idxs = random.sample(range(num_positive_anchors), num_positive_samples)
    negative_anchor_idxs = random.sample(range(num_negative_anchors), num_negative_samples)

    # Construct index expressions into RPN map
    positive_anchors = positive_anchors[positive_anchor_idxs]
    negative_anchors = negative_anchors[negative_anchor_idxs]
    trainable_anchors = np.concatenate([ positive_anchors, negative_anchors ])
    batch_idxs = np.zeros(len(trainable_anchors))
    trainable_idxs = (batch_idxs, trainable_anchors[:,0], trainable_anchors[:,1], trainable_anchors[:,2], 0)

    # Create a copy of the RPN map with samples set as trainable
    rpn_minibatch_map = rpn_map.clone()
    rpn_minibatch_map[:,:,:,:,0] = 0
    rpn_minibatch_map[trainable_idxs] = 1

    return rpn_minibatch_map

  def _label_proposals(self, proposals, gt_boxes, min_background_iou_threshold, min_object_iou_threshold):
    """
    Determines which proposals generated by the RPN stage overlap with ground
    truth boxes and creates ground truth labels for the subsequent detector
    stage.

    Parameters
    ----------
    proposals : torch.Tensor
      Proposal corners, shaped (N, 4).
    gt_boxes : List[datasets.training_sample.Box]
      Ground truth object boxes.
    min_background_iou_threshold : float
      Minimum IoU threshold with ground truth boxes below which proposals are
      ignored entirely. Proposals with an IoU threshold in the range
      [min_background_iou_threshold, min_object_iou_threshold) are labeled as
      background. This value can be greater than 0, which has the effect of
      selecting more difficult background examples that have some degree of
      overlap with ground truth boxes.
    min_object_iou_threshold : float
      Minimum IoU threshold for a proposal to be labeled as an object.

    Returns
    -------
    torch.Tensor, torch.Tensor, torch.Tensor
      Proposals, (N, 4), labeled as either objects or background (depending on
      IoU thresholds, some proposals can end up as neither and are excluded
      here); one-hot encoded class labels, (N, num_classes), for each proposal;
      and box delta regression targets, (N, 2, (num_classes - 1) * 4), for each
      proposal. Box delta target values are present at locations [:,1,:] and
      consist of (ty, tx, th, tw) for the class that the box corresponds to.
      The entries for all other classes and the background classes should be
      ignored. A mask is written to locations [:,0,:]. For each proposal
      assigned a non-background class, there will be 4 consecutive elements
      marked with 1 indicating the corresponding box delta target values are to
      be used. There are no box delta regression targets for background
      proposals and the mask is entirely 0 for those proposals.
    """
    assert min_background_iou_threshold < min_object_iou_threshold, "Object threshold must be greater than background threshold"

    # Convert ground truth box corners to (M,4) tensor and class indices to (M,)
    gt_box_corners = np.array([ box.corners for box in gt_boxes ], dtype = np.float32)
    gt_box_corners = t.from_numpy(gt_box_corners).cuda()
    gt_box_class_idxs = t.tensor([ box.class_index for box in gt_boxes ], dtype = t.long, device = "cuda")

    # Let's be crafty and create some fake proposals that match the ground
    # truth boxes exactly. This isn't strictly necessary and the model should
    # work without it but it will help training and will ensure that there are
    # always some positive examples to train on.
    proposals = t.vstack([ proposals, gt_box_corners ])

    # Compute IoU between each proposal (N,4) and each ground truth box (M,4)
    # -> (N, M)
    ious = t_intersection_over_union(boxes1 = proposals, boxes2 = gt_box_corners)

    # Find the best IoU for each proposal, the class of the ground truth box
    # associated with it, and the box corners
    best_ious = t.max(ious, dim = 1).values         # (N,) of maximum IoUs for each of the N proposals
    box_idxs = t.argmax(ious, dim = 1)              # (N,) of ground truth box index for each proposal
    gt_box_class_idxs = gt_box_class_idxs[box_idxs] # (N,) of class indices of highest-IoU box for each proposal
    gt_box_corners = gt_box_corners[box_idxs]       # (N,4) of box corners of highest-IoU box for each proposal

    # Remove all proposals whose best IoU is less than the minimum threshold
    # for a negative (background) sample. We also check for IoUs > 0 because
    # due to earlier clipping, we may get invalid 0-area proposals.
    idxs = t.where((best_ious >= min_background_iou_threshold))[0]  # keep proposals w/ sufficiently high IoU
    proposals = proposals[idxs]
    best_ious = best_ious[idxs]
    gt_box_class_idxs = gt_box_class_idxs[idxs]
    gt_box_corners = gt_box_corners[idxs]

    # IoUs less than min_object_iou_threshold will be labeled as background
    gt_box_class_idxs[best_ious < min_object_iou_threshold] = 0

    # One-hot encode class labels
    num_proposals = proposals.shape[0]
    gt_classes = t.zeros((num_proposals, self._num_classes), dtype = t.float32, device = "cuda")  # (N,num_classes)
    gt_classes[ t.arange(num_proposals), gt_box_class_idxs ] = 1.0

    # Convert proposals and ground truth boxes into "anchor" format (center
    # points and side lengths). For the detector stage, the proposals serve as
    # the anchors relative to which the final box predictions will be
    # regressed.
    proposal_centers = 0.5 * (proposals[:,0:2] + proposals[:,2:4])          # center_y, center_x
    proposal_sides = proposals[:,2:4] - proposals[:,0:2]                    # height, width
    gt_box_centers = 0.5 * (gt_box_corners[:,0:2] + gt_box_corners[:,2:4])  # center_y, center_x
    gt_box_sides = gt_box_corners[:,2:4] - gt_box_corners[:,0:2]            # height, width

    # Compute box delta regression targets (ty, tx, th, tw) for each proposal
    # based on the best box selected
    box_delta_targets = t.empty((num_proposals, 4), dtype = t.float32, device = "cuda") # (N,4)
    box_delta_targets[:,0:2] = (gt_box_centers - proposal_centers) / proposal_sides # ty = (gt_center_y - proposal_center_y) / proposal_height, tx = (gt_center_x - proposal_center_x) / proposal_width
    box_delta_targets[:,2:4] = t.log(gt_box_sides / proposal_sides)                 # th = log(gt_height / proposal_height), tw = (gt_width / proposal_width)
    box_delta_means = t.tensor(self._detector_box_delta_means, dtype = t.float32, device = "cuda")
    box_delta_stds = t.tensor(self._detector_box_delta_stds, dtype = t.float32, device = "cuda")
    box_delta_targets[:,:] -= box_delta_means                               # mean adjustment
    box_delta_targets[:,:] /= box_delta_stds                                # standard deviation scaling

    # Convert regression targets into a map of shape (N,2,4*(C-1)) where C is
    # the number of classes and [:,0,:] specifies a mask for the corresponding
    # target components at [:,1,:]. Targets are ordered (ty, tx, th, tw).
    # Background class 0 is not present at all.
    gt_box_deltas = t.zeros((num_proposals, 2, 4 * (self._num_classes - 1)), dtype = t.float32, device = "cuda")
    gt_box_deltas[:,0,:] = t.repeat_interleave(gt_classes, repeats = 4, dim = 1)[:,4:]  # create masks using interleaved repetition, remembering to ignore class 0
    gt_box_deltas[:,1,:] = t.tile(box_delta_targets, dims = (1, self._num_classes - 1)) # populate regression targets with straightforward repetition (only those columns corresponding to class are masked on)

    return proposals, gt_classes, gt_box_deltas

  def _sample_proposals(self, proposals, gt_classes, gt_box_deltas, max_proposals, positive_fraction):
    if max_proposals <= 0:
      return proposals, gt_classes, gt_box_deltas

    # Get positive and negative (background) proposals
    class_indices = t.argmax(gt_classes, axis = 1)  # (N,num_classes) -> (N,), where each element is the class index (highest score from its row)
    positive_indices = t.where(class_indices > 0)[0]
    negative_indices = t.where(class_indices <= 0)[0]
    num_positive_proposals = len(positive_indices)
    num_negative_proposals = len(negative_indices)

    # Select positive and negative samples, if there are enough. Note that the
    # number of positive samples can be either the positive fraction of the
    # *actual* number of proposals *or* the *desired* number (max_proposals).
    # In practice, these yield virtually identical results but the latter
    # method will yield slightly more positive samples in the rare cases when
    # the number of proposals is below the desired number. Here, we use the
    # former method but others, such as Yun Chen, use the latter. To implement
    # it, replace num_samples with max_proposals in the line that computes
    # num_positive_samples. I am not sure what the original Faster R-CNN
    # implementation does.
    num_samples = min(max_proposals, len(class_indices))
    num_positive_samples = min(round(num_samples * positive_fraction), num_positive_proposals)
    num_negative_samples = min(num_samples - num_positive_samples, num_negative_proposals)

    # Do we have enough?
    if num_positive_samples <= 0 or num_negative_samples <= 0:
      return proposals[[]], gt_classes[[]], gt_box_deltas[[]] # return 0-length tensors

    # Sample randomly
    positive_sample_indices = positive_indices[ t.randperm(len(positive_indices))[0:num_positive_samples] ]
    negative_sample_indices = negative_indices[ t.randperm(len(negative_indices))[0:num_negative_samples] ]
    indices = t.cat([ positive_sample_indices, negative_sample_indices ])

    # Return
    return proposals[indices], gt_classes[indices], gt_box_deltas[indices]
