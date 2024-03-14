# from CV.Faster_RCNN.util_anchor import generate_rpn_map, generate_anchor_maps
# from CV.Faster_RCNN.util_image import load_image
from dataclasses import dataclass
import numpy as np
from PIL import Image
from typing import List
from typing import Tuple
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import random



# Anchor generation code
#
# Differing from other implementations of Faster R-CNN, I generate a multi-
# dimensional ground truth tensor for the RPN stage that contains a flag
# indicating whether the anchor should be included in training, whether it is
# an object, and the box delta regression targets. It would be simpler and more
# performant to simply return 2D tensors with this information (the model ends
# up converting proposals into lists at a later stage anyway) but this is how
# I first thought to implement it and did not encounter a pressing need to
# change it.
from CV.Faster_RCNN.util_math import intersection_over_union

import itertools
from math import sqrt
import numpy as np

def _compute_anchor_sizes():
  #
  # Anchor scales and aspect ratios.
  #
  # x * y = area          x * (x_aspect * x) = x_aspect * x^2 = area
  # x_aspect * x = y  ->  x = sqrt(area / x_aspect)
  #                       y = x_aspect * sqrt(area / x_aspect)
  #
  areas = [ 128*128, 256*256, 512*512 ]   # pixels
  x_aspects = [ 0.5, 1.0, 2.0 ]           # x:1 ratio

  # Generate all 9 combinations of area and aspect ratio
  heights = np.array([ x_aspects[j] * sqrt(areas[i] / x_aspects[j]) for (i, j) in itertools.product(range(3), range(3)) ])
  widths = np.array([ sqrt(areas[i] / x_aspects[j]) for (i, j) in itertools.product(range(3), range(3)) ])

  # Return as (9,2) matrix of sizes
  return np.vstack([ heights, widths ]).T

def generate_anchor_maps(image_shape, feature_map_shape, feature_pixels):
  """
  Generates maps defining the anchors for a given input image size. There are 9
  different anchors at each feature map cell (3 scales, 3 ratios).

  Parameters
  ----------
  image_shape : Tuple[int, int, int]
    Shape of the input image, (channels, height, width), at the scale it will
    be passed into the Faster R-CNN model.
  feature_map_shape : Tuple[int, int, int]
    Shape of the output feature map, (channels, height, width).
  feature_pixels : int
    Distance in pixels between anchors. This is the size, in input image space,
    of each cell of the feature map output by the feature extractor stage of
    the Faster R-CNN network.

  Returns
  -------
  np.ndarray, np.ndarray
    Two maps, with height and width corresponding to the feature map
    dimensions, not the input image:
      1. A map of shape (height, width, num_anchors*4) containing all anchors,
         each stored as (center_y, center_x, anchor_height, anchor_width) in
         input image pixel space.
      2. A map of shape (height, width, num_anchors) indicating which anchors
         are valid (1) or invalid (0). Invalid anchors are those that cross
         image boundaries and must not be used during training.
  """

  assert len(image_shape) == 3

  #
  # Note that precision can strongly affect anchor labeling in some images.
  # Conversion of both operands to float32 matches the implementation by Yun
  # Chen. That is, changing the final line so as to eliminate the conversion to
  # float32:
  #
  #   return anchor_map, anchor_valid_map
  #
  # Has a pronounced effect on positive anchors in image 2008_000028.jpg in
  # VOC2012.
  #

  # Base anchor template: (num_anchors,4), with each anchor being specified by
  # its corners (y1,x1,y2,x2)
  anchor_sizes = _compute_anchor_sizes()
  num_anchors = anchor_sizes.shape[0]
  anchor_template = np.empty((num_anchors, 4))
  anchor_template[:,0:2] = -0.5 * anchor_sizes  # y1, x1 (top-left)
  anchor_template[:,2:4] = +0.5 * anchor_sizes  # y2, x2 (bottom-right)

  # Shape of map, (H,W), determined by feature extractor backbone
  height = feature_map_shape[-2]  # index from back in case batch dimension is supplied
  width = feature_map_shape[-1]

  # Generate (H,W,2) map of coordinates, in feature space, each being [y,x]
  y_cell_coords = np.arange(height)
  x_cell_coords = np.arange(width)
  cell_coords = np.array(np.meshgrid(y_cell_coords, x_cell_coords)).transpose([2, 1, 0])

  # Convert all coordinates to image space (pixels) at *center* of each cell
  center_points = cell_coords * feature_pixels + 0.5 * feature_pixels

  # (H,W,2) -> (H,W,4), repeating the last dimension so it contains (y,x,y,x)
  center_points = np.tile(center_points, reps = 2)

  # (H,W,4) -> (H,W,4*num_anchors)
  center_points = np.tile(center_points, reps = num_anchors)

  #
  # Now we can create the anchors by adding the anchor template to each cell
  # location. Anchor template is flattened to size num_anchors * 4 to make
  # the addition possible (along the last dimension).
  #
  anchors = center_points.astype(np.float32) + anchor_template.flatten()

  # (H,W,4*num_anchors) -> (H*W*num_anchors,4)
  anchors = anchors.reshape((height*width*num_anchors, 4))

  # Valid anchors are those that do not cross image boundaries
  image_height, image_width = image_shape[1:]
  valid = np.all((anchors[:,0:2] >= [0,0]) & (anchors[:,2:4] <= [image_height,image_width]), axis = 1)

  # Convert anchors to anchor format: (center_y, center_x, height, width)
  anchor_map = np.empty((anchors.shape[0], 4))
  anchor_map[:,0:2] = 0.5 * (anchors[:,0:2] + anchors[:,2:4])
  anchor_map[:,2:4] = anchors[:,2:4] - anchors[:,0:2]

  # Reshape maps and return
  anchor_map = anchor_map.reshape((height, width, num_anchors * 4))
  anchor_valid_map = valid.reshape((height, width, num_anchors))
  return anchor_map.astype(np.float32), anchor_valid_map.astype(np.float32)

def generate_rpn_map(anchor_map, anchor_valid_map, gt_boxes, object_iou_threshold = 0.7, background_iou_threshold = 0.3):
  """
  Generates a map containing ground truth data for training the region proposal
  network.

  Parameters
  ----------
  anchor_map : np.ndarray
    Map of shape (height, width, num_anchors*4) defining the anchors as
    (center_y, center_x, anchor_height, anchor_width) in input image space.
  anchor_valid_map : np.ndarray
    Map of shape (height, width, num_anchors) defining anchors that are valid
    and may be included in training.
  gt_boxes : List[training_sample.Box]
    List of ground truth boxes.
  object_iou_threshold : float
    IoU threshold between an anchor and a ground truth box above which an
    anchor is labeled as an object (positive) anchor.
  background_iou_threshold : float
    IoU threshold below which an anchor is labeled as background (negative).

  Returns
  -------
  np.ndarray, np.ndarray, np.ndarray
    RPN ground truth map, object (positive) anchor indices, and background
    (negative) anchor indices. Map height and width dimensions are in feature
    space.
    1. RPN ground truth map of shape (height, width, num_anchors, 6) where the
       last dimension is:
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
    2. Map of shape (N, 3) of indices (y, x, k) of all N object anchors in the
       RPN ground truth map.
    3. Map of shape (M, 3) of indices of all M background anchors in the RPN
       ground truth map.
  """
  height, width, num_anchors = anchor_valid_map.shape

  # Convert ground truth box corners to (M,4) tensor and class indices to (M,)
  gt_box_corners = np.array([ box.corners for box in gt_boxes ])
  num_gt_boxes = len(gt_boxes)

  # Compute ground truth box center points and side lengths
  gt_box_centers = 0.5 * (gt_box_corners[:,0:2] + gt_box_corners[:,2:4])
  gt_box_sides = gt_box_corners[:,2:4] - gt_box_corners[:,0:2]

  # Flatten anchor boxes to (N,4) and convert to corners
  anchor_map = anchor_map.reshape((-1,4))
  anchors = np.empty(anchor_map.shape)
  anchors[:,0:2] = anchor_map[:,0:2] - 0.5 * anchor_map[:,2:4]  # y1, x1
  anchors[:,2:4] = anchor_map[:,0:2] + 0.5 * anchor_map[:,2:4]  # y2, x2
  n = anchors.shape[0]

  # Initialize all anchors initially as negative (background). We will also
  # track which ground truth box was assigned to each anchor.
  objectness_score = np.full(n, -1)   # RPN class: 0 = background, 1 = foreground, -1 = ignore (these will be marked as invalid in the truth map)
  gt_box_assignments = np.full(n, -1) # -1 means no box

  # Compute IoU between each anchor and each ground truth box, (N,M).
  ious = intersection_over_union(boxes1 = anchors, boxes2 = gt_box_corners)

  # Need to remove anchors that are invalid (straddle image boundaries) from
  # consideration entirely and the easiest way to do this is to wipe out their
  # IoU scores
  ious[anchor_valid_map.flatten() == 0, :] = -1.0

  # Find the best IoU ground truth box for each anchor and the best IoU anchor
  # for each ground truth box.
  #
  # Note that ious == max_iou_per_gt_box tests each of the N rows of ious
  # against the M elements of max_iou_per_gt_box, column-wise. np.where() then
  # returns all (y,x) indices of matches as a tuple: (y_indices, x_indices).
  # The y indices correspond to the N dimension and therefore indicate anchors
  # and the x indices correspond to the M dimension (ground truth boxes).
  max_iou_per_anchor = np.max(ious, axis = 1)           # (N,)
  best_box_idx_per_anchor = np.argmax(ious, axis = 1)   # (N,)
  max_iou_per_gt_box = np.max(ious, axis = 0)           # (M,)
  highest_iou_anchor_idxs = np.where(ious == max_iou_per_gt_box)[0] # get (L,) indices of anchors that are the highest-overlapping anchors for at least one of the M boxes

  # Anchors below the minimum threshold are negative
  objectness_score[max_iou_per_anchor < background_iou_threshold] = 0

  # Anchors that meet the threshold IoU are positive
  objectness_score[max_iou_per_anchor >= object_iou_threshold] = 1

  # Anchors that overlap the most with ground truth boxes are positive
  objectness_score[highest_iou_anchor_idxs] = 1

  # We assign the highest IoU ground truth box to each anchor. If no box met
  # the IoU threshold, the highest IoU box may happen to be a box for which
  # the anchor had the highest IoU. If not, then the objectness score will be
  # negative and the box regression won't ever be used.
  gt_box_assignments[:] = best_box_idx_per_anchor

  # Anchors that are to be ignored will be marked invalid. Generate a mask to
  # multiply anchor_valid_map by (-1 -> 0, 0 or 1 -> 1). Then mark ignored
  # anchors as 0 in objectness score because the score can only really be 0 or
  # 1.
  enable_mask = (objectness_score >= 0).astype(np.float32)
  objectness_score[objectness_score < 0] = 0

  # Compute box delta regression targets for each anchor
  box_delta_targets = np.empty((n, 4))
  box_delta_targets[:,0:2] = (gt_box_centers[gt_box_assignments] - anchor_map[:,0:2]) / anchor_map[:,2:4] # ty = (box_center_y - anchor_center_y) / anchor_height, tx = (box_center_x - anchor_center_x) / anchor_width
  box_delta_targets[:,2:4] = np.log(gt_box_sides[gt_box_assignments] / anchor_map[:,2:4])                 # th = log(box_height / anchor_height), tw = log(box_width / anchor_width)

  # Assemble RPN ground truth map
  rpn_map = np.zeros((height, width, num_anchors, 6))
  rpn_map[:,:,:,0] = anchor_valid_map * enable_mask.reshape((height,width,num_anchors))  # trainable anchors (object or background; excludes boundary-crossing invalid and neutral anchors)
  rpn_map[:,:,:,1] = objectness_score.reshape((height,width,num_anchors))
  rpn_map[:,:,:,2:6] = box_delta_targets.reshape((height,width,num_anchors,4))

  # Return map along with positive and negative anchors
  rpn_map_coords = np.transpose(np.mgrid[0:height,0:width,0:num_anchors], (1,2,3,0))                  # shape (height,width,k,3): every index (y,x,k,:) returns its own coordinate (y,x,k)
  object_anchor_idxs = rpn_map_coords[np.where((rpn_map[:,:,:,1] > 0) & (rpn_map[:,:,:,0] > 0))]      # shape (N,3), where each row is the coordinate (y,x,k) of a positive sample
  background_anchor_idxs = rpn_map_coords[np.where((rpn_map[:,:,:,1] == 0) & (rpn_map[:,:,:,0] > 0))] # shape (N,3), where each row is the coordinate (y,x,k) of a negative sample

  return rpn_map.astype(np.float32), object_anchor_idxs, background_anchor_idxs








# Image loading and pre-processing.
import imageio
from PIL import Image
import numpy as np
from typing import List

def _compute_scale_factor(original_width, original_height, min_dimension_pixels):
  if not min_dimension_pixels:
    return 1.0
  if original_width > original_height:
    scale_factor = min_dimension_pixels / original_height
  else:
    scale_factor = min_dimension_pixels / original_width
  return scale_factor

def _preprocess_vgg16(image_data, preprocessing):
  image_data[:, :, 0] *= preprocessing.scaling
  image_data[:, :, 1] *= preprocessing.scaling
  image_data[:, :, 2] *= preprocessing.scaling
  image_data[:, :, 0] = (image_data[:, :, 0] - preprocessing.means[0]) / preprocessing.stds[0]
  image_data[:, :, 1] = (image_data[:, :, 1] - preprocessing.means[1]) / preprocessing.stds[1]
  image_data[:, :, 2] = (image_data[:, :, 2] - preprocessing.means[2]) / preprocessing.stds[2]
  image_data = image_data.transpose([2, 0, 1])  # (height,width,3) -> (3,height,width)
  return image_data.copy()                      # copy required to eliminate negative stride (which Torch doesn't like)

def load_image(url, preprocessing, min_dimension_pixels = None, horizontal_flip = False):
  """
  Loads and preprocesses an image for use with the Faster R-CNN model.
  This involves standardizing image pixels to ImageNet dataset-level
  statistics and ensuring channel order matches what the model's
  backbone (feature extractor) expects. The image can be resized so
  that the minimum dimension is a defined size, as recommended by
  Faster R-CNN.

  Parameters
  ----------
  url : str
    URL (local or remote file) to load.
  preprocessing : PreprocessingParams
    Image pre-processing parameters governing channel order and normalization.
  min_dimension_pixels : int
    If not None, specifies the size in pixels of the smaller side of the image.
    The other side is scaled proportionally.
  horizontal_flip : bool
    Whether to flip the image horizontally.

  Returns
  -------
  np.ndarray, PIL.Image, float, Tuple[int, int, int]
    Image pixels as float32, shaped as (channels, height, width); an image
    object suitable for drawing and visualization; scaling factor applied to
    the image dimensions; and the original image shape.
  """
  data = imageio.imread(url, pilmode = "RGB")
  image = Image.fromarray(data, mode = "RGB")
  original_width, original_height = image.width, image.height
  if horizontal_flip:
    image = image.transpose(method = Image.FLIP_LEFT_RIGHT)
  if min_dimension_pixels is not None:
    scale_factor = _compute_scale_factor(original_width = image.width, original_height = image.height, min_dimension_pixels = min_dimension_pixels)
    width = int(image.width * scale_factor)
    height = int(image.height * scale_factor)
    image = image.resize((width, height), resample = Image.BILINEAR)
  else:
    scale_factor = 1.0
  image_data = np.array(image).astype(np.float32)
  image_data = _preprocess_vgg16(image_data = image_data, preprocessing = preprocessing)
  return image_data, image, scale_factor, (image_data.shape[0], original_height, original_width)








@dataclass
class Box:
  class_index: int
  class_name: str
  corners: np.ndarray

  def __repr__(self):
    return "[class=%s (%f,%f,%f,%f)]" % (self.class_name, self.corners[0], self.corners[1], self.corners[2], self.corners[3])

  def __str__(self):
    return repr(self)

@dataclass
class TrainingSample:
  anchor_map:                 np.ndarray                # shape (feature_map_height,feature_map_width,num_anchors*4), with each anchor as [center_y,center_x,height,width]
  anchor_valid_map:           np.ndarray                # shape (feature_map_height,feature_map_width,num_anchors), indicating which anchors are valid (do not cross image boundaries)
  gt_rpn_map:                 np.ndarray                # TODO: describe me
  gt_rpn_object_indices:      List[Tuple[int,int,int]]  # list of (y,x,k) coordinates of anchors in gt_rpn_map that are labeled as object
  gt_rpn_background_indices:  List[Tuple[int,int,int]]  # list of (y,x,k) coordinates of background anchors
  gt_boxes:                   List[Box]                 # list of ground-truth boxes, scaled
  image_data:                 np.ndarray                # shape (3,height,width), pre-processed and scaled to size expected by model
  image:                      Image                     # PIL image data (for debug rendering), scaled
  filepath:                   str                       # file path of image


class Dataset:
  """
  A VOC dataset iterator for a particular split (train, val, etc.)
  """

  num_classes = 21
  class_index_to_name = {
    0:  "background",
    1:  "aeroplane",
    2:  "bicycle",
    3:  "bird",
    4:  "boat",
    5:  "bottle",
    6:  "bus",
    7:  "car",
    8:  "cat",
    9:  "chair",
    10: "cow",
    11: "diningtable",
    12: "dog",
    13: "horse",
    14: "motorbike",
    15: "person",
    16: "pottedplant",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "tvmonitor"
  }

  def __init__(self, split, image_preprocessing_params, compute_feature_map_shape_fn, feature_pixels = 16, dir = "VOCdevkit/VOC2007", augment = True, shuffle = True, allow_difficult = False, cache = True):
    """
    Parameters
    ----------
    split : str
      Dataset split to load: train, val, or trainval.
    image_preprocessing_params : dataset.image.PreprocessingParams
      Image preprocessing parameters to apply when loading images.
    compute_feature_map_shape_fn : Callable[Tuple[int, int, int], Tuple[int, int, int]]
      Function to compute feature map shape, (channels, height, width), from
      input image shape, (channels, height, width).
    feature_pixels : int
      Size of each cell in the Faster R-CNN feature map in image pixels. This
      is the separation distance between anchors.
    dir : str
      Root directory of dataset.
    augment : bool
      Whether to randomly augment (horizontally flip) images during iteration
      with 50% probability.
    shuffle : bool
      Whether to shuffle the dataset each time it is iterated.
    allow_difficult : bool
      Whether to include ground truth boxes that are marked as "difficult".
    cache : bool
      Whether to training samples in memory after first being generated.
    """
    self.split = split
    self._dir = dir
    self.class_index_to_name = self._get_classes()
    self.class_name_to_index = { class_name: class_index for (class_index, class_name) in self.class_index_to_name.items() }
    self.num_classes = len(self.class_index_to_name)
    assert self.num_classes == Dataset.num_classes, "Dataset does not have the expected number of classes (found %d but expected %d)" % (self.num_classes, Dataset.num_classes)
    assert self.class_index_to_name == Dataset.class_index_to_name, "Dataset does not have the expected class mapping"
    self._filepaths = self._get_filepaths()
    self.num_samples = len(self._filepaths)
    self._gt_boxes_by_filepath = self._get_ground_truth_boxes(filepaths = self._filepaths, allow_difficult = allow_difficult)
    self._i = 0
    self._iterable_filepaths = self._filepaths.copy()
    self._image_preprocessing_params = image_preprocessing_params
    self._compute_feature_map_shape_fn = compute_feature_map_shape_fn
    self._feature_pixels = feature_pixels
    self._augment = augment
    self._shuffle = shuffle
    self._cache = cache
    self._unaugmented_cached_sample_by_filepath = {}
    self._augmented_cached_sample_by_filepath = {}

  def __iter__(self):
    self._i = 0
    if self._shuffle:
      random.shuffle(self._iterable_filepaths)
    return self

  def __next__(self):
    if self._i >= len(self._iterable_filepaths):
      raise StopIteration

    # Next file to load
    filepath = self._iterable_filepaths[self._i]
    self._i += 1

    # Augment?
    flip = random.randint(0, 1) != 0 if self._augment else 0
    cached_sample_by_filepath = self._augmented_cached_sample_by_filepath if flip else self._unaugmented_cached_sample_by_filepath

    # Load and, if caching, write back to cache
    if filepath in cached_sample_by_filepath:
      sample = cached_sample_by_filepath[filepath]
    else:
      sample = self._generate_training_sample(filepath = filepath, flip = flip)
    if self._cache:
      cached_sample_by_filepath[filepath] = sample

    # Return the sample
    return sample

  def _generate_training_sample(self, filepath, flip):
    # Load and preprocess the image
    scaled_image_data, scaled_image, scale_factor, original_shape = load_image(url = filepath, preprocessing = self._image_preprocessing_params, min_dimension_pixels = 600, horizontal_flip = flip)
    _, original_height, original_width = original_shape

    # Scale ground truth boxes to new image size
    scaled_gt_boxes = []
    for box in self._gt_boxes_by_filepath[filepath]:
      if flip:
        corners = np.array([
          box.corners[0],
          original_width - 1 - box.corners[3],
          box.corners[2],
          original_width - 1 - box.corners[1]
        ])
      else:
        corners = box.corners
      scaled_box = Box(
        class_index = box.class_index,
        class_name = box.class_name,
        corners = corners * scale_factor
      )
      scaled_gt_boxes.append(scaled_box)

    # Generate anchor maps and RPN truth map
    anchor_map, anchor_valid_map = generate_anchor_maps(image_shape = scaled_image_data.shape, feature_map_shape = self._compute_feature_map_shape_fn(scaled_image_data.shape), feature_pixels = self._feature_pixels)
    gt_rpn_map, gt_rpn_object_indices, gt_rpn_background_indices = generate_rpn_map(anchor_map = anchor_map, anchor_valid_map = anchor_valid_map, gt_boxes = scaled_gt_boxes)

    # Return sample
    return TrainingSample(
      anchor_map = anchor_map,
      anchor_valid_map = anchor_valid_map,
      gt_rpn_map = gt_rpn_map,
      gt_rpn_object_indices = gt_rpn_object_indices,
      gt_rpn_background_indices = gt_rpn_background_indices,
      gt_boxes = scaled_gt_boxes,
      image_data = scaled_image_data,
      image = scaled_image,
      filepath = filepath
    )

  def _get_classes(self):
    imageset_dir = os.path.join(self._dir, "ImageSets", "Main")
    classes = set([ os.path.basename(path).split("_")[0] for path in Path(imageset_dir).glob("*_" + self.split + ".txt") ])
    assert len(classes) > 0, "No classes found in ImageSets/Main for '%s' split" % self.split
    class_index_to_name = { (1 + v[0]): v[1] for v in enumerate(sorted(classes)) }
    class_index_to_name[0] = "background"
    return class_index_to_name

  def _get_filepaths(self):
    image_list_file = os.path.join(self._dir, "ImageSets", "Main", self.split + ".txt")
    with open(image_list_file) as fp:
      basenames = [ line.strip() for line in fp.readlines() ] # strip newlines
    image_paths = [ os.path.join(self._dir, "JPEGImages", basename) + ".jpg" for basename in basenames ]
    return image_paths

    """
    # Debug: 60 car training images. Handy for quick iteration and testing.
    image_paths = [
      "2008_000028",
      "2008_000074",
      "2008_000085",
      "2008_000105",
      "2008_000109",
      "2008_000143",
      "2008_000176",
      "2008_000185",
      "2008_000187",
      "2008_000189",
      "2008_000193",
      "2008_000199",
      "2008_000226",
      "2008_000237",
      "2008_000252",
      "2008_000260",
      "2008_000315",
      "2008_000346",
      "2008_000356",
      "2008_000399",
      "2008_000488",
      "2008_000531",
      "2008_000563",
      "2008_000583",
      "2008_000595",
      "2008_000613",
      "2008_000619",
      "2008_000719",
      "2008_000833",
      "2008_000944",
      "2008_000953",
      "2008_000959",
      "2008_000979",
      "2008_001018",
      "2008_001039",
      "2008_001042",
      "2008_001104",
      "2008_001169",
      "2008_001196",
      "2008_001208",
      "2008_001274",
      "2008_001329",
      "2008_001359",
      "2008_001375",
      "2008_001440",
      "2008_001446",
      "2008_001500",
      "2008_001533",
      "2008_001541",
      "2008_001631",
      "2008_001632",
      "2008_001716",
      "2008_001746",
      "2008_001860",
      "2008_001941",
      "2008_002062",
      "2008_002118",
      "2008_002197",
      "2008_002202",
      "2011_003247"
    ]
    return [ os.path.join(self._dir, "JPEGImages", path) + ".jpg" for path in image_paths ]
    """

  def _get_ground_truth_boxes(self, filepaths, allow_difficult):
    gt_boxes_by_filepath = {}
    for filepath in filepaths:
      basename = os.path.splitext(os.path.basename(filepath))[0]
      annotation_file = os.path.join(self._dir, "Annotations", basename) + ".xml"
      tree = ET.parse(annotation_file)
      root = tree.getroot()
      assert tree != None, "Failed to parse %s" % annotation_file
      assert len(root.findall("size")) == 1
      size = root.find("size")
      assert len(size.findall("depth")) == 1
      depth = int(size.find("depth").text)
      assert depth == 3
      boxes = []
      for obj in root.findall("object"):
        assert len(obj.findall("name")) == 1
        assert len(obj.findall("bndbox")) == 1
        assert len(obj.findall("difficult")) == 1
        is_difficult = int(obj.find("difficult").text) != 0
        if is_difficult and not allow_difficult:
          continue  # ignore difficult examples unless asked to include them
        class_name = obj.find("name").text
        bndbox = obj.find("bndbox")
        assert len(bndbox.findall("xmin")) == 1
        assert len(bndbox.findall("ymin")) == 1
        assert len(bndbox.findall("xmax")) == 1
        assert len(bndbox.findall("ymax")) == 1
        x_min = int(bndbox.find("xmin").text) - 1  # convert to 0-based pixel coordinates
        y_min = int(bndbox.find("ymin").text) - 1
        x_max = int(bndbox.find("xmax").text) - 1
        y_max = int(bndbox.find("ymax").text) - 1
        corners = np.array([ y_min, x_min, y_max, x_max ]).astype(np.float32)
        box = Box(class_index = self.class_name_to_index[class_name], class_name = class_name, corners = corners)
        boxes.append(box)
      assert len(boxes) > 0
      gt_boxes_by_filepath[filepath] = boxes
    return gt_boxes_by_filepath
