# Image loading and pre-processing.
from dataclasses import dataclass
from enum import Enum
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
