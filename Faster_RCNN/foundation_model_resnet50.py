from math import ceil
import torch as t
from torch import nn
from torch.nn import functional as F
import torchvision

from dataclasses import dataclass

@dataclass
class PreprocessingParams:
  """
  Image preprocessing parameters. Channel order may be either ChannelOrder.RGB or ChannelOrder.BGR.
  Scaling factor is applied first, followed by standardization with supplied means and standard
  deviations supplied in the order specified by channel_order.
  """
  channel_order: str
  scaling: float
  means: List[float]
  stds: List[float]
  
class FeatureExtractor(nn.Module):
  def __init__(self, resnet):
    super().__init__()

    # Feature extractor layers
    self._feature_extractor = nn.Sequential(
      resnet.conv1,     # 0
      resnet.bn1,       # 1
      resnet.relu,      # 2
      resnet.maxpool,   # 3
      resnet.layer1,    # 4
      resnet.layer2,    # 5
      resnet.layer3     # 6
    )

    # Freeze initial layers
    self._freeze(resnet.conv1)
    self._freeze(resnet.bn1)
    self._freeze(resnet.layer1)

    # Ensure that all batchnorm layers are frozen
    self._freeze_batchnorm(self._feature_extractor)

  # Override nn.Module.train()
  def train(self, mode = True):
    super().train(mode)

    #
    # During training, set all frozen blocks to evaluation mode and ensure that
    # all the batchnorm layers are also in evaluation mode. This is extremely
    # important and neglecting to do this will result in severely degraded
    # training performance.
    #
    if mode:
      # Set fixed blocks to be in eval mode
      self._feature_extractor.eval()
      self._feature_extractor[5].train()
      self._feature_extractor[6].train()

      # *All* batchnorm layers in eval mode
      def set_bn_eval(module):
        if type(module) == nn.BatchNorm2d:
          module.eval()
      self._feature_extractor.apply(set_bn_eval)

  def forward(self, image_data):
    y = self._feature_extractor(image_data)
    return y

  @staticmethod
  def _freeze(layer):
    for name, parameter in layer.named_parameters():
      parameter.requires_grad = False

  def _freeze_batchnorm(self, block):
    for child in block.modules():
      if type(child) == nn.BatchNorm2d:
        self._freeze(layer = child)







class PoolToFeatureVector(nn.Module):
  def __init__(self, resnet):
    super().__init__()
    self._layer4 = resnet.layer4
    self._freeze_batchnorm(self._layer4)

  def train(self, mode = True):
    # See comments in FeatureVector.train()
    super().train(mode)
    if mode:
      def set_bn_eval(module):
        if type(module) == nn.BatchNorm2d:
          module.eval()
      self._layer4.apply(set_bn_eval)

  def forward(self, rois):
    y = self._layer4(rois)  # (N, 1024, 7, 7) -> (N, 2048, 4, 4)

    # Average together the last two dimensions to remove them -> (N, 2048).
    # It is also possible to max pool, e.g.:
    # y = F.adaptive_max_pool2d(y, output_size = 1).squeeze()
    # This may even be better (74.96% mAP for ResNet50 vs. 73.2% using the
    # current method).
    y = y.mean(-1).mean(-1) # use mean to remove last two dimensions -> (N, 2048)
    return y

  @staticmethod
  def _freeze(layer):
    for name, parameter in layer.named_parameters():
      parameter.requires_grad = False

  def _freeze_batchnorm(self, block):
    for child in block.modules():
      if type(child) == nn.BatchNorm2d:
        self._freeze(layer = child)












# provide feature extraction and pooled feature reduction layers from the classifier stages.
#
# The backbone in Faster R-CNN is used in two places:
#
#   1. In Stage 1 as the feature extractor. Given an input image, a feature map
#      is produced that is then passed into both the RPN and detector stages.
#   2. In Stage 3, the detector, proposal regions are pooled and cropped from
#      the feature map (to produce RoIs) and fed into the detector layers,
#      which perform classification and bounding box regression. Each RoI must
#      first be converted into a linear feature vector. With VGG-16, for
#      example, the fully-connected layers following the convolutional layers
#      and preceding the classifier layer, are used to do this.


class ResNetBackbone():
  def __init__(self):

    # Backbone properties. Image preprocessing parameters are common to all Torchvision ResNet models and are described in the documentation, e.g.,
    # https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50
    self.feature_map_channels = 1024  # feature extractor output channels
    self.feature_pixels = 16          # ResNet feature maps are 1/16th of the original image size
    self.feature_vector_size = 2048   # linear feature vector size after pooling # length of linear feature vector after pooling and just before being passed to detector heads
    self.image_preprocessing_params = PreprocessingParams(channel_order = "RGB", scaling = 1.0 / 255.0, means = [ 0.485, 0.456, 0.406 ], stds = [ 0.229, 0.224, 0.225 ])

    # Loading IMAGENET1K_V1 pre-trained weights for Torchvision resnet50 backbone
    resnet = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1)

    # Feature extractor: given image data of shape (batch_size, channels, width, height), produces a feature map of shape (batch_size, feature_map_channels = 1024, W = ceil(height/16), H = ceil(width/16))
    self.feature_extractor = FeatureExtractor(resnet = resnet)

    # Conversion of pooled features i.e. RoIs (N, feature_map_channels, 7, 7) to head input i.e. (N, feature_vector_size)
    self.pool_to_feature_vector = PoolToFeatureVector(resnet = resnet)

  def compute_feature_map_shape(self, image_shape):
    """
    Computes the shape of the feature extractor output given an input image
    shape. This is used primarily for anchor generation and depends entirely on
    the architecture of the backbone.

    Unlike VGG-16, ResNet
    convolutional layers use padding and the resultant dimensions are therefore
    not simply an integral division by 16. The calculation here works well
    enough but it is not guaranteed that the simple conversion of feature map
    coordinates to input image pixel coordinates in anchors.py is absolutely
    correct.

    Parameters
    ----------
    image_shape : Tuple[int, int, int]
      Shape of the input image, (channels, height, width). Only the last two
      dimensions are relevant, allowing image_shape to be either the shape
      of a single image or the entire batch.

    Returns
    -------
    Tuple[int, int, int]
      Shape of the feature map produced by the feature extractor,
      (feature_map_channels, feature_map_height, feature_map_width).
    """


    image_width = image_shape[-1]
    image_height = image_shape[-2]
    return (self.feature_map_channels, ceil(image_height / self.feature_pixels), ceil(image_width / self.feature_pixels))
