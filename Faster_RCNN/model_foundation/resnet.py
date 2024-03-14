class ResNetBackbone(Backbone):
  def __init__(self):
    super().__init__()

    # Backbone properties. Image preprocessing parameters are common to all
    # Torchvision ResNet models and are described in the documentation, e.g.,
    # https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50
    self.feature_map_channels = 1024  # feature extractor output channels
    self.feature_pixels = 16          # ResNet feature maps are 1/16th of the original image size, similar to VGG-16 feature extractor
    self.feature_vector_size = 2048   # linear feature vector size after pooling
    self.image_preprocessing_params = PreprocessingParams(channel_order = ChannelOrder.RGB, scaling = 1.0 / 255.0, means = [ 0.485, 0.456, 0.406 ], stds = [ 0.229, 0.224, 0.225 ])

    # Loading IMAGENET1K_V1 pre-trained weights for Torchvision resnet50 backbone
    resnet = torchvision.models.resnet50(weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1)

    # Feature extractor: given image data of shape (batch_size, channels, height, width), produces a feature map of shape (batch_size, 1024, ceil(height/16), ceil(width/16))
    self.feature_extractor = FeatureExtractor(resnet = resnet)

    # Conversion of pooled features to head input
    self.pool_to_feature_vector = PoolToFeatureVector(resnet = resnet)

  def compute_feature_map_shape(self, image_shape):
    """
    Computes feature map shape given input image shape. Unlike VGG-16, ResNet
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