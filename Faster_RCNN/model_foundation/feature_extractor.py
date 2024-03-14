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