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