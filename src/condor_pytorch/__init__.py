# Garrett Jenkinson 2021
# orca_pytorch
# Author: Garrett Jenkinson <github.com/GarrettJenkinson>
#
# License: MIT

from .version import __version__

from .dataset import label_to_levels
from .dataset import levels_from_labelbatch
from .dataset import proba_to_label
from .dataset import logits_to_label
from .losses import CondorOrdinalCrossEntropy
#from .metrics import MeanAbsoluteErrVorLabels
#from .activations import ordinal_softmax
#from .labelencoder import CondorOrdinalEncoder

# if somebody does "from somepackage import *", this is what they will
# be able to access:
__all__ = [
  'label_to_levels',
  'levels_from_labelbatch',
  'proba_to_label',
  'logits_to_label',
  'CondorOrdinalCrossEntropy',
#  'MeanAbsoluteErrorLabels',
#  'ordinal_softmax',
#  'CondorOrdinalEncoder',
]
