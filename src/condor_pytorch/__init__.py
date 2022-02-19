from .version import __version__

from .activations import ordinal_softmax
from .dataset import label_to_levels
from .dataset import levels_from_labelbatch
from .dataset import proba_to_label
from .dataset import logits_to_label
from .losses import CondorOrdinalCrossEntropy
from .losses import condor_negloglikeloss
from .metrics import earth_movers_distance
from .metrics import mean_absolute_error
from .metrics import ordinal_accuracy
#from .labelencoder import CondorOrdinalEncoder

# if somebody does "from somepackage import *", this is what they will
# be able to access:
__all__ = [
  'ordinal_softmax',
  'label_to_levels',
  'levels_from_labelbatch',
  'proba_to_label',
  'logits_to_label',
  'CondorOrdinalCrossEntropy',
  'condor_negloglikeloss',
  'mean_absolute_error',
  'ordinal_accuracy',
#  'CondorOrdinalEncoder',
]
