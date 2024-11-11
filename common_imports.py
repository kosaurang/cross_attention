# common_imports.py
import pandas as pd
import torch
import os
from sklearn.model_selection import train_test_split
import torch.nn as nn
from PIL import Image
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import copy
import math
import random
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

#model
from tqdm import tqdm
from transformers import MBartPreTrainedModel, MBartConfig, AutoModel, AutoConfig, AutoTokenizer, BeitImageProcessor

from transformers.modeling_outputs  import (
    ModelOutput,
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from torch.nn import CrossEntropyLoss
from torch import optim
from transformers import AutoModel
from torch.utils.checkpoint import checkpoint
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
#metric
import re
import unicodedata
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
from collections import defaultdict
