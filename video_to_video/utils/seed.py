# Copyright (c) Alibaba, Inc. and its affiliates.

import random
import numpy as np
import lightning as L

def setup_seed(seed):
    L.seed_everything(seed)
