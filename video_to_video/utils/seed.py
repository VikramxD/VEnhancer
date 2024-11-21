# Copyright (c) Alibaba, Inc. and its affiliates.

import random
import numpy as np
import Lightning as L

def setup_seed(seed):
    L.seed_everything(seed)
