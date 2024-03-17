# -*- coding: utf-8 -*-

import os
import logging

# set start index for BiLLM, default is 0. If set to -1, it will disable BiLLM.
BiLLM_START_INDEX = int(os.getenv('BiLLM_START_INDEX', 0))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('BiLLM')
