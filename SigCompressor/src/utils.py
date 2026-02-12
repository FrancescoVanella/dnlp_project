import os
import random
import numpy as np
import torch
import nltk

def seed_everything(seed=42):
    """
    Set seed for reproducibility
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def download_nltk_resources():
    """
    Download necessary NLTK resources if not present
    """
    resources = ['stopwords', 'punkt', 'punkt_tab']
    for r in resources:
        try:
            nltk.data.find(f'tokenizers/{r}')
        except LookupError:
            nltk.download(r, quiet=True)
