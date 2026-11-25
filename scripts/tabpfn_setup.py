"""One-time TabPFN setup: triggers Prior Labs sign-in / license acceptance and
downloads the model weights, then checks that a small fit works.

    python followup/scripts/tabpfn_setup.py
"""
import numpy as np
from tabpfn import TabPFNClassifier

X = np.random.default_rng(0).random((50, 3))
TabPFNClassifier(device="cpu").fit(X, np.arange(50) % 2)
print("TabPFN ready")
