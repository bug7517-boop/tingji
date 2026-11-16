"""Bound BLAS threads before importing audio/model libraries on desktop PCs."""
import os

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
