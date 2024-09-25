from re import sub
import sys
import random

from networkx import DiGraph
sys.path.append('/home/vincent/graphrule/src')
from sklearn.preprocessing import normalize
from graph.sample import sample_graph
from graph.standard import Graph24PointI, Graph24PointII
from llm.tag import Tag
from llm.embedding import get_embedding

from graph.model import GraphClassifier

import networkx as nx
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch

import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import cosine

s1 = "2 + 2 = 4 (left: 4 9 12)"
s2 = "2 + 2 = 4 (left: 4, 9, 12)"
s3 = "2 * 2 = 4 (left: 4, 9, 12)"

vec1 = np.array(get_embedding(s1))
vec2 = np.array(get_embedding(s2))
vec3 = np.array(get_embedding(s3))

vecs = np.array([vec1, vec2, vec3])
vecs = normalize(vecs, axis=0, norm='l2')

for i in range(len(vecs)):
    for j in range(i + 1, len(vecs)):
        cos_sim = 1 - cosine(vecs[i], vecs[j])
        print(f"Cosine similarity between vecs[{i}] and vecs[{j}]: {cos_sim}")
