import numpy as np
import spacy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def get_sentence_embedding(corpus,word_embeddings,embedding_dim=300):
    """Create sentence embedding by averaging word embeddings

    Args:
        corpus (_type_): _description_
        word_embeddings (_type_): _description_
        embedding_dim (int, optional): _description_. Defaults to 300.
    """