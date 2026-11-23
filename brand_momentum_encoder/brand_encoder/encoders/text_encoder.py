from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer
from ..preprocessing.text import chunks
class TextEncoder:
    def __init__(self, model_name:str, device:str, max_chars:int=6000, normalize:bool=True):
        self.model_name=model_name; self.max_chars=max_chars; self.normalize=normalize
        self.model=SentenceTransformer(model_name,device=device)
        self.dimension=self.model.get_sentence_embedding_dimension()
    def encode_one(self,text:str)->np.ndarray:
        parts=chunks(text,self.max_chars)
        if not parts:return np.zeros(self.dimension,dtype=np.float32)
        arr=self.model.encode(parts,convert_to_numpy=True,normalize_embeddings=self.normalize,show_progress_bar=False)
        v=arr.mean(axis=0).astype(np.float32)
        if self.normalize:
            n=np.linalg.norm(v); v=v/n if n else v
        return v
