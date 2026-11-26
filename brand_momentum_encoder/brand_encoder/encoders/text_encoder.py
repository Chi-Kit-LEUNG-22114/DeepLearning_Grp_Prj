from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer
from ..preprocessing.text import chunks
class TextEncoder:
    def __init__(self, model_name:str, device:str, max_chars:int=6000, normalize:bool=True, offline:bool=False):
        self.model_name=model_name; self.max_chars=max_chars; self.normalize=normalize
        self.model=SentenceTransformer(model_name,device=device,local_files_only=offline)
        self.dimension=int(self.model.get_embedding_dimension() if hasattr(self.model,"get_embedding_dimension") else self.model.get_sentence_embedding_dimension())
    def encode_one(self,text:str)->np.ndarray:
        return self.encode_many([text])[0]
    def encode_many(self,texts,batch_size:int=16)->list[np.ndarray]:
        all_chunks=[];owners=[]
        for i,text in enumerate(texts):
            parts=chunks(text or "",self.max_chars)
            all_chunks.extend(parts);owners.extend([i]*len(parts))
        result=[np.zeros(self.dimension,dtype=np.float32) for _ in texts]
        if not all_chunks:return result
        arr=self.model.encode(all_chunks,batch_size=batch_size,convert_to_numpy=True,
                              normalize_embeddings=self.normalize,show_progress_bar=False)
        by_owner=[[] for _ in texts]
        for owner,vector in zip(owners,arr):by_owner[owner].append(vector)
        for i,parts in enumerate(by_owner):
            if parts:
                vector=np.mean(parts,axis=0).astype(np.float32)
                if self.normalize:
                    norm=np.linalg.norm(vector);vector=vector/norm if norm else vector
                result[i]=vector
        return result
