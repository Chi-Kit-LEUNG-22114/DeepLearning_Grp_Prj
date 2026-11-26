from __future__ import annotations
import numpy as np, torch
from transformers import CLIPModel, CLIPProcessor
from ..preprocessing.image import load_rgb
class ClipEncoder:
    """CLIP image and matching text tower. Only these outputs share a cosine space."""
    def __init__(self,model_name:str,device:str,offline:bool=False):
        self.model_name=model_name; self.device=device
        self.model=CLIPModel.from_pretrained(model_name,local_files_only=offline).to(device).eval()
        self.processor=CLIPProcessor.from_pretrained(model_name,local_files_only=offline)
        self.dimension=int(self.model.config.projection_dim)
    def image(self,path)->np.ndarray:
        im=load_rgb(path); batch=self.processor(images=im,return_tensors="pt")
        batch={k:v.to(self.device) for k,v in batch.items()}
        with torch.no_grad(): x=self.model.get_image_features(**batch)
        x=torch.nn.functional.normalize(x,dim=-1)
        return x[0].cpu().numpy().astype(np.float32)
    def text(self,text:str)->np.ndarray:
        return self.text_many([text])[0]
    def text_many(self,texts,batch_size:int=16)->list[np.ndarray]:
        result=[np.zeros(self.dimension,dtype=np.float32) for _ in texts]
        indices=[i for i,text in enumerate(texts) if text]
        for start in range(0,len(indices),batch_size):
            current=indices[start:start+batch_size]
            batch=self.processor(text=[texts[i] for i in current],return_tensors="pt",padding=True,truncation=True)
            batch={k:v.to(self.device) for k,v in batch.items()}
            with torch.no_grad(): x=self.model.get_text_features(**batch)
            x=torch.nn.functional.normalize(x,dim=-1)
            for i,vector in zip(current,x.cpu().numpy()):
                result[i]=vector.astype(np.float32)
        return result
