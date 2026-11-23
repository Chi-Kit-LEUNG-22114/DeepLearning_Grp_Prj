from __future__ import annotations
import numpy as np, torch
from transformers import CLIPModel, CLIPProcessor
from ..preprocessing.image import load_rgb
class ClipEncoder:
    """CLIP image and matching text tower. Only these outputs share a cosine space."""
    def __init__(self,model_name:str,device:str):
        self.model_name=model_name; self.device=device
        self.model=CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor=CLIPProcessor.from_pretrained(model_name)
        self.dimension=int(self.model.config.projection_dim)
    def image(self,path)->np.ndarray:
        im=load_rgb(path); batch=self.processor(images=im,return_tensors="pt")
        batch={k:v.to(self.device) for k,v in batch.items()}
        with torch.no_grad(): x=self.model.get_image_features(**batch)
        x=torch.nn.functional.normalize(x,dim=-1)
        return x[0].cpu().numpy().astype(np.float32)
    def text(self,text:str)->np.ndarray:
        if not text:return np.zeros(self.dimension,dtype=np.float32)
        batch=self.processor(text=[text],return_tensors="pt",padding=True,truncation=True)
        batch={k:v.to(self.device) for k,v in batch.items()}
        with torch.no_grad(): x=self.model.get_text_features(**batch)
        x=torch.nn.functional.normalize(x,dim=-1)
        return x[0].cpu().numpy().astype(np.float32)
