from __future__ import annotations
import html, re, unicodedata
from ..common import normalize_missing_text
URL=re.compile(r"https?://\S+|www\.\S+",re.I)
USER=re.compile(r"(?<!\w)@[A-Za-z0-9_.]+")
SPACE=re.compile(r"\s+")
def clean_text(value)->str:
    s=normalize_missing_text(value)
    if not s:return ""
    s=unicodedata.normalize("NFKC",html.unescape(s))
    s=URL.sub("[URL]",s); s=USER.sub("[USER]",s)
    return SPACE.sub(" ",s).strip()
def chunks(text:str,max_chars:int):
    text=clean_text(text)
    if not text:return []
    return [text[i:i+max_chars] for i in range(0,len(text),max_chars)]
