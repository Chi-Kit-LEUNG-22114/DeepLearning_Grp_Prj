#!/usr/bin/env python3
import json,re,unicodedata,math
from collections import defaultdict
from datetime import datetime,timezone
import pandas as pd

EMOJI=((0x1F000,0x1FAFF),(0x2600,0x27BF),(0x2300,0x23FF),(0x2B00,0x2BFF),(0xFE00,0xFE0F),(0xE0020,0xE007F))
def clean(x):
 if x is None or pd.isna(x):return None
 s=unicodedata.normalize('NFC',str(x));s=''.join(ch for ch in s if not(any(a<=ord(ch)<=b for a,b in EMOJI) or unicodedata.category(ch) in {'Cc','Cf','Cs','Co','Cn'}));s=re.sub(r'\s+',' ',s).strip();return s or None
def integer(x):return None if x is None or pd.isna(x) else int(x)
def boolean(x):return None if x is None or pd.isna(x) else (x if isinstance(x,bool) else str(x).lower() in {'true','1','yes'})
def iso(x):
 t=pd.to_datetime(x,utc=True,errors='coerce');return None if pd.isna(t) else t.isoformat().replace('+00:00','Z')
def bid(s):return re.sub(r'[^a-z0-9]+','_',unicodedata.normalize('NFKD',s).encode('ascii','ignore').decode().lower()).strip('_')
def week(x):
 t=pd.to_datetime(x,utc=True).tz_convert('Asia/Hong_Kong');st=(t-pd.Timedelta(days=t.weekday())).normalize();return st.date().isoformat(),(st+pd.Timedelta(days=6)).date().isoformat(),t.isoformat()

with open('brand_embedding_input.json',encoding='utf-8') as f:old=json.load(f)
channels=pd.read_csv('channels.csv',encoding='utf-8-sig')
videos=pd.read_excel('merged_videos.xlsx',sheet_name='videos',engine='openpyxl')
comments=pd.read_excel('merged_comments.xlsx',sheet_name='comments',engine='openpyxl')
old_by_channel={b['youtube_channel']['channel_id']:b for b in old['brands']}
comments_by_video=defaultdict(list)
for _,r in comments.iterrows():
 rec={'comment_id':clean(r.comment_id),'parent_comment_id':clean(r.parent_id),'is_reply':bool(clean(r.parent_id)),'author_display_name':clean(r.author_display_name),'text':clean(r.text),'text_available':bool(clean(r.text)),'like_count':integer(r.like_count),'reply_count':integer(r.reply_count),'published_at_utc':iso(r.published_at),'updated_at_utc':iso(r.updated_at),'retrieved_at_hkt':clean(r.retrieved_at_hkt)}
 comments_by_video[clean(r.video_id)].append(rec)
for x in comments_by_video.values():x.sort(key=lambda z:(z['published_at_utc'] or '',z['comment_id']))
video_by_brand=defaultdict(list)
for _,r in videos.iterrows():video_by_brand[clean(r.brand)].append(r)
brands=[]
for _,ch in channels.iterrows():
 name=clean(ch.brand);cid=clean(ch.channel_id);trusted=old_by_channel.get(cid);display=trusted['brand_name'] if trusted else name
 periods={}
 for r in sorted(video_by_brand.get(display,video_by_brand.get(name,[])),key=lambda z:str(z.published_at)):
  st,en,hkt=week(r.published_at);vid=clean(r.video_id);cs=comments_by_video.get(vid,[])
  vr={'video_id':vid,'video_url':clean(r.video_url),'title':clean(r.title),'title_available':bool(clean(r.title)),'published_at_utc':iso(r.published_at),'published_at_hkt':hkt,'duration_iso':clean(r.duration_iso),'duration_seconds':integer(r.duration_seconds),'metrics':{'view_count':integer(r.view_count),'like_count':integer(r.like_count),'comment_count_reported':integer(r.comment_count),'comments_in_file':len(cs)},'attributes':{'privacy_status':clean(r.privacy_status),'made_for_kids':boolean(r.made_for_kids),'category_id':clean(r.category_id),'tags_count':integer(r.tags_count)},'thumbnail':{'is_representative_shortlist_thumbnail':False,'url':None,'local_path':None},'retrieved_at_hkt':clean(r.retrieved_at_hkt),'comments':cs}
  periods.setdefault(st,{'week_start':st,'week_end':en,'videos':[]})['videos'].append(vr)
 plist=[]
 for st in sorted(periods):
  p=periods[st];vs=p['videos'];p['weekly_metrics']={'video_count':len(vs),'view_count_total':sum(v['metrics']['view_count'] or 0 for v in vs),'like_count_total':sum(v['metrics']['like_count'] or 0 for v in vs),'comment_count_reported_total':sum(v['metrics']['comment_count_reported'] or 0 for v in vs),'comments_in_file_total':sum(len(v['comments']) for v in vs)};plist.append(p)
 oldchan=trusted.get('youtube_channel',{}) if trusted else {}
 brands.append({'brand_id':bid(display),'brand_name':display,'youtube_channel':{'channel_id':cid,'handle':clean(ch.handle),'url':clean(ch.channel_url),'channel_title':clean(ch.channel_title),'custom_url':clean(ch.custom_url),'channel_role':clean(ch.channel_role),'official_status':clean(ch.official_status),'country':clean(ch.country),'published_at_utc':iso(ch.published_at),'subscriber_count':integer(ch.subscriber_count),'hidden_subscriber_count':boolean(ch.hidden_subscriber_count),'video_count':integer(ch.video_count),'channel_view_count':integer(ch.channel_view_count),'uploads_playlist_id':clean(ch.uploads_playlist_id),'retrieved_at_hkt':clean(ch.retrieved_at_hkt),'channel_thumbnail_url':oldchan.get('channel_thumbnail_url'),'channel_thumbnail_path':oldchan.get('channel_thumbnail_path')},'google_trends':trusted.get('google_trends',{'locked_query':None,'status':'not_provided'}) if trusted else {'locked_query':None,'status':'not_provided'},'source_notes':trusted.get('source_notes') if trusted else 'Added from final merged source files.','periods':plist})
brands.sort(key=lambda x:x['brand_id'])
top=[]
for b in brands:
 for p in b['periods']:
  for v in p['videos']:
   for c in v['comments']:top.append({'brand_id':b['brand_id'],'brand_name':b['brand_name'],'week_start':p['week_start'],'video_id':v['video_id'],**c})
out={'schema_version':'2.0.0','dataset_status':'final_pre_embedding_aggregate','encoding':'UTF-8','generated_at_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),'week_definition':'Monday-Sunday; timezone=Asia/Hong_Kong','text_cleaning':{'unicode_normalization':'NFC','emoji_removed':True,'format_and_control_characters_removed':True,'multilingual_letters_preserved':True},'hierarchy':'brands -> periods -> videos -> comments','source_files':['channels.csv','merged_videos.xlsx','merged_comments.xlsx'],'brands':brands,'comments':top}
with open('brand_embedding_input_final.json','w',encoding='utf-8',newline='\n') as f:json.dump(out,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
# strict assertions
by={b['brand_name']:b for b in brands};allv=[v for b in brands for p in b['periods'] for v in p['videos']];allc=[c for v in allv for c in v['comments']]
assert len(brands)==32 and {'JELLYCAT','STANLEY'}<=set(by);assert len(allv)==4102;assert len(allc)==4824;assert len(top)==4824;assert len({v['video_id'] for v in allv})==4102;assert len({c['comment_id'] for c in allc})==4824
for n in ['JELLYCAT','STANLEY']:print(n,sum(len(p['videos']) for p in by[n]['periods']),sum(len(v['comments']) for p in by[n]['periods'] for v in p['videos']))
print('TOTAL',len(brands),len(allv),len(allc))
