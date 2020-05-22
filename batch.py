from pycarol import Carol, Staging, ApiKeyAuth, Storage, PwdAuth
from sentence_transformers import SentenceTransformer
import pandas as pd
import pickle
import numpy as np
import re

def remove_html_tags(text):
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)
    return " ".join(re.sub(r'\s([?.!"](?:\s|$))', r'\1', text).strip().split())

def get_question(body):   
    body = body.replace('\n', '').replace('<br>', '')

    m = re.search('(?<=<strong>D(ú|u)vida).*?(?=<strong>Ambiente)', body, re.IGNORECASE)
    if m:
        return remove_html_tags(m.group(0))
        
    m = re.search('(?<=<strong>Ocorr(ê|e)ncia).*?(?=<strong>Ambiente)', body, re.IGNORECASE)
    if m:
        return remove_html_tags(m.group(0))
    
    return np.nan

def get_question_type(body):   
    body = body.replace('\n', '').replace('<br>', '')

    m = re.search('(?<=<strong>D(ú|u)vida).*?(?=<strong>Ambiente)', body, re.IGNORECASE)
    if m:
        return 'question'

    m = re.search('(?<=<strong>Ocorr(ê|e)ncia).*?(?=<strong>Ambiente)', body, re.IGNORECASE)
    if m:
        return 'occurrence'
    
    return np.nan

def get_environment(body):   
    body = body.replace('\n', '').replace('<br>', '')

    m = re.search('(?=<strong>Ambiente).*?(?=<strong>Solu(ç|c)(ã|a)o)', body, re.IGNORECASE)
    if not m:
        return np.nan
    return remove_html_tags(m.group(0))

def get_solution(body):    
    body = body.replace('\n', '').replace('<br>', '')

    m = re.search('(?<=<strong>Solu(ç|c)(ã|a)o)(?s)(.*$)', body, re.IGNORECASE)
    if not m:
        return np.nan
    return m.group(0)

def get_sanitized_solution(body):    
    body = body.replace('\n', '').replace('<br>', '')

    m = re.search('(?<=<strong>Solu(ç|c)(ã|a)o)(?s)(.*$)', body, re.IGNORECASE)
    if not m:
        return np.nan
    return remove_html_tags(m.group(0))

def save_object_to_storage(obj, filename):
    with open(filename, "bw") as f:
        pickle.dump(obj, f)
    stg.save(filename, obj, format='pickle')

# Carol instances

origin = Carol(domain='monitoriaqa',
app_name=' ',
organization='totvs',
auth=PwdAuth('fmendes@totvs.com.br', 'Intxdx70248821*'))

target = Carol()

staging = Staging(target)
stg = Storage(target)

df = Staging(origin).fetch_parquet(staging_name='articles',
                        connector_name='carol_connect_zerado',
                        cds=True
                        )

df = df.astype(str)

df = df.replace('', np.nan)

df = df[(df.source_locale == 'pt-br') & (~df.body.isnull())]

df['question'] = df.body.apply(get_question)
df['question_type'] = df.body.apply(get_question_type)
df['environment'] = df.body.apply(get_environment)
df['solution'] = df.body.apply(get_solution)
df['sanitized_solution'] = df.body.apply(get_sanitized_solution)

df = df.replace('', np.nan)

df = df.dropna(subset=['question', 'question_type', 'environment', 'solution'])

df = df.reset_index(drop=True)

df['id'] = df['id'].apply(lambda x: x.replace('.0', ''))

staging.send_data(staging_name='articles', connector_id='175e781f1cce470eb0457323430f1407', data=df,)
                             #crosswalk_auto_create=['idinternalmdb'], auto_create_schema=True)


# Load Sentence model
model = SentenceTransformer('distiluse-base-multilingual-cased')

# Embed a list of sentences
sentence_embeddings = model.encode(df.question.values)

id_list = df['id'].values

index_to_question_id_mapping = dict(enumerate(id_list))
question_id_to_index_mapping = {j: i for i,j in index_to_question_id_mapping.items()} 

# Save objects in Storage

save_object_to_storage(sentence_embeddings, 'sentence_embeddings')

save_object_to_storage(index_to_question_id_mapping, 'index_to_question_id_mapping')

save_object_to_storage(question_id_to_index_mapping, 'question_id_to_index_mapping')