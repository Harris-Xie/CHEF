import nltk

from django.views.decorators.csrf import csrf_exempt
import json
from django.http import HttpResponse
import fasttext
import pandas as pd
import os
import faiss
import re
import numpy as np
import openai
import time
from tqdm import tqdm

os.environ['CHATGPT_API_KEY'] = ''
oai_key = os.environ.get("CHATGPT_API_KEY")
openai.api_type = "azure"
openai.api_base = "https://cloudgpt.openai.azure.com/"
openai.api_key = oai_key
openai.api_version = "2023-03-15-preview"
chatgpt_model_name= "gpt-4-32k-20230321"


test_data =pd.read_json('../data/test.json')
y_true=[]
if not os.path.exists('y_pred_gpt4.json'):
    y_pred=[]
else:
    with open('y_pred_gpt4.json','r') as f:
        y_pred=json.load(f)
false_case=[]
for i in tqdm(range(0,len(test_data))):
    prompt="""Context:"""
    prompt+="""Now we have a Claim and several pieces of evidence that have been retrieved. You should make a choice in {Supprted, Refuted, Can't Judge}. Please provide the choice first in your response like "Supported,...".\n"""
    prompt+="""Claim:"""+test_data['claim'][i]+"\n"
    prompt+="""Evidence:\n"""
    j=1
    for key,value in test_data['gold evidence'][i].items():
        if value['text']=='':
            break
        prompt+=str(j)+':'+value['text']+'\n'
        j+=1
    prompt+='Judgement:'
    retry_count = 0
    max_retries = 6
    flag = False
    while True:
        try:
            print('front')
            response = openai.ChatCompletion.create(
                engine=chatgpt_model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            break
        except Exception as e:
            print(e)
            retry_count += 1
            print("{}th retry".format(retry_count))
            time.sleep(4)
            if retry_count > max_retries:
                raise e
    if i==334:
        y_pred.append(1)
        with open('y_pred_gpt4.json', 'w') as f:
            json.dump(y_pred, f)
        continue
    text = response['choices'][0]['message']['content']
    print('done')
    if text.lower().startswith("supported"):
        y_pred.append(0)
    elif text.lower().startswith("refuted"):
        y_pred.append(1)
    elif text.lower().startswith("can't"):
        y_pred.append(2)
    else:
        y_pred.append(0)
        false_case.append(i)
    with open('y_pred_gpt4.json','w') as f:
        json.dump(y_pred,f)
    print(len(false_case))
    time.sleep(1)
for i in range(len(test_data)):
    y_true.append(test_data['label'][i])
with open('false_case.json','w') as f:
    json.dump(false_case,f)




