import requests
import numpy as np 
import os
from dotenv import load_dotenv
load_dotenv()
API_kEY= os.getenv('EURI_AI_API_KEY')
def embedings_model(text,model='text-embedding-3-small'):
    url ="https://api.euron.one/api/v1/euri/embeddings"
    headers= {
        "Authorization":f"Bearer {API_kEY}",
        "Content-Type":"application/json"

    }
    pyload = {
        "input":text,
        "model":model
    }

    response = requests.post(url,headers=headers,json=pyload)
    data = response.json()
    # Now access the embedding
    # if 'data' not in data:
    #     raise KeyError(f"'data' key missing. Full response: {data}")
    return np.array(data["data"][0]["embedding"])
   