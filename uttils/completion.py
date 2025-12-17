import requests
import os 
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("EURI_AI_API_KEY")
def generate_completion(prompt,model="gpt-4.1-nano"):
    url = "https://api.euron.one/api/v1/euri/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "model": model,
        "max_tokens": 50,
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=payload)
    return  response.json()['choices'][0]['message']['content']
    # print(data)
