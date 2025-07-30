import requests

TOGETHER_API_KEY = "6390ff6e47b4b385bd9cfc29586c5787d0cff150716814b1c534c5a349819b94"

url = "https://api.together.xyz/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {TOGETHER_API_KEY}",
    "Content-Type": "application/json"
}

data = {
    "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "messages": [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "what are the requriments of cmmc level 3?"}
    ],
    "temperature": 0.7,
    "max_tokens": 300
}

response = requests.post(url, headers=headers, json=data)

if response.status_code == 200:
    reply = response.json()["choices"][0]["message"]["content"]
    print("Mixtral says:\n", reply)
else:
    print("Error:", response.status_code, response.text)
