import base64
import os

import requests

os.environ['no_proxy'] = '*'
# Base64でエンコードする画像のパス
target_file = "sample.jpg"


with open(target_file, 'rb') as f:
    data = f.read()

# Base64で画像をエンコード
encoded = base64.b64encode(data).decode('utf-8')

url = 'http://localhost:5000'
payload = {'image': encoded}

res = requests.post(url, json=payload)
print(res.text)