import json
import requests

url = "url"
data = {"data": [[4.3, 1.3, 5.0, 2.2]]}
body = str.encode(json.dumps(data))
api_key = 'api-key'
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key), 'azureml-model-deployment': 'iris-model-1' }
r = requests.post(url, data=json.dumps(data), headers=headers)
# print(r.text)
print("Predicted class label:", r.json()["data"][0])