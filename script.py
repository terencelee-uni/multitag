import requests
import json

resp = requests.post("http://localhost:5000/predict",
                     files={"file": open('./image.jpeg','rb')})
print(resp)
data = resp.json()
data = json.dumps(data)
print(type(data))
jsonFile = open("data.json", "w")
jsonFile.write(data)
jsonFile.close()
