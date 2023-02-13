import requests

url = "https://text-analysis12.p.rapidapi.com/sentiment-analysis/api/v1.1"
payload = {
    "language": "english",
    "text": "Falcon 9’s first stage has landed on the Of Course I Still Love You droneship – "+
    "the 9th landing of this booster"
}
headers = {
	"content-type": "application/json",
	"X-RapidAPI-Key": "117cd9915amsh12494a2bdb16d7ep13b3fdjsnd8861fc6f602",
	"X-RapidAPI-Host": "text-analysis12.p.rapidapi.com"
}
response = requests.request("POST", url, json=payload, headers=headers)
print(response.text)

