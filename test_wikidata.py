import requests

batch = ["Q18747", "P31", "rdfs:label"]
url = "https://www.wikidata.org/w/api.php"
params = {
    "action": "wbgetentities",
    "ids": "|".join(batch),
    "languages": "en",
    "props": "labels",
    "format": "json"
}
resp = requests.get(url, params=params).json()
print(resp)
