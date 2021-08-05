import json
import requests

invalid = {'time_stamp'}

class Requester:
    def get_response(self, url) -> dict:
        response = requests.get(url)
        json_obj = json.loads(response.json())
        return self.without_keys(json_obj, invalid) 

    def without_keys(self, d, keys):
        return {x: d[x] for x in d if x not in keys}