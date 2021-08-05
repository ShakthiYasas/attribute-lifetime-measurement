import json
import requests
from util import without_keys

invalid = {'time_stamp'}

class Requester:
    def get_response(self, url) -> dict:
        response = requests.get(url)
        json_obj = json.loads(response.json())
        return without_keys(json_obj, invalid) 