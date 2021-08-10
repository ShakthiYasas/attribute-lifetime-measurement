import requests
from lib.util import without_keys

invalid = {'time_stamp', 'session_id'}

class Requester:
    def get_response(self, url) -> dict:
        response = requests.get(url)
        return without_keys(response.json(), invalid) 