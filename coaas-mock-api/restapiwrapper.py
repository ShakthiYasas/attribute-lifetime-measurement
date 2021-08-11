import time
import requests
from lib.util import without_keys

invalid = {'time_stamp', 'session_id'}

class Requester:
    def get_response(self, url) -> dict:
        response = self.retry(url)
        return without_keys(response.json(), invalid) 

    def retry(self, url):
        response = None
        while True:
            try:
                response = requests.get(url)
                break
            except(Exception):
                print('retrying')
                time.sleep(0.5)
        
        return response