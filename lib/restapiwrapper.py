import time
import requests
from lib.util import without_keys

invalid = {'time_stamp', 'session_id'}

# Wrapper class for requests library
# Makes the actual API call to the context service
class Requester:
    def get_response(self, url) -> dict:
        response = self.get_retry(url)
        return without_keys(response.json(), invalid) 
    
    def post_response(self, url) -> dict:
        response = self.post_retry(url)
        return without_keys(response.json(), invalid) 

    # Retries connecting to the context service upto 20 times
    # At 500ms intervals
    def get_retry(self, url):
        response = None
        count = 0
        while count<20:
            try:
                response = requests.get(url)
                break
            except(Exception):
                count+=1
                time.sleep(0.5)
        
        return response

    def post_retry(self, url):
        response = None
        count = 0
        while count<20:
            try:
                response = requests.post(url,{})
                break
            except(Exception):
                count+=1
                time.sleep(0.5)
        
        return response