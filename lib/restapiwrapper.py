import time
import requests
from lib.util import without_keys

invalid = {'time_stamp', 'session_id'}

# Wrapper class for requests library
# Makes the actual API call to the context service
class Requester:
    def get_response(self, url) -> dict:
        response = self.retry(url)
        return without_keys(response.json(), invalid) 

    # Retries connecting to the context service upto 20 times
    # At 500ms intervals
    def retry(self, url):
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

# Simple Context Service Resolution
class ServiceSelector:
    def __init__(self, db):
        self.db_instance = db
        self.requester = Requester()

    def get_response_for_entity(self, entityid):
        rec = self.db_instance.read_single('service-registry',{'entity':entityid})
        return self.requester.get_response(rec['url'])