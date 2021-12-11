from datetime import datetime

response_object = {
    'time_stamp': ''
}

# Produces the generic response from the API
def parse_response(dictionary, session = None, meta = None):
    now = datetime.now() # current date and time
    response = response_object.copy()
    response.update(dictionary)
    response['time_stamp'] =  now.strftime("%Y-%m-%d %H:%M:%S")
    if(session != None):
        response['session_id'] = session
    if(meta != None):
        response['meta'] = meta        
    
    return response