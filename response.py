from datetime import datetime

response_object = {
    'time_stamp': ''
}

def parse_response(dictionary):
    now = datetime.now() # current date and time
    response = response_object.copy()
    response.update(dictionary)
    response['time_stamp'] =  now.strftime("%d/%m/%Y %H:%M:%S")
    return response