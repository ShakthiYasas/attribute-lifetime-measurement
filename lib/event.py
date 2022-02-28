# This is the event handler
subscribers = dict()

# Subscribe function to an event
def subscribe(event_type: str, fn):
    if not event_type in subscribers:
        subscribers[event_type] = []
    subscribers[event_type].append(fn)

# Publish funnction of an event
def post_event_with_params(event_type: str, data):
    if not event_type in subscribers:
        return
    for fn in subscribers[event_type]:
        fn(data)

def post_event(event_type: str):
    if not event_type in subscribers:
        return
    for fn in subscribers[event_type]:
        fn()