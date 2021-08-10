import pymongo

class MongoClient:
  db = None

  def __init__(self, connection: str, database: str):
    myclient = pymongo.MongoClient(connection)
    self.db = myclient[database]
  
  def read_last(self, collection, sorting_col = '_id'):
    col = self.db[collection]
    return col.find().sort(sorting_col,-1).limit(1)

  def read_all(self, collection, condition, sorting_col = '_id'):
    col = self.db[collection]
    return col.find(condition).sort(sorting_col,-1)
    
  def insert_one(self, collection, data):
    col = self.db[collection]
    try:
      x = col.insert_one(data)
      return x.inserted_id
    except(Exception):
      print('Insertion into mongo db failed!')
      return -1

  def insert_many(self, collection, data):
    col = self.db[collection]
    try:
      x = col.insert_many(data)
      return x.inserted_ids
    except(Exception):
      print('Insertion into mongo db failed!')
      return -1