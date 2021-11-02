import pymongo

class MongoClient:
  __db = None

  def __init__(self, connection: str, database: str):
    myclient = pymongo.MongoClient(connection)
    self.__db = myclient[database]
  
  # Retrieve the first record of the collection by a condition
  def read_single(self, collection, condition, sorting_col = '_id'):
    col = self.__db[collection]
    return col.find(condition).sort(sorting_col).limit(1)

  # Retrieve the last record of the collection
  def read_last(self, collection, sorting_col = '_id'):
    col = self.__db[collection]
    return col.find().sort(sorting_col,-1).limit(1)

  # Retrieve all the items in the collection
  def read_all(self, collection, condition, sorting_col = '_id'):
    col = self.__db[collection]
    return col.find(condition).sort(sorting_col,-1)

  # Retrieve all the items in the collection
  def read_all_with_limit(self, collection, condition, limit, sorting_col = '_id'):
    col = self.__db[collection]
    return col.find(condition).sort(sorting_col,-1).limit(limit)
    
  # Insert a single new item to the collection
  # Returns: _id of the inserted item
  def insert_one(self, collection, data):
    col = self.__db[collection]
    try:
      x = col.insert_one(data)
      return x.inserted_id
    except(Exception):
      print('Insertion into mongo db failed!')
      return -1

  # Insert multiple items to the collection
  # Returns: list of _ids of the inserted
  def insert_many(self, collection, data):
    col = self.__db[collection]
    try:
      x = col.insert_many(data)
      return x.inserted_ids
    except(Exception):
      print('Insertion into mongo db failed!')
      return -1