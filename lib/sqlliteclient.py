import sqlite3

class SQLLiteClient:
    def __init__(self, db_name):
        self.__dbname = db_name
        self.__service_description = {
            1: ['regno'],
            2: ['regno'],
            3: ['longitude', 'latitude'],
            4: ['longitude', 'latitude']
        }

    # Check the consumer and retrieve freshness
    def get_freshness_for_consumer(self, consumerid, slaid):
        self.__conn = sqlite3.connect(self.__dbname+'.db', check_same_thread=False)
        res = self.__conn.execute(
            "SELECT *\
            FROM ContextConsumer\
            WHERE isActive=1 AND id="+str(consumerid)).fetchall()
        if(len(res)):
            freshness = self.__conn.execute(
                "SELECT freshness, price, penalty\
                FROM SLA\
                WHERE isActive=1 AND id="+str(slaid)+"\
                LIMIT 1").fetchone()
            if(freshness == None or len(freshness)==0):
                # No SLA set at the moment. So, assuming no freshness requirement. 
                return (0.5,1.0,1.0)
            else:
                # (fthr, price, penalty)
                return (freshness[0],freshness[1],freshness[2])
        else:
            # No Consumer Found.
            return None

    # Check and retrieve any context data available in the provider SLA 
    def get_provider_meta(self, providerid, attributes):
        self.__conn = sqlite3.connect(self.__dbname+'.db', check_same_thread=False)
        producer = self.__conn.execute(
                "SELECT entityId, latitude, longitude, regno \
                FROM ContextProducer \
                WHERE id="+str(providerid)+" AND isActive=1").fetchone()

        retrievable = self.__service_description[producer[0][0]]
        needed = set(retrievable) & set(attributes)
        if(needed):
            return {
                'latitude': producer[0][1],
                'longitude': producer[0][2],
                'regno': producer[0][3]
            }
        else:
            return None

    # Retrieve context producers for an entity
    def get_providers_for_entity(self,entityid):
        self.__conn = sqlite3.connect(self.__dbname+'.db', check_same_thread=False)
        producers = self.__conn.execute(
                "SELECT id \
                FROM ContextProducer \
                WHERE entityId="+str(entityid)+" AND isActive=1").fetchall()
        return list(map(lambda x: x[0], producers))

    # Retrieve the URLs of the providers given ids
    def get_context_producers_by_ids(self, providers) -> dict:
        self.__conn = sqlite3.connect(self.__dbname+'.db', check_same_thread=False)
        output = {}
        producers = self.__conn.execute(
                "SELECT id, url \
                FROM ContextProducer \
                WHERE isActive=1 AND id IN "+str(providers)).fetchall()
        
        for prod in producers:
            output[prod[1]] = {'url':prod[1]}
        
        return output

    # Retrieve the URLs of the matching providers and lifetimes of the attributes
    def get_context_producers(self, entityid, attributes) -> dict:
        self.__conn = sqlite3.connect(self.__dbname+'.db', check_same_thread=False)
        output = {}
        if(len(attributes)>0):
            context_in_desc = self.__service_description[entityid] if entityid in self.__service_description else None          
            producers = self.__conn.execute(
                "SELECT id, url, price, samplingrate \
                FROM ContextProducer \
                WHERE entityId="+str(entityid)+" AND isActive=1").fetchall()
            
            if(context_in_desc):
                attributes = set(attributes) - set(context_in_desc)

            if(len(producers)>0):
                att_string = "name='"+attributes[0]+"'"
                if(len(attributes)>1):
                    for idx in range(1,len(attributes)):
                        att_string = att_string+" OR name='"+attributes[idx]+"'"

                for prod in  producers:
                    sampling_interval = 1/prod[3]
                    att_res = self.__conn.execute(
                        "SELECT name, lifetime \
                        FROM ContextAttribute \
                        WHERE producerId="+str(prod[0])+" AND ("+att_string+")").fetchall()
                    if(len(att_res)==len(attributes)):
                        lts = {}
                        for att in att_res:
                            lts[att[0]] = -1 if att[1] == -1 else max(sampling_interval,att[1])
                        if(context_in_desc):
                            for i in range(0,len(context_in_desc)):
                                 lts[context_in_desc[i]] = -1
                        
                        output[prod[0]] = {
                            'url': prod[1],
                            'lifetimes': lts,
                            'cost': prod[2]
                        }

        return output
    
    def add_cached_life(self, entityid, attribute, lifetime):
        self.__conn = sqlite3.connect(self.__dbname+'.db', check_same_thread=False)
        cursor=self.__conn.cursor()
        statement = "INSERT INTO CachedLifetime (entityid, attribute, lifetime, cached) VALUES\
            ("+str(entityid)+",'"+str(attribute)+"','"+str(lifetime)+"', datetime('now'))"
        self.__conn.execute(statement)
        return cursor.lastrowid

    def update_cached_life(self, entityid, attribute, lifetime):
        self.__conn = sqlite3.connect(self.__dbname+'.db', check_same_thread=False)
        self.__conn.execute(
            "UPDATE CachedLifetime SET lifetime='"+str(lifetime)+"'\
            entityid="+str(entityid)+" AND attribute='"+attribute+"')")

    def remove_cached_life(self, entityid, attribute):
        self.__conn = sqlite3.connect(self.__dbname+'.db', check_same_thread=False)
        self.__conn.execute(
            "DELETE FROM CachedLifetime WHERE\
            entityid="+str(entityid)+" AND attribute='"+attribute+"')")
    
    def get_cached_life(self, entityid, attribute):
        self.__conn = sqlite3.connect(self.__dbname+'.db', check_same_thread=False)
        res = self.__conn.execute(
            "SELECT lifetime, cached FROM CachedLifetime\
            WHERE entityid="+str(entityid)+" AND attribute='"+attribute+"'\
            ORDER BY Id DESC\
            LIMIT 1)").fetchone()
        return res[0]

    def get_expired_cached_lifetimes(self):
        self.__conn = sqlite3.connect(self.__dbname+'.db', check_same_thread=False)
        res = self.__conn.execute(
            "SELECT entityid, attribute FROM CachedLifetime\
            WHERE lifetime<=datetime('now')").fetchall()
        return res[0]

    # Current retrieval latency
    def update_ret_latency(self, latency):
        self.__conn = sqlite3.connect(self.__dbname+'.db', check_same_thread=False)
        self.__conn.execute(
            "UPDATE CurrentRetrievalLatency SET latency="+str(latency)+"\
            Id=1)")
    
    def get_ret_latency(self):
        self.__conn = sqlite3.connect(self.__dbname+'.db', check_same_thread=False)
        res = self.__conn.execute(
            "SELECT latency cached FROM CurrentRetrievalLatency\
            LIMIT 1)").fetchone()
        return res[0]


    # Initialize the SQLite instance 
    def seed_db_at_start(self):
        try:
            self.__create_tables()
            self.__seed_tables_with_data()
        except(Exception):
            print('An error occured when seeding to database')

    def __seed_tables_with_data(self):
        self.__conn = sqlite3.connect(self.__dbname+'.db', check_same_thread=False)
        self.__conn.execute(
            "INSERT INTO Entity (id,name) VALUES\
            (1,'Car'),(2,'Bike'),(3,'CarPark'), (4,'Weather')")
       
        self.__conn.execute(
            "INSERT INTO SLA (id,freshness,price,penalty,rtmax,isActive) VALUES\
            (1,0.9,1.2,2.0,0.5,1),(2,0.8,1.0,2.0,0.6,1),(3,0.7,0.8,1.8,0.8,1),(4,0.6,0.75,1.25,1.0,0)")
       
        self.__conn.execute(
            "INSERT INTO ContextConsumer (id,name,isActive) VALUES\
            (100, 'Shakthi', 1),(102, 'Alexey', 1),(104, 'Amin', 1),(106, 'Himadri', 0)")

        self.__conn.execute(
            "INSERT INTO ConsumerSLA VALUES\
            (100, 2),(100, 3),(102, 3),(104, 2),(106, 1)")
        
        self.__conn.execute(
            "INSERT INTO ContextService VALUES\
            (1,'Carpark Tracker',1),(2,'Vehicle Tracker',1),(3,'Weather Monitor',1)")
        
        self.__conn.execute(
            "INSERT INTO ContextProducer VALUES\
                (1,2,1,'http://localhost:5000/bikes',0.6, 1, NULL, NULL, 'CX123'),\
                (3,1,1,'http://localhost:5000/cars?id=3',0.25, 0.5, NULL, NULL, '1HR800'),\
                (4,1,1,'http://localhost:5000/cars?id=4',0.4, 1, NULL, NULL, '1VC546'),\
                (5,1,1,'http://localhost:5000/cars?id=5',0.3, 0.2, NULL, NULL, '1DH8906'),\
                (6,1,1,'http://localhost:5000/cars?id=6',0.2, 4, NULL, NULL, '1KP1244'),\
                (8,3,1,'http://localhost:5000/carparks?id=8',0.4, 0.017, -37.84938300336436, 145.11336178206872, NULL),\
                (9,3,1,'http://localhost:5000/carparks?id=9',0.75, 0.033, -37.84586713387071', 145.1149120988647, NULL),\
                (10,3,1,'http://localhost:5000/carparks?id=10',0.3, 0.017, -37.84621449228698', 145.11596352479353, NULL),\
                (11,4,1,'http://localhost:5000/weather?id=1',0.2, 0.017, -37.848027507269634, 145.1155451001933, NULL),")
        
        self.__conn.execute(
            # Assume that each car park has a varying parking cost (i.e. peak and off-peak price)
            "INSERT INTO ContextAttribute VALUES\
                (1,'speed',1,0,'kmph'),"+      
                "(3,'speed',3,0,'kmph'),\
                (4,'height',3,-1,'m'),\
                (5,'capacity',3,-1,'number'),\
                (6,'model',3,-1,'text'),\
                (7,'speed',4,0,'kmph'),\
                (8,'height',4,-1,'m'),\
                (9,'capacity',4,-1,'number'),\
                (10,'speed',5,0,'kmph'),\
                (11,'height',5,-1,'m'),\
                (12,'capacity',5,-1,'number'),\
                (13,'model',5,-1,'text'),\
                (14,'speed',6,0,'kmph'),\
                (15,'height',6,-1,'m'),"+            
                "(16,'maxheight',8,-1,'m'),\
                (17,'availability',8,30,'s'),\
                (18,'price',8,-1,'/hr'),\
                (19,'totalslots',8,-1,'number'),\
                (20,'availability',9,60,'s'),\
                (21,'price',9,-1,'/hr'),\
                (22,'maxheight',10,-1,'m'),\
                (23,'availability',10,150,'s'),\
                (24,'price',10,-1,'/hr'),\
                (25,'temperature',11,1800,'C'),\
                (26,'windspeed',10,10,'kmph'),\
                (27,'winddiretion',10,10,'degrees'),\
                (28,'humidity',10,900,'percentage')")
        
        self.__conn.execute(
            "INSERT INTO ContextServiceProducer VALUES\
                (2,1),(2,3),(2,4),(1,6),(1,7),(3,11)")
        
        self.__conn.execute(
            "INSERT INTO CurrentRetrievalLatency VALUES\
                (1,0)")
        
        self.__conn.commit()

    def __create_tables(self):
        self.__conn = sqlite3.connect(self.__dbname+'.db', check_same_thread=False)
        self.__conn.execute(
            '''CREATE TABLE Entity
            (
                id INT PRIMARY KEY NOT NULL,
                name TEXT NOT NULL
            );''')

        self.__conn.execute(
            '''CREATE TABLE ContextService
            (
                id INT PRIMARY KEY NOT NULL,
                name TEXT NOT NULL,
                isActive BOOLEAN NOT NULL
            );''')

        self.__conn.execute(
            '''CREATE TABLE ContextProducer
            (
                id INT PRIMARY KEY NOT NULL,
                entityId INT NOT NULL,
                isActive BOOLEAN NOT NULL,
                url TEXT NOT NULL,
                price REAL NOT NULL,
                samplingrate REAL NOT NULL,
                latitude REAL NULL,
                longitude REAL NULL,
                regno TEXT NULL,
                FOREIGN KEY (entityId) REFERENCES Entity(id)
            );''')

        self.__conn.execute(
            '''CREATE TABLE ContextServiceProducer
            (
                serviceId INT NOT NULL,
                producerId INT NOT NULL,
                PRIMARY KEY (serviceId, producerId),
                FOREIGN KEY (serviceId) REFERENCES ContextService(id)
                FOREIGN KEY (producerId) REFERENCES ContextProducer(id)
            );''')

        self.__conn.execute(
            '''CREATE TABLE ContextAttribute
            (
                id INT PRIMARY KEY NOT NULL,
                name TEXT NOT NULL,
                producerId INT NOT NULL,
                lifetime INT NOT NULL,
                unit TEXT NOT NULL,
                FOREIGN KEY (producerId) REFERENCES ContextProducer(id)
            );''')

        self.__conn.execute(
            '''CREATE TABLE ContextConsumer
            (
                id INT PRIMARY KEY NOT NULL,
                name TEXT NOT NULL,
                isActive BOOLEAN NOT NULL
            );''')

        self.__conn.execute(
            '''CREATE TABLE SLA
            (
                id INT PRIMARY KEY NOT NULL,
                freshness REAL NOT NULL,
                price REAL NOT NULL,
                penalty REAL NOT NULL,
                rtmax REAL NOT NULL,
                isActive BOOLEAN NOT NULL
            );''')

        self.__conn.execute(
            '''CREATE TABLE ConsumerSLA
            (
                consumerId INT NOT NULL,
                slaId INT NOT NULL,
                PRIMARY KEY (consumerId, slaId),
                FOREIGN KEY (consumerId) REFERENCES ContextConsumer(id)
                FOREIGN KEY (slaId) REFERENCES SLA(id)
            );''')
        
        self.__conn.execute(
            '''CREATE TABLE CachedLifetime
            (
                Id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                entityid INT NOT NULL,
                attribute TEXT NOT NULL,
                lifetime DATETIME NOT NULL,
                cached DATETIME NOT NULL
            );''')
        
        self.__conn.execute(
            '''CREATE TABLE CurrentRetrievalLatency
            (
                Id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                latency REAL NOT NULL
            );''')
        
        self.__conn.commit()