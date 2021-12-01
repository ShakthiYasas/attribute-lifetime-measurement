import re
import sqlite3
import datetime

REGEX = r"\..+"
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

class SQLLiteClient:
    def __init__(self, db_name):
        self.__dbname = db_name
        self.__service_description = {
            1: ['regno'],
            2: ['regno'],
            3: ['location', 'address'],
            4: ['location'],
            5: ['location'],
            6: ['location'],
            7: ['location', 'address'],
            8: ['location', 'address'],
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
                "SELECT freshness, price, penalty, rtmax\
                FROM SLA\
                WHERE isActive=1 AND id="+str(slaid)+"\
                LIMIT 1").fetchone()
            if(freshness == None or len(freshness)==0):
                # No SLA set at the moment. So, assuming no freshness requirement. 
                return (0.5,1.0,1.0), -1
            else:
                # (fthr, price, penalty)
                return (freshness[0],freshness[1],freshness[2]), freshness[3]
        else:
            # No Consumer Found.
            return None

    # Check and retrieve any context data available in the provider SLA 
    def get_provider_meta(self, providerid, attributes):
        self.__conn = sqlite3.connect(self.__dbname+'.db', check_same_thread=False)
        query_string = "SELECT entityId, location, regno, address\
                FROM ContextProducer \
                WHERE id="+str(providerid)+" AND isActive=1\
                LIMIT 1"

        producer = self.__conn.execute(query_string).fetchone()
        retrievable = self.__service_description[producer[0]]
        needed = set(retrievable) & set(attributes)
        if(needed):
            return {
                'location': producer[1],
                'regno': producer[2],
                'address': producer[3]
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
        if(not isinstance(providers,list)):
            providers = list(providers)
        provider_str = str(providers[0])
        for i in range(1,len(providers)):
            provider_str = provider_str + "," + str(providers[i])

        query_string = "SELECT id, url \
                FROM ContextProducer \
                WHERE isActive=1 AND id IN ("+provider_str+")"
        producers = self.__conn.execute(query_string).fetchall()
        
        for prod in producers:
            output[prod[0]] = {'url':prod[1]}
        
        return output

    # Retrieve the URLs of the matching providers and lifetimes of the attributes
    def get_context_producers(self, entityid, attributes, conditions = []) -> dict:
        self.__conn = sqlite3.connect(self.__dbname+'.db', check_same_thread=False)
        output = {}
        if(len(attributes)>0):
            context_in_desc = self.__service_description[entityid] if entityid in self.__service_description else None          
            if(context_in_desc):
                attributes = list(set(attributes) - set(context_in_desc))
        
            query_string = "SELECT id, url, price, samplingrate \
                FROM ContextProducer \
                WHERE entityId="+str(entityid)+" AND isActive=1"
        
            if(conditions):
                for cond in conditions:
                    add = ' AND '+cond
                    query_string += add

            producers = self.__conn.execute(query_string).fetchall()
            if(len(producers)>0):
                if(attributes):
                    # Some or all of the attributes need to be fetched from the response
                    att_string = "name='"+attributes[0]+"'"
                    if(len(attributes)>1):
                        for idx in range(1,len(attributes)):
                            att_string = att_string+" OR name='"+attributes[idx]+"'"

                    for prod in  producers:
                        sampling_interval = 1/prod[3]

                        query_string_2 = "SELECT name, lifetime\
                            FROM ContextAttribute\
                            WHERE producerId="+str(prod[0])+" AND ("+att_string+")"
                        
                        att_res = self.__conn.execute(query_string_2).fetchall()

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
                else:
                    # All of the attributes has been fetched from the service description
                    for prod in  producers:
                        lts = {}
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
        if(self.get_cached_life(entityid, attribute)):
            return self.update_cached_life(entityid, attribute, lifetime)
        else:
            statement = "INSERT INTO CachedLifetime (entityid, attribute, lifetime, cached) VALUES\
                ("+str(entityid)+",'"+str(attribute)+"','"+str(lifetime)+"', datetime('now'))"
            self.__conn.execute(statement)
            self.__conn.commit()
            return cursor.lastrowid

    def update_cached_life(self, entityid, attribute, lifetime):
        self.__conn = sqlite3.connect(self.__dbname+'.db', check_same_thread=False)
        cursor=self.__conn.cursor()
        query_string = "UPDATE CachedLifetime SET lifetime='"+str(lifetime)+"'\
            WHERE entityid="+str(entityid)+" AND attribute='"+attribute+"'"
        self.__conn.execute(query_string)
        self.__conn.commit()
        return cursor.lastrowid

    def remove_cached_life(self, entityid, attribute):
        self.__conn = sqlite3.connect(self.__dbname+'.db', check_same_thread=False)
        self.__conn.execute(
            "DELETE FROM CachedLifetime WHERE\
            entityid="+str(entityid)+" AND attribute='"+attribute+"'")
        self.__conn.commit()
    
    def remove_cached_life_entity(self, entityid):
        self.__conn = sqlite3.connect(self.__dbname+'.db', check_same_thread=False)
        self.__conn.execute(
            "DELETE FROM CachedLifetime WHERE\
            entityid="+str(entityid))
        self.__conn.commit()
    
    def get_cached_life(self, entityid, attribute):
        self.__conn = sqlite3.connect(self.__dbname+'.db', check_same_thread=False)
        res = self.__conn.execute(
            "SELECT lifetime, cached FROM CachedLifetime\
            WHERE entityid="+str(entityid)+" AND attribute='"+attribute+"'\
            LIMIT 1").fetchone()

        if(res):
            life = datetime.datetime.strptime(re.sub(REGEX,'',res[0]), DATETIME_FORMAT)
            cached = datetime.datetime.strptime(re.sub(REGEX,'',res[1]), DATETIME_FORMAT)
            return (life,cached)
        else:
            return res

    def get_longest_cache_lifetime_for_entity(self, entityid):
        self.__conn = sqlite3.connect(self.__dbname+'.db', check_same_thread=False)
        res = self.__conn.execute(
            "SELECT lifetime, cached FROM CachedLifetime \
            WHERE lifetime = (SELECT MAX(lifetime) \
                FROM CachedLifetime \
                WHERE entityid="+str(entityid)+") AND entityid="+str(entityid)+" \
                LIMIT 1").fetchone()         
        if(res):
            life = datetime.datetime.strptime(re.sub(REGEX,'',res[0]), DATETIME_FORMAT)
            cached = datetime.datetime.strptime(re.sub(REGEX,'',res[1]), DATETIME_FORMAT)
            return (life,cached)
        else:
            return res

    def get_expired_cached_lifetimes(self):
        self.__conn = sqlite3.connect(self.__dbname+'.db', check_same_thread=False)
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        res = self.__conn.execute(
            "SELECT entityid, attribute FROM CachedLifetime\
            WHERE lifetime<='"+now+"'").fetchall()
        return res

    # Current retrieval latency
    def update_ret_latency(self, latency):
        self.__conn = sqlite3.connect(self.__dbname+'.db', check_same_thread=False)
        self.__conn.execute(
            "UPDATE CurrentRetrievalLatency SET latency="+str(latency)+" WHERE Id=1")
        self.__conn.commit()
    
    def get_ret_latency(self):
        self.__conn = sqlite3.connect(self.__dbname+'.db', check_same_thread=False)
        res = self.__conn.execute("SELECT latency FROM CurrentRetrievalLatency WHERE Id=1 LIMIT 1").fetchone()
        return res

    # Initialize the SQLite instance 
    def seed_db_at_start(self):
        try:
            self.__create_tables()
            self.__seed_tables_with_data()
        except Exception:
            print('An error occured when seeding to database')

    def __seed_tables_with_data(self):
        self.__conn = sqlite3.connect(self.__dbname+'.db', check_same_thread=False)
        ent_str = "INSERT INTO Entity (id,name) VALUES (1,'Car'),(2,'Bike'),(3,'CarPark'),(4,'Weather'),(5,'BikePark'), (6,'Junction'),(7,'Building'),(8,'Park')"
        self.__conn.execute(ent_str)
       
        sla_str = "INSERT INTO SLA VALUES (1,0.9,1.2,2.0,0.5,1),\
            (2,0.8,1.0,2.0,0.6,1),\
            (3,0.7,0.8,1.8,0.8,1),\
            (4,0.6,0.75,1.25,1.0,0),\
            (5,0.75,1.25,2.25,0.75,1),\
            (6,0.5,0.5,1.25,1.0,1),\
            (7,0.9,1.5,3.0,0.5,1),\
            (8,0.65,1.0,1.5,0.8,1),\
            (9,0.95,2.0,4.0,0.25,1),\
            (10,0.4,0.25,0.5,1.0,1)"
            
        self.__conn.execute(sla_str)
       
        self.__conn.execute(
            "INSERT INTO ContextConsumer (id,name,isActive) VALUES\
            (101, 'Shakthi', 1),\
            (102, 'Alexey', 1),\
            (103, 'Amin', 1),\
            (104, 'Himadri', 0),\
            (105, 'Ali', 1),\
            (106, 'Ravindi', 1),\
            (107, 'Seng', 1),\
            (108, 'Arkady', 1)")

        self.__conn.execute(
            "INSERT INTO ConsumerSLA VALUES\
            (101, 2), (101, 3),\
            (102, 4), (102, 8),\
            (103, 1),\
            (104, 6), (104, 10),\
            (105, 7),\
            (106, 5),\
            (107, 1), (107, 9),\
            (108, 9)")
        
        self.__conn.execute(
            "INSERT INTO ContextService VALUES\
            (1,'Vehicle Parking Service',1),\
            (2,'Vehicle Monitor',1),\
            (3,'Weather Service',1),\
            (4,'Bike Park Service', 1),\
            (5,'VicRoads',1),\
            (6,'Building Registry',1),\
            (7,'Recreation and Parks', 1)")

        self.__conn.execute(
            "INSERT INTO ContextProducer(id, entityId, isActive, url, price, samplingrate, location, regno, address) VALUES\
                (1,1,1,'http://localhost:5000/cars?id=1',0.25, 0.033, NULL, '1HR800', NULL),\
                (2,1,1,'http://localhost:5000/cars?id=2',0.4, 0.1, NULL, '1VC546', NULL),\
                (3,1,1,'http://localhost:5000/cars?id=3',0.3, 0.2, NULL, '1DH8906', NULL),\
                (4,1,1,'http://localhost:5000/cars?id=4',0.2, 0.017, NULL, '1KP1244', NULL),\
                (5,1,1,'http://localhost:5000/cars?id=5',0.2, 0.1, NULL, '1QD7788', NULL),\
                (7,2,1,'http://localhost:5000/bikes?id=7',0.6, 0.2, NULL, 'CX123', NULL),\
                (8,2,1,'http://localhost:5000/bikes?id=8',0.5, 0.1, NULL, 'ESCBR', NULL),\
                (9,2,1,'http://localhost:5000/bikes?id=9',0.6, 0.2, NULL, 'UL146', NULL),\
                (11,3,1,'http://localhost:5000/carparks?id=11',0.4, 0.017, '-37.84938300336436, 145.11336178206872', NULL, 'Parking Lot Burwood Highway Burwood VIC 3125'),\
                (12,3,1,'http://localhost:5000/carparks?id=12',0.75, 0.033, '-37.84586713387071, 145.1149120988647', NULL, 'Building HH Burwood Highway Burwood VIC 3125'),\
                (13,3,1,'http://localhost:5000/carparks?id=13',0.3, 0.017, '-37.84621449228698, 145.11596352479353', NULL, 'Building HG Burwood Highway Burwood VIC 3125'),\
                (15,4,1,'http://localhost:5000/weather?id=15',0.2, 0.017, '-37.848027507269634, 145.1155451001933', NULL, NULL),\
                (17,5,1,'http://localhost:5000/bikeparks?id=17',0.4, 0.0033, '-37.849121741619584, 145.11557006850464', NULL, NULL),\
                (18,6,1,'http://localhost:5000/junctions?id=18',0.1, 2, '-37.850488866096384, 145.11997157347662', NULL, NULL),\
                (19,6,1,'http://localhost:5000/junctions?id=19',0.1, 2, '-37.84987043337564, 145.11519724149585', NULL, NULL),\
                (20,6,1,'http://localhost:5000/junctions?id=20',0.1, 2, '-37.84957392269724, 145.11298710129802', NULL, NULL),\
                (21,6,1,'http://localhost:5000/junctions?id=21',0.1, 2, '-37.8465240293699, 145.11367374679637', NULL, NULL),\
                (22,7,1,'http://localhost:5000/buildings?id=22',0.2, 0.00028, '-37.8495569791939, 145.11465007086437', NULL, 'Deakin Coperate Center'),\
                (23,7,1,'http://localhost:5000/buildings?id=23',0.2, 0.00028, '-37.84696457731187, 145.11455351134117', NULL, 'Deakin Library'),\
                (24,7,1,'http://localhost:5000/buildings?id=24',0.2, 0.00028, '-37.848004514177646, 145.11557275075276', NULL, 'Deakin Building G'),\
                (25,8,1,'http://localhost:5000/parks?id=25',0.4, 0.00056, '-37.84834762657227, 145.1110612751735', NULL, 'Gardiners Creek')")
        
        self.__conn.execute(
            # Assume that each car park has a varying parking cost (i.e. peak and off-peak price)
            "INSERT INTO ContextAttribute(id, name, producerId, lifetime, unit) VALUES\
                (1,'speed',1,0,'kmph'),\
                (2,'location',1,0,'cordinates'),\
                (3,'height',2,-1,'m'),\
                (4,'capacity',2,-1,'number'),\
                (5,'model',2,-1,'text'),\
                (6,'speed',2,0,'kmph'),\
                (7,'location',3,0,'cordinates'),\
                (8,'capacity',3,-1,'number'),\
                (9,'type',3,-1,'text'),\
                (10,'speed',4,0,'kmph'),\
                (11,'location',4,0,'cordianates'),\
                (12,'model',4,-1,'text'),\
                (13,'model',5,-1,'text'),\
                (14,'location',5,0,'cordinates'),\
                (15,'speed',5,0,'kmph'),\
                (16,'speed',7,0,'kmph'),\
                (17,'location',7,0,'cordinates'),\
                (18,'heading',7,0,'degrees'),\
                (19,'location',8,0,'cordinates'),\
                (20,'heading',8,0,'degrees'),\
                (21,'speed',9,0,'kmph'),\
                (22,'location',9,0,'cordinates'),\
                (23,'price',11,43200,'dollars/hr'),\
                (24,'totalSlots',11,-1,'number'),\
                (25,'availability',11,150,'number'),\
                (26,'availability',12,300,'number'),\
                (27,'maxHeight',12,-1,'m'),\
                (28,'owner',12,-1,'text'),\
                (29,'availability',13,10,'number'),\
                (30,'price',13,28800,'dollars/hr'),\
                (31,'maxHeight',13,-1,'m'),\
                (32,'temperature',15,1800,'C'),\
                (33,'windspeed',15,10,'kmph'),\
                (34,'winddiretion',15,10,'degrees'),\
                (35,'humidity',15,900,'percentage'),\
                (36,'availability',17,120,'number')")
        
        self.__conn.execute(
            "INSERT INTO ContextServiceProducer(serviceId, producerId) VALUES\
                (1,11),(1,12),(3,13),\
                (2,1),(2,2),(2,3),(2,4),(2,5),(2,7),(2,8),(2,9),\
                (3,15),\
                (4,17)")
        
        self.__conn.execute(
            "INSERT INTO CurrentRetrievalLatency VALUES (1,0)")
        
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
                location TEXT NULL,
                regno TEXT NULL,
                address TEXT NULL,
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