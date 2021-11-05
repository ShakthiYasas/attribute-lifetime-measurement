import sqlite3

class SQLLiteClient:
    def __init__(self, db_name):
        self.__dbname = db_name

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
            if(len(freshness)==0):
                # No SLA set at the moment. So, assuming no freshness requirement. 
                return (0.5,1.0,1.0)
            else:
                # (fthr, price, penalty)
                return (freshness[0],freshness[1],freshness[2])
        else:
            # No Consumer Found.
            return None

    # Retrieve context producers for an entity
    def get_providers_for_entity(self,entityid):
        self.__conn = sqlite3.connect(self.__dbname+'.db', check_same_thread=False)
        producers = self.__conn.execute(
                "SELECT id \
                FROM ContextProducer \
                WHERE entityId="+str(entityid)+" AND isActive=1").fetchall()
        return list(map(lambda x: x[0], producers))

    # Retrieve the URLs of the matching providers and lifetimes of the attributes
    def get_context_producers(self, entityid, attributes) -> dict:
        self.__conn = sqlite3.connect(self.__dbname+'.db', check_same_thread=False)
        output = {}
        if(len(attributes)>0):
            producers = self.__conn.execute(
                "SELECT id, url, price, samplingrate \
                FROM ContextProducer \
                WHERE entityId="+str(entityid)+" AND isActive=1").fetchall()
            
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
                        output[prod[0]] = {
                            'url': prod[1],
                            'lifetimes': lts,
                            'cost': prod[2]
                        }
        return output
    
    def add_cached_life(self, entityid, attribute, lifetime):
        self.__conn = sqlite3.connect(self.__dbname+'.db', check_same_thread=False)
        cursor=self.__conn.cursor()
        self.__conn.execute(
            "INSERT INTO CachedLifetime (entityid, attribute, lifetime) VALUES\
            ("+str(entityid)+",'"+attribute+"',"+str(lifetime)+")")
        return cursor.lastrowid

    def remove_cached_life(self, entityid, attribute):
        self.__conn = sqlite3.connect(self.__dbname+'.db', check_same_thread=False)
        self.__conn.execute(
            "DELETE FROM CachedLifetime WHERE\
            entityid="+str(entityid)+" AND attribute='"+attribute+"')")
    
    def get_cached_life(self, entityid, attribute):
        self.__conn = sqlite3.connect(self.__dbname+'.db', check_same_thread=False)
        res = self.__conn.execute(
            "SELECT FROM CachedLifetime\
            WHERE entityid="+str(entityid)+" AND attribute='"+attribute+"'\
            ORDER BY Id DESC\
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
            (1,'Car'),(2,'Bike'),(3,'CarPark')")
       
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
            (1,'Carpark Tracker',1),(2,'Vehicle Tracker',1),(3,'Weather Monitor',0)")
        
        self.__conn.execute(
            "INSERT INTO ContextProducer VALUES\
                (1,2,1,'http://localhost:5000/bikes',0.6, 1),\
                (3,1,1,'http://localhost:5000/cars?id=3',0.25, 0.5),\
                (4,1,1,'http://localhost:5000/cars?id=4',0.4, 1),\
                (5,1,1,'http://localhost:5000/cars?id=5',0.3, 0.2),\
                (6,1,1,'http://localhost:5000/cars?id=6',0.2, 1),\
                (8,3,1,'http://localhost:5000/carparks?id=8',0.4, 0.017),\
                (9,3,1,'http://localhost:5000/carparks?id=9',0.75, 0.033),\
                (10,3,1,'http://localhost:5000/carparks?id=10',0.3, 0.017)")
        
        self.__conn.execute(
            "INSERT INTO ContextAttribute VALUES\
                (1,'speed',1,0,'kmph'),\
                (2,'location',1,0,'cordinate'),\
                (3,'regno',1,-1,'text'),"+             
                "(4,'speed',3,0,'kmph'),\
                (5,'location',3,0,'cordinate'),\
                (6,'height',3,-1,'m'),\
                (7,'capacity',3,-1,'number'),\
                (8,'model',3,-1,'text'),\
                (9,'regno',3,-1,'text'),\
                (10,'speed',4,0,'kmph'),\
                (11,'location',4,0,'cordinate'),\
                (12,'height',4,-1,'m'),\
                (13,'capacity',4,-1,'number'),\
                (14,'regno',4,-1,'text'),\
                (16,'speed',5,0,'kmph'),\
                (17,'location',5,0,'cordinate'),\
                (18,'height',5,-1,'m'),\
                (19,'capacity',5,-1,'number'),\
                (20,'model',5,-1,'text'),\
                (21,'regno',5,-1,'text'),\
                (23,'speed',6,0,'kmph'),\
                (24,'location',6,0,'cordinate'),\
                (25,'height',6,-1,'m'),\
                (26,'regno',6,-1,'text'),"+              
                "(28,'maxheight',8,-1,'m'),\
                (29,'location',8,-1,'cordinate'),\
                (30,'availability',8,30,'s'),\
                (31,'price',8,-1,'/hr'),\
                (33,'totalslots',8,-1,'number'),\
                (35,'location',9,-1,'cordinate'),\
                (36,'availability',9,60,'s'),\
                (37,'price',9,-1,'/hr'),\
                (39,'maxheight',10,-1,'m'),\
                (40,'location',10,-1,'cordinate'),\
                (41,'availability',10,150,'s'),\
                (42,'price',10,-1,'/hr')")
        
        self.__conn.execute(
            "INSERT INTO ContextServiceProducer VALUES\
                (2,1),(2,3),(2,4),(1,6),(1,7)")
        
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
                lifetime REAL NOT NULL
            );''')
        
        self.__conn.commit()