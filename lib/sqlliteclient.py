import sqlite3

class SQLLiteClient:
    def __init__(self, db_name):
        self.__conn = sqlite3.connect(db_name+'.db')

    # Check the consumer and retrieve freshness
    def get_freshness_for_consumer(self, consumerid, slaid):
        res = self.__conn.execute(
            "SELECT * \
            FROM ContextConsumer \
            WHERE isActive=1 AND id="+consumerid)
        if(len(res) != 0):
            freshness = self.__conn.execute(
                "SELECT freshness, price, penalty \
                FROM SLA \
                WHERE isActive=1 AND id="+slaid +"\
                LIMIT 1")
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
        producers = self.__conn.execute(
                "SELECT id \
                FROM ContextProducer \
                WHERE entityId="+entityid+" AND isActive=1")
        return list(map(lambda x: x[0], producers))

    # Retrieve the URLs of the matching providers and lifetimes of the attributes
    def get_context_producers(self, entityid, attributes) -> dict:
        output = {}
        if(len(attributes)>0):
            producers = self.__conn.execute(
                "SELECT id, url, price \
                FROM ContextProducer \
                WHERE entityId="+entityid+" AND isActive=1")
            
            if(len(producers)>0):
                att_string = 'name='+attributes[0]
                if(len(attributes)>1):
                    for idx in range(1,len(attributes)):
                        att_string = att_string+'OR name='+attributes[idx]

                for prod in  producers:
                    att_res = self.__conn.execute(
                        "SELECT name, lifetime \
                        FROM ContextAttribute \
                        WHERE producerId="+prod[0]+"AND ("+att_string+")")
                    if(len(att_res)==len(attributes)):
                        lts = {}
                        for att in att_res:
                            lts[att[0]] = att[1]
                        output[prod[0]] = {
                            'url': prod[1],
                            'lifetimes': lts,
                            'cost': prod[2]
                        }
        return output
    
    # Initialize the SQLite instance 
    def seed_db_at_start(self):
        self.__create_tables()
        self.__seed_tables_with_data()
        try:
            self.__conn.commit()
        except(Exception):
            print('An error occured when seeding to database')

    def __seed_tables_with_data(self):
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
                (1,2,1,'http://localhost:5000/bike',0.6),\
                (3,1,1,'http://localhost:5000/car?id=1',0.25),\
                (4,1,1,'http://localhost:5000/car?id=10',0.25),\
                (6,3,1,'http://localhost:5000/carpark?id=5',0.4),\
                (7,3,1,'http://localhost:5000/carpark?id=12',0.3)")
        
        self.__conn.execute(
            "INSERT INTO ContextAttribute VALUES\
                (1,'speed',1,0,'kmph'),\
                (3,'speed',2,0,'kmph'),\
                (4,'speed',3,0,'kmph'),\
                (5,'hieght',3,-1,'m'),\
                (6,'hieght',4,-1,'m'),\
                (7,'availability',6,10,'s'),\
                (8,'availability',7,20,'s'),\
                (9,'maxheight',6,8,'m')")
        
        self.__conn.execute(
            "INSERT INTO ContextServiceProducer VALUES\
                (2,1),(2,3),(2,4),(1,6),(1,7)")

    def __create_tables(self):
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
                isActive BOOLEAN NOT NULL,
                FOREIGN KEY (slaId) REFERENCES SLA(id)
            );''')

        self.__conn.execute(
            '''CREATE TABLE SLA
            (
                id INT PRIMARY KEY NOT NULL,
                freshness REAL NOT NULL,
                price REAL NOT NULL,
                penalty REAL NOT NULL,
                maxrt REAL NOT NULL,
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