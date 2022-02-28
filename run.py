import os, sys
import time
import configparser

config = configparser.ConfigParser()
config.read(os.getcwd()+'/carpark-simulator/config.ini')
default_config = config['DEFAULT']

print('Starting simulator and mock platform')
os.system("python3 ./carpark-simulator/main.py &")
os.system("python3 ./coaas-mock-api/main.py &")

time.sleep(int(default_config['TotalTime'])/1000)
sys.exit(0)