# ACOCA-A PROTOTYPE 
This code repository contains the ACOCA-A Prototype. 

## Publications arising from the repository:
1) S. Weerasinghe, A. Zaslavsky, S. W. Loke, A. Medvedev, and A. Abken, ‘Estimating the Lifetime of Transient Context for Adaptive Caching in IoT Applications’, in ACM Symposium on Applied Computing, Brno, Czech Republic: ACM, Apr. 2022, p. 10. doi: 10.1145/3477314.3507075.
2) S. Weerasinghe, A. Zaslavsky, S. W. Loke, A. Medvedev, and A. Abken, ‘Estimating the dynamic lifetime of transient context in near real-time for cost-efficient adaptive caching’, SIGAPP Appl. Comput. Rev., vol. 22, no. 2, pp. 44–58, Jun. 2022, doi: 10.1145/3558053.3558057.
3) S. Weerasinghe, A. Zaslavsky, S. W. Loke, A. Abken, A. Hassani, and A. Medvedev, ‘Adaptive Context Caching for Efficient Distributed Context Management Systems’, in ACM Symposium on Applied Computing, Tallinn, Estonia: ACM, Mar. 2023, p. 10. doi: 10.1145/3555776.3577602.
4) An Agent Based Learning Approach to Adaptive Context Caching in Distributed Context Management Systems (Journal Paper - Being processed at ACM Transactions in IoT Journal) - also available in S. Weerasinghe, A. Zaslavsky, S. W. Loke, A. Abken, and A. Hassani, ‘Reinforcement Learning Based Approaches to Adaptive Context Caching in Distributed Context Management Systems’. arXiv, Dec. 22, 2022. Accessed: Dec. 23, 2022. [Online]. Available: http://arxiv.org/abs/2212.11709.

### Datasets 
Dataset used to simulate the behaviours of the entities involved in the use case can be found at /datasets. 
Datasets related to carparks, vehicles, weather, and places can be found here. 
These datasets can be simulated using the entitySimulator found in the code OR using the IoT dats simulator (https://github.com/IBA-Group-IT/IoT-data-simulator/tree/master).

Context consumer SLAs are available in contextConsumer.json.
All the registered Context Providers are available in contextService.json.

### Experimental Setup
The JMeter file (.jmx) used to simulate the context queries can be found at /experiment/test-plan.jmx.
Use the requests.csv in /datasets to set the context query load. 

## NOTICES
The service containers assuming an alreday running container instance of 'mongodb'.
Please use the dockerized-solution branch for the solution using Docker containers related to the publication 'Estimating the lifetime of transient context items'. 
