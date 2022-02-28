import numpy as np
from datetime import datetime
from scipy.spatial.distance import cosine

from agents.classifier.subcluster import Subcluster

# The online clustering algorithm
class LinkedCluster:
    def __init__(self, cluster_sim_thresh, subcluster_sim_thresh, max_pair_similarity):        
        self.__clusters = []
        self.__max_clusters = 32

        self.temp_deleted_clusters = {}
        self.__max_pair_similarity = max_pair_similarity
        self.__cluster_similarity_threshold = cluster_sim_thresh
        self.__subcluster_similarity_threshold = subcluster_sim_thresh

    # Get all the cluster information
    def get_clusters(self):
        return self.__clusters

    # Get a snapshot of the current state space
    # Return a feacture vector of all the centroids in the state space
    def get_current_state(self):
        states = np.zeros([1, self.__max_clusters*16])
        for cls_idx in range(0, len(self.__clusters)):
            centroid_func = lambda inp: [sum(m)/float(len(m)) for m in zip(*inp)]
            
            cent_coll = [sb.centroid for sb in self.__clusters[cls_idx][0]]
            main_cluster_centroid = centroid_func(cent_coll)
            for j in range(0,len(main_cluster_centroid)):
                states[np.arange(1), cls_idx+j] = main_cluster_centroid[j]

        return states
        
    # Estimate the cluster id for new_vector.
    def predict(self, new_vector: np.ndarray) -> int:
        if(len(self.__clusters) == 0):
            # Handle first vector
            self.__clusters.append(([Subcluster(new_vector)], datetime.now()))
            return 0

        best_subcluster = None
        best_similarity = -np.inf
        best_subcluster_id = None
        best_subcluster_cluster_id = None

        for cl_idx, cl in enumerate(self.__clusters):
            for sc_idx, sc in enumerate(cl[0]):
                cossim = 1.0 - cosine(new_vector, sc.centroid)
                if(cossim > best_similarity):
                    best_subcluster = sc
                    best_similarity = cossim
                    best_subcluster_cluster_id = cl_idx
                    best_subcluster_id = sc_idx

        if(best_similarity >= self.__subcluster_similarity_threshold):
            # Add to existing subcluster
            best_subcluster.add(new_vector)
            self.__update_cluster(best_subcluster_cluster_id, best_subcluster_id)
            assigned_cluster = best_subcluster_cluster_id
        else:
            # Create new subcluster
            new_subcluster = Subcluster(new_vector)
            cossim = 1.0 - cosine(new_subcluster.centroid, best_subcluster.centroid)
            if(cossim >= self.__get_similarity_threshold(best_subcluster.n_vectors, 1)): 
                # New subcluster is part of existing cluster
                self.__add_edge(best_subcluster, new_subcluster)
                
                val = self.__clusters[best_subcluster_cluster_id]
                val[0].append(new_subcluster)
                self.__clusters[best_subcluster_cluster_id] = val

                assigned_cluster = best_subcluster_cluster_id
            else:
                # New subcluster is a new cluster
                if(len(self.__clusters) >= self.__max_clusters):
                    # Evict the time weighted least frequently used cluster
                    evicting_cluster_id = 0

                    self.temp_deleted_clusters[evicting_cluster_id] = self.__clusters[evicting_cluster_id]
                    self.__clusters[evicting_cluster_id] = ([new_subcluster], datetime.now())
                else:
                    self.__clusters.append(([new_subcluster], datetime.now()))
                    assigned_cluster = len(self.__clusters) - 1

        return assigned_cluster

    # Add an edge between subclusters sc1, and sc2.
    @staticmethod
    def __add_edge(sc1: Subcluster, sc2: Subcluster):
        sc1.connected_subclusters.add(sc2)
        sc2.connected_subclusters.add(sc1)

    # Update cluster
    def __update_cluster(self, cl_idx: int, sc_idx: int):
        updated_sc = self.__clusters[cl_idx][0][sc_idx]
        severed_subclusters = []
        connected_scs = set(updated_sc.connected_subclusters)
        
        for connected_sc in connected_scs:
            connected_sc_idx = None
            for c_sc_idx, sc in enumerate(self.__clusters[cl_idx][0]):
                if(sc == connected_sc):
                    connected_sc_idx = c_sc_idx
            
            if(connected_sc_idx is None):
                raise ValueError(f"Connected subcluster of {sc_idx} was not found in cluster list of {cl_idx}.")
            
            cossim = 1.0 - cosine(updated_sc.centroid, connected_sc.centroid)
            if(cossim >= self.__subcluster_similarity_threshold):
                self.__merge_subclusters(cl_idx, sc_idx, connected_sc_idx)
            else:
                are_connected = self.__update_edge(updated_sc, connected_sc)
                if(not are_connected):
                    severed_subclusters.append(connected_sc_idx)
        
        for severed_sc_id in severed_subclusters:
            severed_sc = self.__clusters[cl_idx][0][severed_sc_id]
            
            if(len(severed_sc.connected_subclusters) == 0):
                for cluster_sc in self.__clusters[cl_idx][0]:
                    if(cluster_sc != severed_sc):
                        cossim = 1.0 - cosine(cluster_sc.centroid,severed_sc.centroid)

                        if(cossim >= self.__get_similarity_threshold(cluster_sc.n_vectors,severed_sc.n_vectors)):
                            self.__add_edge(cluster_sc, severed_sc)

            if(len(severed_sc.connected_subclusters) == 0):
                val = list(self.__clusters[cl_idx])
                val[0] = self.__clusters[cl_idx][0][:severed_sc_id] + self.__clusters[cl_idx][0][severed_sc_id + 1:]
                self.__clusters[cl_idx] = tuple(val)
                self.__clusters.append([severed_sc])

    # Merge subclusters with id's sc_idx1 and sc_idx2 of cluster with id cl_idx.
    def __merge_subclusters(self, cl_idx, sc_idx1, sc_idx2):
        sc2 = self.__clusters[cl_idx][0][sc_idx2]

        self.__clusters[cl_idx][0][sc_idx1].merge(sc2)
        self.__update_cluster(cl_idx, sc_idx1)

        self.__clusters[cl_idx] = (self.__clusters[cl_idx][0][:sc_idx2] + self.__clusters[cl_idx][0][sc_idx2 + 1:], datetime.now())
        for sc in self.__clusters[cl_idx][0]:
            if(sc2 in sc.connected_subclusters):
                sc.connected_subclusters.remove(sc2)
    
    # Compare subclusters sc1 and sc2, remove or add an edge depending on cosine similarity.
    # Returns,
    # True if the edge is valid
    # False if the edge is not valid
    def __update_edge(self, sc1: Subcluster, sc2: Subcluster):
        cossim = 1.0 - cosine(sc1.centroid, sc2.centroid)
        threshold = self.__get_similarity_threshold(sc1.n_vectors, sc2.n_vectors)
        if(cossim < threshold):
            try:
                sc1.connected_subclusters.remove(sc2)
                sc2.connected_subclusters.remove(sc1)
            except Exception:
                print("Attempted to update an invalid edge that didn't exist. Edge remains nonexistant.")

            return False
        else:
            sc1.connected_subclusters.add(sc2)
            sc2.connected_subclusters.add(sc1)

            return True

    # Compute the similarity threshold
    def __get_similarity_threshold(self, k: int, kp: int) -> float:
        sim = (1.0 + 1.0 / k * (1.0 / self.__cluster_similarity_threshold ** 2 - 1.0))
        sim *= (1.0 + 1.0 / kp * (1.0 / self.__cluster_similarity_threshold ** 2 - 1.0))
        sim = 1.0 / np.sqrt(sim) 
        sim = self.__cluster_similarity_threshold ** 2 + (self.__max_pair_similarity - 
            self.__cluster_similarity_threshold ** 2) / (1.0 - self.__cluster_similarity_threshold ** 2) \
            * (sim - self.__cluster_similarity_threshold ** 2)

        return sim