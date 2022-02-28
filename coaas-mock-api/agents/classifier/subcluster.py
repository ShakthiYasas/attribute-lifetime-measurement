import numpy as np

class Subcluster:
    def __init__(self, initial_vector: np.ndarray):
        self.n_vectors = 1
        self.centroid = initial_vector
        self.connected_subclusters = set()
        self.input_vectors = [initial_vector]
        
    # Add a new vector to the subcluster and updates the centroid.
    def add(self, vector: np.ndarray):
        self.n_vectors += 1
        if(self.centroid is None):
            self.centroid = vector
        else:
            self.centroid = (self.n_vectors-1)/(self.n_vectors*self.centroid) + (vector/self.n_vectors)

    # Merge subcluster_merge into self. Update centroids.
    def merge(self, subcluster_merge, delete_merged = True):
        # Update centroid and n_vectors
        self.centroid = (self.n_vectors * self.centroid) + (subcluster_merge.n_vectors * subcluster_merge.centroid)
        self.centroid /= self.n_vectors + subcluster_merge.n_vectors
        self.n_vectors += subcluster_merge.n_vectors
        try:
            subcluster_merge.connected_subclusters.remove(self)
            self.connected_subclusters.remove(subcluster_merge)
        except Exception:
            print('merging unconnected clusters')

        for sc in subcluster_merge.connected_subclusters:
            sc.connected_subclusters.remove(subcluster_merge)
            if self not in sc.connected_subclusters and sc != self:
                sc.connected_subclusters.update({self})

        self.connected_subclusters.update(subcluster_merge.connected_subclusters)

        if(delete_merged):
            del subcluster_merge