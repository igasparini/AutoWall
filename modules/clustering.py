import numpy as np
import laspy
from sklearn.cluster import DBSCAN
import open3d as o3d

class PointCloudCluster:
    def __init__(self, las_file_path):
        self.las_file_path = las_file_path
        self.points = None
        self.labels = None
        self.n_clusters_ = 0
        
    def load_data(self):
        las_data = laspy.read(self.las_file_path)
        self.points = np.vstack((las_data.x, las_data.y, las_data.z)).transpose()

    def cluster_data(self, eps=5.0, min_samples=100):
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(self.points)
        self.labels = db.labels_
        self.n_clusters_ = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        print('Estimated number of clusters:', self.n_clusters_)

    def visualize(self):
        geom = o3d.geometry.PointCloud()
        geom.points = o3d.utility.Vector3dVector(self.points)

        colors = np.array([(np.random.rand(), np.random.rand(), np.random.rand()) for _ in range(self.n_clusters_)])
        geom.colors = o3d.utility.Vector3dVector(colors[self.labels])

        o3d.visualization.draw_geometries([geom])