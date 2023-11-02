from modules.clustering import PointCloudCluster

data_path = './data/nivo_lidar_clipped.las'

if __name__ == "__main__":
    pcl_cluster = PointCloudCluster(data_path)
    pcl_cluster.load_data()
    pcl_cluster.cluster_data()
    pcl_cluster.visualize()