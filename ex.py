import os

print(os.stat("./topical_clustering/Data/output.txt").st_size/(1024*1024))