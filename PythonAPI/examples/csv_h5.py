import os
import glob
import numpy as np
import h5py

mypath = "C:/test/plc_labels/SemSeg/"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles = [mypath + f for f in onlyfiles]

for filename in onlyfiles:
    point_cloud = np.genfromtxt(filename.pop(), delimiter=',', skip_header=1, usecols=(0, 1, 2))

    for filename in onlyfiles:
        arr = np.genfromtxt(file_list[0], delimiter=',', skip_header=1, usecols=(0, 1, 2))
        point_cloud = np.concatenate((point_cloud, arr), axis=0)

    with h5py.File(os.path.join(path, "{}.h5".format(filename[25:])), 'w') as out_file:
        out_file.create_dataset('data', data=point_cloud.astype(np.float32), chunks=True,
                                compression=1)
        out_file.create_dataset('label', data=np.zeros((point_cloud.shape[0],)).astype(np.uint8),
                                chunks=True,
                                compression=1)