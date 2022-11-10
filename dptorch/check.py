import numpy as np
arr = np.loadtxt("V_comp_636_0.txt")
dim_size = 2

for indx in range(arr.shape[0]):
    for indx2 in range(indx+1,arr.shape[0]):
        diff = np.linalg.norm(arr[indx,:dim_size] - arr[indx2,:dim_size])
        if diff < 0.1:
            print(f"{indx} {indx2} {diff} {arr[indx,:dim_size]} {arr[indx2,:dim_size]}")