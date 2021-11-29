# -*- coding: utf-8 -*-
import cv2
import numpy as np
from numba import jit

@jit
def run(row, col, img_org, img, x, y, D):
    for r in range(row):
        for c in range(col):
            ri = D * (row - 1- r)
            ci = D * c
            rl = D * (row - 1- y[r, c])
            cl = D * (x[r, c])
            img[ri:ri+D, ci:ci+D, :] = img_org[rl:rl+D, cl:cl+D,:3]
    return img

#%%
row   = 512
col   = row
D     = 4
N     = row//D
row_t = 60
col_t = 600

#%%
img_org = cv2.imread("cat.png")
img_org = cv2.resize(img_org, (col, row), interpolation = cv2.INTER_AREA)

x,y = np.meshgrid(range(N),range(N))

img = np.zeros((row, col, 3), dtype=np.uint8)
wname = f"ArnoldsCatMap_{N:03d}"
video  = cv2.VideoWriter(f"{wname}.mp4",     cv2.VideoWriter_fourcc(*"h264"), 30, (col,   row))
tideo  = cv2.VideoWriter(f"{wname}_num.mp4", cv2.VideoWriter_fourcc(*"h264"), 30, (col_t, row_t))

print("N =", N)
for i in range(1000):
    print(i)
    img = run(N, N, img_org, img, x, y, D)
    tmg = np.zeros((row_t, col_t, 3), dtype=np.uint8)
    cv2.putText(tmg, "N=%3d, i=%3d" % (N, i), (10, 55), cv2.FONT_HERSHEY_TRIPLEX, 2.5, (255, 255, 255), thickness=2,lineType=cv2.LINE_AA)
    
    x, y = [(2*x+y) % N, (x+y) % N]
    # cv2.imwrite(f"{wname}_{i:03d}.png", img)
    for j in range(4):
        video.write(img)
        tideo.write(tmg)
        
    if i != 0 and (np.sum(np.abs(img_org - img))) == 0:
        print("return !")
        break
    
video.release()
tideo.release()