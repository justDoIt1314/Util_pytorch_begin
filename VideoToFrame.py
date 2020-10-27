import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
# img = cv2.imread('G:/Course/Algorithm/LeetCode/VR-Image/QueryImages/1.png')
# rows,cols = img.shape[:2]
# pts1 = np.float32([[64,152],[659,749],[659,151]])
# pts2 = np.float32([[0,73],[cols,826],[cols,75]])
# M = cv2.getAffineTransform(pts1,pts2)
# #第三个参数：变换后的图像大小
# res = cv2.warpAffine(img,M,(rows,cols))
# cv2.imshow("src",img)
# cv2.imshow("res",res)
# cv2.waitKey()

# 加载标签
def loadLabels():
    label_paths = sorted(glob.glob(os.path.join("G:\\Course\\Algorithm\\Anomaly_Detection\\data\\shanghai-part2\\", '*')))
    labels = []
    for path in label_paths:
        labels = np.append(labels,np.load(path))

    # for path in label_paths:
    #     labels.append(np.load(path))
    nor = 30724*[0]
    nor = np.asarray(nor)[np.newaxis,:]
    labels = labels[np.newaxis,:]
    labels = np.concatenate((labels,nor),1)
    np.save("G:\\Course\\Algorithm\\Anomaly_Detection\\data\\shanghai.npy",labels) 

# 加载多个视频
def loadVideos():
    video_paths = sorted(glob.glob(os.path.join("X:\\Anomaly_Dataset\\UCSD_Anomaly_Dataset.v1p2\\subway_exit\\video\\", '*')))
    scen_index = 13
    count = 0
    for video_path in video_paths: 
        
        cap = cv2.VideoCapture(video_path)

        # shanghai dataset
        # scen_index,count = video_path.split("\\")[-1].split('.')[0].split('_')
        # dir = os.path.join("X:\\Anomaly_Dataset\\UCSD_Anomaly_Dataset.v1p2\\subway_exit\\training\\frames\\", str(int(scen_index)+7)+"_"+str(count).zfill(4))
        
        dir = os.path.join("X:\\Anomaly_Dataset\\UCSD_Anomaly_Dataset.v1p2\\subway_exit\\training\\frames\\","01")
        if not os.path.exists(dir):
            os.mkdir(dir)
        frame_index = 0
        sample_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % 5 == 0:
                cv2.imwrite(os.path.join(dir,str(sample_index).zfill(5))+".jpg",frame)
                sample_index += 1
            frame_index += 1
        
    
#loadVideos()
labels = 12980 * [0]
anomaly_index = [[8180,8210],[8292,8322],[10082,10112],[10196,10216],[12033,12073],[12090,12168]]
for index in anomaly_index:
    for i in range(index[0],index[1],1):
        labels[i] = 1
labels = np.asarray(labels)
labels = labels[np.newaxis,:]
np.save("G:\\Course\\Algorithm\\Anomaly_Detection\\data\\subway_exit.npy",labels)
    
