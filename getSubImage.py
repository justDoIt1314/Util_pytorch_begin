import cv2
import argparse
import numpy as np
import os
parser = argparse.ArgumentParser(description='getsubImage')
parser.add_argument('--srcImage', default="D:\\MyWork\\VRAuto\\VRAuto\\VRAuto\\VRAuto\\bin\\x64\\Debug\\SaveImage\\2020_6_22_15_49_04.jpg", type=str, help='srcImage')
parser.add_argument('--saveImage', default="D:\\MyWork\\VRAuto\\VRAuto\\VRAuto\\VRAuto\\bin\\x64\\Debug\\SaveImage\\SingleImage", type=str, help='saveImage')

def getSubImage(srcPath):
    srcImage = cv2.imread(srcPath)
    width = np.shape(srcImage)[1]
    desWidth = int(width/2)
    srcImage = srcImage[:,0:desWidth,:]
    return srcImage

if __name__ == "__main__":
    args = parser.parse_args()
    srcPath = args.srcImage
    split = str(srcPath).lstrip().rstrip().split('\\')
    ImageName = split[-1]
    savePath = os.path.join(args.saveImage,ImageName)
    subImage = getSubImage(srcPath)
    cv2.imwrite(savePath,subImage)
    
