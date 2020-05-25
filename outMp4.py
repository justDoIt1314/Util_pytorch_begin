import os
import cv2
name = "car1"
path = "X:\\Downloads\\VOT2017and2018\\"+name+"\\"
filelist = os.listdir(path)
image = cv2.imread(path+filelist[0])

fps = 24 #视频每秒24帧
size = (image.shape[1], image.shape[0]) #需要转为视频的图片的尺寸
#可以使用cv2.resize()进行修改

video = cv2.VideoWriter(name+".avi", cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
#视频保存在当前目录下

for item in filelist:
    if item.endswith('.jpg'): 
    #找到路径中所有后缀名为.png的文件，可以更换为.jpg或其它
        item = path + item
        img = cv2.imread(item)
        video.write(img)

video.release()
cv2.destroyAllWindows()