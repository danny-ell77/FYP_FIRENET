from glob import glob                                                           
import cv2 
pngs = glob('/home/jupyter/fh-fire_dataset-2/Normal/*.png')

for j in pngs:
    img = cv2.imread(j)
    cv2.imwrite(j[:-3] + 'jpg', img)