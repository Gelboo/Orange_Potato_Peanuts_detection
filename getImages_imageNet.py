import urllib.request
import cv2
import numpy as np
import os

def store_raw_images(url,name):
    images_link = url
    image_urls = urllib.request.urlopen(images_link).read().decode()

    if not os.path.exists(name):
        os.makedirs(name)
    count = 1
    for i in image_urls.split('\n'):
        print(count)
        try:
            urllib.request.urlretrieve(i,name+"/"+str(count)+".jpg")
            img = cv2.imread(name+"/"+str(count)+".jpg",cv2.IMREAD_GRAYSCALE)
            resized_image = cv2.resize(img,(100,100))
            cv2.imwrite(name+"/"+str(count)+'.jpg',resized_image)
            count += 1
        except Exception as e:
            print(str(e))
store_raw_images("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07711799","Potato")
