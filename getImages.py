from bs4 import BeautifulSoup
import requests
import re
import urllib2
import os
import cookielib
import json

def get_soup(url,header):
    return BeautifulSoup(urllib2.urlopen(urllib2.Request(url,headers=header)),'html.parser')

def gather_images(url):
    query = raw_input("Enter Directroy Name: ")
    image_type=query
    query= query.split()

    query='+'.join(query)
    print(url)
    #add the directory for your image here
    DIR="Pictures"
    header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"
    }
    soup = get_soup(url,header)


    ActualImages=[]# contains the link for Large original images, type of  image
    for a in soup.find_all("div",{"class":"rg_meta"}):
        link , Type =json.loads(a.text)["ou"]  ,json.loads(a.text)["ity"]
        ActualImages.append((link,Type))

    print("there are total" , len(ActualImages),"images")

    if not os.path.exists(DIR):
                os.mkdir(DIR)
    DIR = os.path.join(DIR, query.split()[0])

    if not os.path.exists(DIR):
                os.mkdir(DIR)
    ###print images
    for i , (img , Type) in enumerate( ActualImages):
        try:
            req = urllib2.Request(img, headers={'User-Agent' : header})
            raw_img = urllib2.urlopen(req).read()

            cntr = len([i for i in os.listdir(DIR) if image_type in i]) + 1
            print(cntr)
            if len(Type)==0:
                f = open(os.path.join(DIR , image_type + "2_" + str(cntr)+".jpg"), 'wb')
            else :
                f = open(os.path.join(DIR , image_type + "2_" +  str(cntr)+"."+Type), 'wb')


            f.write(raw_img)
            f.close()
        except Exception as e:
            print("could not load : "+img)
            print(e)

gather_images("https://www.google.com.eg/search?q=potatoes&tbm=isch&tbs=rimg:CQ9_1d924PbC_1IjiuHZYoaFuxsPK6W3iwvE6cmtBv2iLuW5pwlH37lrbXt5MUfEy6QekepJtqBSs_1fVfx4BUbCgqwTSoSCa4dlihoW7GwEc1uF2R8a8g0KhIJ8rpbeLC8TpwRF0BN8DLxK7EqEgma0G_1aIu5bmhGHwrTFKyb24yoSCXCUffuWtte3EU6N64K-GXamKhIJkxR8TLpB6R4RXmT6gGfz8eIqEgmkm2oFKz99VxEbTNFkiNI78ioSCfHgFRsKCrBNEa7RBtuJF2es&tbo=u&sa=X&ved=2ahUKEwiSm5XmlpTcAhVM6KQKHWS2BioQ9C96BAgBEBs&biw=1920&bih=960&dpr=1")
