from configparser import Interpolation
import cv2 as cv
import os

folders = ["paper","scissors","rock"]


for folder in folders:
    photos = os.listdir(folder)
    photos = [photo for photo in photos if photo[0] != "."]
    unprocessed_photos = []
    for photo in photos:
        try:
            if int(photo[-6:-4]) <= 49:
                unprocessed_photos.append(photo)
        except:
            unprocessed_photos.append(photo)

    for photo in unprocessed_photos:
        filename = os.path.join(folder,photo)
        pic = cv.imread(filename)
        if pic.shape[0] != pic.shape[1]:
            resized = cv.resize(pic,(640,640),interpolation=cv.INTER_AREA)
            cv.imshow("{}".format(resized.shape),resized)
            if cv.waitKey(0) == ord('q'):
                break
            print("Saving to {}".format(filename))
            cv.imwrite(filename,resized)
