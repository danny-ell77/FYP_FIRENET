import os
import csv

path = 'C:\Users\user\Desktop\PROJECT\Dataset\fire-dataset-kaggle\fire_dataset\fire_images'

with open('output.csv', 'w',
          newline='') as csvfile:  # Loop through path and add all files matching *.jpg to array files
    files = []
    for r, d, f in os.walk(path):
        for _file in f:
            if '.jpg' in _file:
                files.append(_file)

    writer = csv.writer(csvfile, delimiter=',')  # Create a writer from csv module
    for f in files:  # find type of file
        t = f[0:-5]  # cut off the number and .jpg from file, leaving only the type (this may have to be changed.)

        if "dog" in t:
            t = 0
        if "cat" in t:
            t = 1
        if "mouse" in t:
            t = 2
        if "elephant" in t:
            t = 3

        writer.writerow([f, t])  # write the row to the file output.csv