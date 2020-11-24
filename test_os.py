import os

for file_name in os.listdir('known_faces'):
    print(file_name)
    for img_name in os.listdir('known_faces/' + file_name):
        print(file_name + ' == ' + img_name)