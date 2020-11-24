import os
import cv2
import face_recognition

for img_name in os.listdir('group_photo'):
    print(f"Reading { img_name }")

    # Read image
    pure_image = cv2.imread(f'group_photo/{img_name}')
    face_detect = face_recognition.load_image_file(f'group_photo/{img_name}')

    # Find face from the picture using model hog
    face_locations = face_recognition.face_locations(face_detect)

    # Display number of face found
    num_face = len(face_locations)
    print(f"No of faces : {num_face}")

    # Iterate through all the picture found
    for face_location in face_locations:
        # Get the point form face location
        (top, right, bottom, left) = face_location
        # print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

        # Draw rectangle on the image
        img_rect = cv2.rectangle(pure_image, (left, top), (right, bottom), (0, 0, 255), 2)

    # if no face then return a pure image
    if num_face == 0:
        img_rect = pure_image

    # Check if the process folder exits or not
    if not os.path.isdir('process'):
        os.mkdir('process')
    cv2.imwrite(f'process/process_{ img_name }', img_rect)