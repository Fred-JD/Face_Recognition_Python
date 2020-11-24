"""
Reference : https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py
"""

import os
import cv2
import face_recognition
import numpy as np

TOLERANCE = 0.5
# Open webcam
web_cam = cv2.VideoCapture(0)

person_face_encodings = []
person_name = []

# Iterate through the known faces
for folder in os.listdir('known_faces'):
    for face_file in os.listdir('known_faces/' + folder):
        # Load the picture
        person_image = face_recognition.load_image_file(f'./known_faces/{folder}/{face_file}')
        # Process the face detail which is face encoding
        face_encoding = face_recognition.face_encodings(person_image)[0]

        # Store the face encoding into the list
        person_face_encodings.append(face_encoding)
        # Store the person name which is the file name
        person_name.append(folder)

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

print(person_name)

while True:
    # get the single frame of the video of webcam
    ret, frame = web_cam.read()

    # Resize frame of video resolution to 1/4
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color to RGB color
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video
    if process_this_frame:
        # Find all the face of location and encodinngs from every single frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame)

        face_name = []
        for face_encoding in face_encodings:
            # Check if the face is match or not
            matches = face_recognition.compare_faces(person_face_encodings, face_encoding, TOLERANCE)
            name = "Unknown"

            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = person_name[first_match_index]

            face_distances = face_recognition.face_distance(person_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = person_name[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (225, 0, 0), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 0, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (225, 225, 255), 1)

    # Display the result image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

web_cam.release()
cv2.destroyAllWindows()