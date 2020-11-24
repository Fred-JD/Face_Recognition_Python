# importing the required libraries
import os
import cv2
import face_recognition

# The lower the value the accurate the person
TOLERANCE = 0.45

# capture the video from default camera
webcam_video_stream = cv2.VideoCapture(0)

# initialize the array variable to hold all person face
known_face_encodings = []
known_face_names = []

for file_name in os.listdir('known_faces'):
    # Iterate through the known faces
    for face_file in os.listdir('known_faces/' + file_name):
        # Load the picture
        person_image = face_recognition.load_image_file(f'./known_faces/{file_name}/{face_file}')
        # Process the face detail which is face encoding
        face_encodings = face_recognition.face_encodings(person_image)

        if len(face_encodings) == 0:
            continue

        face_encoding = face_encodings[0]

        # Store the face encoding into the list
        known_face_encodings.append(face_encoding)
        # Store the person name which is the file name
        known_face_names.append(file_name)

# initialize the array variable to hold all face locations, encodings and names
all_face_locations = []
all_face_encodings = []
all_face_names = []

# loop through every frame in the video
while True:
    # get the current frame from the video stream as an image
    ret, current_frame = webcam_video_stream.read()
    # resize the current frame to 1/4 size to proces faster
    current_frame_small = cv2.resize(current_frame, (0, 0), fx=0.25, fy=0.25)
    # detect all faces in the image
    # arguments are image,no_of_times_to_upsample, model
    all_face_locations = face_recognition.face_locations(current_frame_small, number_of_times_to_upsample=1,
                                                         model='hog')

    # detect face encodings for all the faces detected
    all_face_encodings = face_recognition.face_encodings(current_frame_small, all_face_locations)

    # looping through the face locations and the face embeddings
    for current_face_location, current_face_encoding in zip(all_face_locations, all_face_encodings):
        # splitting the tuple to get the four position values of current face
        top_pos, right_pos, bottom_pos, left_pos = current_face_location

        # change the position maginitude to fit the actual size video frame
        top_pos = top_pos * 4
        right_pos = right_pos * 4
        bottom_pos = bottom_pos * 4
        left_pos = left_pos * 4

        # find all the matches and get the list of matches
        all_matches = face_recognition.compare_faces(known_face_encodings, current_face_encoding, TOLERANCE)

        # string to hold the label
        name_of_person = 'Unknown face'

        # check if the all_matches have at least one item
        # if yes, get the index number of face that is located in the first index of all_matches
        # get the name corresponding to the index number and save it in name_of_person
        if True in all_matches:
            first_match_index = all_matches.index(True)
            name_of_person = known_face_names[first_match_index]

        # draw rectangle around the face
        cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (255, 0, 0), 2)

        # display the name as text in the image
        cv2.rectangle(current_frame, (left_pos, bottom_pos - 25), (right_pos, bottom_pos), (255, 0, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(current_frame, name_of_person, (left_pos, bottom_pos), font, 0.7, (255, 255, 255), 1)

    # display the video
    cv2.imshow("Webcam Video", current_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the stream and cam
# close all opencv windows open
webcam_video_stream.release()
cv2.destroyAllWindows()