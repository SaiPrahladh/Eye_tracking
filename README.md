# REAL TIME EYE TRACKING:
Eye tracking implementation to aid ocular pursuit exercises of Huntington's disease patients.

##### MOTIVATION:
This implementation is a part of a larger project which involved designing a sensing system to monitor
the progress of Huntington's chorea, which is a neurodegenerative disease which is mostly inherited,
causing loss of voluntary motor control. One such area of loss of voluntary control is in the eye muscles,
where as the diseases progresses, the patients are not able to maintain the same range of eye movement as 
a normal person. This is where the eye tracking implementation comes into play. Ocular pursuit is one of the 
prominent exercises used to observe this reduction in the range of eye movement. The eye tracking algorithm
will log the coordinates of the left and the right pupil throughout the duration of the exercise and then 
calculate the horizontal and vertical range of the patient's eye movement. The logging of these ranges will
allow us to quantify this degeneration and helps us maintain a medical history.

##### CODE AND THEORY CREDITS: https://towardsdatascience.com/real-time-eye-tracking-using-opencv-and-dlib-b504ca724ac6

The below implementation is an eye tracking system which is used for helping gather more information 
from the ocular pursuit exercises performed by Huntington's disease patient.

<img src="https://github.com/SaiPrahladh/Eye_tracking/blob/main/eye_track.gif" width="300" height="300" />

##### IMPLEMENTATION DETAILS AND DEPENDENCIES:
1. Install numpy, cv2 and dlib. Instructions to install dlib which I followed can be found in the link below:
https://www.pyimagesearch.com/2018/01/22/install-dlib-easy-complete-guide/
2. Make sure that 'shape_68.dat' is present in your code folder or specify the path to import from. 
3. Install the requests library for pushing data into a database.

##### HOW THE TRACKING IS PERFORMED?
1. The first step is to detect the face in the video frame using the 'get_frontal_face_detector' feature
from the dlib library.
2. The predictor is initialized using the pretrained 68 keypoints facial feature detector ('shape_68.dat').
3. Obtain video frames from a webcam and perform the tracking operation frame by frame.
4. Perform preprocessing steps as mentioned in the comments of the code and obtain a masked and 
thresholded image. 
5. Obtain the coordinates of the pupils and contour the thresholded image to highlight the pupils in each frame.
6. Append these coordinates to obtain a list of coordinates of the left and the right pupil.
7. Calculate the range of horizontal and vertical motion for both the eyes.
8. (Optional) Push these ranges onto an MQTT database.

The supporting functions are present in support.py
##### RUN THE CODE:
- python3 dlib_gaze.py
