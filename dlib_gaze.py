import cv2
import dlib
import numpy as np
from support import shape_to_np,eye_on_mask,contouring
import requests


# Function to push data as an mqtt client
def publish_data(patient,value):
    URL = "http://ec2-xx-xxx-xx-xx.compute-x.amazonaws.com/senddata.php"
    user_name='*****'
    password='*****'
    rtype='put'
    table='eye_tracking'
    patient_id = patient
    sensor_id ='12'
    time_stamp = "2012-12-30 12:12:12"
    value = value
    PARAMS ={'user_name':user_name,'password':password,'rtype':rtype,'table':table, 'patient_id': patient_id, 
             'sensor_id':sensor_id, 'time_stamp':time_stamp, 'value':value}
    output = requests.get(url = URL, params = PARAMS).text


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_68.dat')

# points acquired from dlib's 68 point detector
left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

cap = cv2.VideoCapture(0)

# List of coordinates for the left and right eye
left_eye = []
right_eye = []
cv2.namedWindow('image')

# Kernel used for dilation
kernel = np.ones((9, 9), np.uint8)

def nothing(x):
    pass

# Nothing function is just a place holder to be filled for creating a trackbar
cv2.createTrackbar('threshold', 'image', 0, 255, nothing)

# Number of frames for recording the video is 'i'
i = 200
while(i):

    ret, img = cap.read()
    print(i,'<<<<<<<<')
    thresh = img.copy()

    # Convert image into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Get the rectangles, i.e detect face in the image
    # 1 represents the number of expected faces in a frame
    rects = detector(gray, 1)
    
    # Looping through each person's face in a frame
    for rect in rects:

    	# predictor is used to identify the 68 keypoints 
    	# on a detected face
        shape = predictor(gray, rect)
        
     	# the shape_to_np function returns
     	# x,y coordinates of the 68 keypoints
        shape = shape_to_np(shape)

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        
        # eye_on_mask function will create a mask
        # for both left and right eye.
        mask = eye_on_mask(mask, shape, left)
        mask = eye_on_mask(mask, shape, right)

        # Perform morphological transformation of 
        # dilation on the image which increases object area
        mask = cv2.dilate(mask, kernel, 5)

        # mask the image to just get the eyes
        eyes = cv2.bitwise_and(img, img, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid = (shape[42][0] + shape[39][0]) // 2
        
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        #threshold = cv2.getTrackbarPos('threshold', 'image')
        
        #Experimentally found threshold value
        threshold = 83


        # Obtain the thresholded image
        _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY_INV)
        thresh = cv2.erode(thresh, None, iterations=2) #1
        thresh = cv2.dilate(thresh, None, iterations=4) #2
        thresh = cv2.medianBlur(thresh, 3) #3
        
        
        left_pupil = contouring(thresh[:, 0:mid], mid, img)
        right_pupil = contouring(thresh[:, mid:], mid, img, True)
        
        if left_pupil is not None:
        	left_eye.append(left_pupil[0])
        	left_eye.append(left_pupil[1])

        if right_pupil is not None:
        	right_eye.append(right_pupil[0])
        	right_eye.append(right_pupil[1])

      
    cv2.imshow('eyes', img)
    #cv2.imshow("image", thresh)
    i-=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
left_eye = np.array(left_eye)
right_eye = np.array(right_eye)



left_eye = left_eye.reshape(-1,2)

right_eye = right_eye.reshape(-1,2)

# np.save('/home/sai/Desktop/rapid_HD_CV/left_pupil.npy',left_eye)
# np.save('/home/sai/Desktop/rapid_HD_CV/right_pupil.npy',right_eye)


range_x_left = np.max(left_eye[:,0]) - np.min(left_eye[:,0])
range_y_left = np.max(left_eye[:,1]) - np.min(left_eye[:,1])


range_x_right = np.max(right_eye[:,0]) - np.min(right_eye[:,0])
range_y_right = np.max(right_eye[:,1]) - np.min(right_eye[:,1])

score_x_left, score_y_left = min((range_x_left/57) * 0.75,0.75), min((range_y_left/57) * 0.75,0.75)
print('Left X Range score:',score_x_left,'points')
print('Left Y Range score:',score_y_left,'points')
print('Total Left eye range score:',score_x_left + score_y_left,'points')
score_x_right, score_y_right = min((range_x_right/57) * 0.75,0.75), min((range_y_right/57) * 0.75,0.75) 
print('------------------------------------------------')
print('Right X Range score:',score_x_right,'points')
print('Right Y Range score:',score_y_right,'points')
print('Total Right eye range score:',score_x_right + score_y_right,'points')
print('------------------------------------------------')
full_score = score_x_right+score_x_left + score_y_right + score_y_left
if full_score >= 3:
	full_score = 3
print('Total Range Score =',full_score ,'/','3')

if (full_score/3 > 0.9):
	print('Good job! Your eyes are working perfectly fine')
elif (full_score/3 > 0.75):
	print('On Track! Can progress more.')
elif (full_score/3 < 0.5):
	print('Please contact your doctor soon!')

push_val = str(score_x_left),str(score_y_left),str(score_x_right),str(score_y_right)
print(push_val)
publish_data(12,push_val)


cap.release()
cv2.destroyAllWindows()
