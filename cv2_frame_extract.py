import cv2
import time
import numpy as np
from skimage.measure import compare_ssim as ssim

cap = cv2.VideoCapture('stream_whole.avi')
cap.set(1, 20000)

nth_frame = 10
prev_frame = None

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return float(err)

def mse2(x, y):
    return np.linalg.norm(x - y)

sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

def store_line(line):
    # grayscale it 
    line = cv2.cvtColor(line, cv2.COLOR_RGB2GRAY);
    
    cv2.imwrite('frames/frame_%d.png' % frameId, line)
    # cv2.imshow('frame', line)

while cap.isOpened():
    rval, frame = cap.read()
    frameId = cap.get(1)

    if frameId > 8400 and frameId % nth_frame == 0:
        
        line = frame[0:70,].copy()
                
        if prev_frame is None:
            prev_frame = line
        else:  
            similarity = mse(line, prev_frame)
            
            if similarity < 4000:
                print('%d too similar to prev frame (%.2f)' % (frameId, similarity))
                continue
            
        print("Saving #%d" % frameId,  similarity)
        
        store_line(line)
        prev_frame = line
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
