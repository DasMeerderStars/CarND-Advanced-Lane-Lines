import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
#matplotlib inline
#img=cv2.imread('./camera_cal/calibration9.jpg')
images=glob.glob('./camera_cal/calibration*.jpg')
#plt.imshow(imges)
#plt.show()

#3d point in real world space
objpoints=[]
#2d points in image plane
imgpoints=[] 

nx=9
ny=6
objp=np.zeros((ny*nx,3),np.float32)
objp[:,:2]=np.mgrid[0:nx,0:ny].T.reshape(-1,2)
#print(objp)
for idx, fname in enumerate(images):
    img=cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
    if ret==True:
        imgpoints.append(corners)
        objpoints.append(objp)
        img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
        plt.imshow(img)
        plt.show()
        cv2.waitKey(500)
cv2.destroyAllWindows()


#Test undistortion on an image
img=cv2.imread('./camera_cal/calibration1.jpg')
img_size = (img.shape[1], img.shape[0])
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,imgpoints, img_size,None,None)
undist = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('calibration_wide/test_undist.jpg',undist)
# Save the camera calibration result for later use

dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "camera_cal/cal_dist_pickle.p", "wb" ) )

# Visualize undistortion

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(undist)
ax2.set_title('Undistorted Image', fontsize=30)
plt.show()




