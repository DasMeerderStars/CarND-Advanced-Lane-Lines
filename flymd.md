

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistorted.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./test_images/tracked0.jpg "Road Transformed"
[image4]: ./output_images/binary_image0.jpg  "Binary Example"
[image5]: ./output_images/warped1.jpg  "Warp Example"
[image6]: ./output_images/color_fit_lines1.jpg "Fit Visual"
[image7]: ./output_images/tracked0.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1.Before lane detection pipeline,it is that computed the camera matrix and distortion coefficients.

The code for this step in the file called `Camera_Calibration.py`.
The camera using the chessboard images in 'camera_cal/*.jpg' to calibrate. AS following:
* Convert images to grayscale
* Find chessboard corners with `findChessboardCorners()` function, assuming a 9x6 board


I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. 

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]
The final calibration matrices saved in the pickle file of 'cal_dist_pickle.p' in 'camera_cal/'

###Pipeline (single images)

####1. First image is original and Second is a distortion-corrected image.Using the function of cv2.undistort in image_gen1.py to apply the distortion correction to one of the test images like this one:
![alt text][image2]

![alt text][image3]
####2. I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # 144-148 # in `image_gen.py`). Gradx is x gradient and Grady is y gradient. Here's an example of my output for this step.

![alt text][image4]

####3. The code for my perspective transform includes a function called `cv2.warpPerspective`, which appears in lines 126 in the file `image_gen.py`.  The function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:


This resulted in the following source and destination points:

| Source                 | Destination   | 
|:----------------------:|:-------------:| 
| 575,446.3999939        | 320, 0        | 
| 704,446.3999939        | 320, 720      | 
| 1126.40002441,673.20001221| 960, 720   | 
| 153.6000061,673.20001221| 320, 720     | 

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

####4. Lane-line is detected in the function of find_window_centroids in  the Tracker1 class in the file `image_gen.py`.
First, a histogram is applied to the bottom half of binary threshold image.
Second, using sliding window method to find all lane line centres in the image for both lane lines
Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image6]

####5. Radius of curvature of the lane is calculated as follow:
           First,converts detected lane pixels space to meters. 
           Second,calculate the curvature using a function.
       Position of the vehicle is that subtracting the center of the image from lane's center point.
            Lane's center point is the average of the left and right lane's fitted polynomial.
I did this in lines 277 through 293 in my code in `image_gen.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 243 through 272 in my code in `image_gen.py` in the function `cv2.fillPoly`.  Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output1_tracked.mp4)

---
### Project code

- `Camera_Calibration.py`: for calibrates the camera 
- `image_gen1.py`: for distortion-corrected 
- `image_gen.py`:  the main script 
- `vedio_gen_r.py`: for generate vedio 


### Discussion

####1. When there are a lot of shadows on the road, the sliding windows can't find a appropriate lane line.
fLyMd-mAkEr

Here I'll talk about the approach I took,  where the pipeline might fail when I use to the challenge video.
