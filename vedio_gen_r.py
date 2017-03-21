import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
from moviepy.editor import VideoFileClip
from IPython.display import HTML


dist_pickle=pickle.load(open('./camera_cal/cal_dist_pickle.p','rb'))

mtx = dist_pickle['mtx']
dist = dist_pickle['dist']
#print(mtx)


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0,255)):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # 3) Take the absolute value of the derivative or gradient
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    binary_output = np.zeros_like(scaled_sobel)
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    # 6) Return this mask as your binary_output image
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    #binary_output = np.copy(img) # Remove this line
    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def color_threshold(image, sthresh=(0, 255),vthresh=(0,255)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= sthresh[0]) & (s_channel <= sthresh[1])] = 1

    hsv= cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hls[:,:,2]
    v_binary = np.zeros_like(s_channel)
    v_binary[(v_channel >= vthresh[0]) & (v_channel <= vthresh[1])] = 1
    
    output=np.zeros_like(s_channel)
    output[(s_binary==1)&(v_binary==1)]=1
    return output

def window_mask(width,height,img_ref,center,level):
    output=np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width)):min(int(center+width),img_ref.shape[1])]=1
    return output

class Tracker1 :
    def __init__(self, Mywindow_width=50, Mywindow_height=80, Mymargin=25, My_ym=10/72, My_xm=4/384, Mysmooth_factor=15):
        self.recent_centers=[]
        self.window_width=Mywindow_width  #宽度
        self.window_height=Mywindow_height #高度
        self.margin=Mymargin  #像素距离
        self.ym_per_pix=My_ym  #垂直方向的像素
        self.xm_per_pix=My_xm  #水平方向的像素
        self.smooth_factor=Mysmooth_factor
        self.curvatures = []


    


    def find_window_centroids(self,warped):  #车道线位置 左右中心线
        window_width=self.window_width
        window_height=self.window_height
        margin=self.margin
        window_centroids=[]
        window=np.ones(window_width) # 用于卷积的窗口模版

        #左右中心线的始位置  四分之一处开始
        l_sum=np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)],axis=0)
        l_center=np.argmax(np.convolve(window,l_sum))-window_width/2
        r_sum=np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):],axis=0)
        r_center=np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)
        window_centroids.append((l_center,r_center))  # 第一层

          # 在每一层找到最大像素的位置［0］是高度 9层
        for level in range(1,(int)(warped.shape[0]/window_height)):
            image_layer=np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:],axis=0)
            conv_signal=np.convolve(window,image_layer)
            offset=window_width/2
            l_min_index=int(max(l_center+offset-margin,0))
            l_max_index=int(min(l_center+offset+margin,warped.shape[1]))
            l_center=np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
            
            r_min_index=int(max(r_center+offset-margin,0))
            r_max_index=int(min(r_center+offset+margin,warped.shape[1]))
            r_center=np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
            window_centroids.append((l_center,r_center))

        self.recent_centers.append(window_centroids)
        #返回的是中心线的平均值
        return np.average(self.recent_centers[-self.smooth_factor:],axis=0)


def process_image(img):
    global left_fit_prev   
    global right_fit_prev
  
   
    img=cv2.undistort(img,mtx,dist,None,mtx)

    preprocessImage=np.zeros_like(img[:,:,0])
    gradx=abs_sobel_thresh(img,orient='x',thresh=(12,255)) 
    grady=abs_sobel_thresh(img,orient='y',thresh=(25,255))  
    c_binary=color_threshold(img,sthresh=(100,255),vthresh=(50,255)) 
    preprocessImage[((gradx==1)&(grady==1)|(c_binary==1))]=255

    img_size=(img.shape[1],img.shape[0])
    bot_width=.76  # .76
    mid_width=.08  #.08
    height_pct=.62  #.62
    bottom_trim=.935 # .935


    src=np.float32([[img.shape[1]*(.5-mid_width/2),img.shape[0]*height_pct],[img.shape[1]*(.5+mid_width/2),img.shape[0]*height_pct],[img.shape[1]*(.5+bot_width/2),img.shape[0]*bottom_trim],[img.shape[1]*(.5-bot_width/2),img.shape[0]*bottom_trim]])  
    offset=img_size[0]*.25  
    dst=np.float32([[offset,0],[img_size[0]-offset,0],[img_size[0]-offset,img_size[1]],[offset,img_size[1]]])
    
    # perform the transform 俯视图
    M=cv2.getPerspectiveTransform(src,dst)
    Minv=cv2.getPerspectiveTransform(dst,src)
    warped=cv2.warpPerspective(preprocessImage,M,img_size,flags=cv2.INTER_LINEAR)
    #print(warped.shape[0])
    window_width=25  
    window_height=80
    curve_centers=Tracker1(Mywindow_width=window_width,Mywindow_height=window_height,Mymargin=25,My_ym=10/720,My_xm=4/384,Mysmooth_factor=15)
    window_centroids=curve_centers.find_window_centroids(warped)
    #print(window_centroids)
    l_points=np.zeros_like(warped)
    r_points=np.zeros_like(warped)
    
    leftx=[]
    rightx=[]
    for level in range(0,len(window_centroids)):
        leftx.append(window_centroids[level][0])
        rightx.append(window_centroids[level][1])
        l_mask=window_mask(window_width,window_height,warped,window_centroids[level][0],level)  
        r_mask=window_mask(window_width,window_height,warped,window_centroids[level][1],level)  
    
        l_points[(l_points==255)|((l_mask==1))]=255
        r_points[(r_points==255)|((r_mask==1))]=255
         # Draw the results
   
    template=np.array(r_points+l_points,np.uint8)
    zero_channel=np.zeros_like(template)
    template=np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8)
    warpage=np.array(cv2.merge((warped,warped,warped)),np.uint8)
    result=cv2.addWeighted(warpage,1,template,0.5,0.0)  
    #plt.imshow(result)
    #plt.title('window fitting results')
    #plt.show()
   ## fit the   lane left and right of center 绘制啮合曲线
    yvals=range(0,warped.shape[0])
  
    res_yvals=np.arange(warped.shape[0]-(window_height/2),0,-window_height)  
   
    
    ## If less than 5 points 
    if len(leftx)<5:
        left_fit=left_fit_prev
    else:
        left_fit=np.polyfit(res_yvals,leftx,2)

    if len(rightx)<5:
        right_fit=left_fit_prev
    else:
        right_fit=np.polyfit(res_yvals,rightx,2)

    ## Check error between current coefficient and on from previous frame
    error_c_l= np.sum((left_fit[0]-left_fit_prev[0])**2) 
    error_c_l= np.sqrt(error_c_l)
    if error_c_l>.0005:
        left_fit=left_fit_prev
    else:
        left_fit = .05*left_fit+.95*left_fit_prev

    ## Check error between current coefficient and on from previous frame
    error_c_r= np.sum((right_fit[0]-right_fit_prev[0])**2) 
    error_c_r= np.sqrt(error_c_r)
    if error_c_l>.0005:
        right_fit = right_fit_prev
    else:
        right_fit = .05*right_fit+.95*right_fit_prev


    Distance_thre= (400, 560)
    l_current_fit_poly = np.poly1d(left_fit)
    r_current_fit_poly = np.poly1d(right_fit)
    distance=np.abs(l_current_fit_poly(719) - r_current_fit_poly(719))
    Rem= Distance_thre[0] < distance < Distance_thre[1]
    if Rem==False:
         left_fit=left_fit_prev
         right_fit=right_fit_prev
    
    
    left_fitx=left_fit[0]*yvals*yvals+left_fit[1]*yvals+left_fit[2]
    left_fitx=np.array(left_fitx,np.int32)
   
   
    right_fitx=right_fit[0]*yvals*yvals+right_fit[1]*yvals+right_fit[2]
    right_fitx=np.array(right_fitx,np.int32)
    

    left_fit_prev =left_fit
    right_fit_prev=right_fit

    
    
    left_lane=np.array(list(zip(np.concatenate((left_fitx-window_width/2,left_fitx[::-1]+window_width/2),axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
    right_lane=np.array(list(zip(np.concatenate((right_fitx-window_width/2,right_fitx[::-1]+window_width/2),axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)
 
    middle_marker=np.array(list(zip(np.concatenate((left_fitx+window_width/2,right_fitx[::-1]-window_width/2),axis=0),np.concatenate((yvals,yvals[::-1]),axis=0))),np.int32)

    road=np.zeros_like(img)
    road_bkg=np.zeros_like(img)
    cv2.fillPoly(road,[left_lane],color=[255,0,0])  #红的
    cv2.fillPoly(road,[right_lane],color=[0,0,255])  #蓝的
    cv2.fillPoly(road_bkg,[left_lane],color=[255,255,255])
    cv2.fillPoly(road_bkg,[right_lane],color=[255,255,255])
    cv2.fillPoly(road_bkg,[middle_marker],color=[151, 255, 183])

    #plt.imshow(road)
    #plt.show()
   
    road_warped=cv2.warpPerspective(road,Minv,img_size,flags=cv2.INTER_LINEAR)
    road_warped_bkg=cv2.warpPerspective(road_bkg,Minv,img_size,flags=cv2.INTER_LINEAR)
  
    base=cv2.addWeighted(img,1.0,road_warped_bkg,-1.0,0.0)
    result=cv2.addWeighted(base,1.0,road_warped,1,0.0)  
    #plt.imshow(result)
    #plt.title('window fitting results')
    #plt.show()

    ym_per_pix=curve_centers.ym_per_pix  #Define conversions in x and y from pixels space to meters10/72
    xm_per_pix=curve_centers.xm_per_pix  #4/384
    curve_fit_cr=np.polyfit(np.array(res_yvals,np.float32)*ym_per_pix,np.array(leftx,np.float32)*xm_per_pix,2)
    
#Calculate radius of curvature
    left_curverad = ((1 + (2*curve_fit_cr[0]*yvals[-1] + curve_fit_cr[1])**2)**1.5) / np.absolute(2*curve_fit_cr[0])
    
    camera_center=(left_fitx[-1]+right_fitx[-1])/2
    center_diff=(camera_center-warped.shape[1]/2)*xm_per_pix
    side_pos='left'
    if center_diff<=0:
           side_pos='right'

    cv2.putText(result,'Radius of Curvature='+str(round(left_curverad,3))+'(m)',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    
    cv2.putText(result,'Vehicle is '+str(abs(round(center_diff,3)))+'m'+side_pos+' of center',(50,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    return result


Output_video='output1_tracked.mp4'
Input_video='project_video.mp4'
              
clip1= VideoFileClip(Input_video)
video_clip= clip1.fl_image(process_image)
video_clip.write_videofile(Output_video,audio=False)
   





