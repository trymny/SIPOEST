'''
==================================================
            Pose estimation of salmon
==================================================
 Info:

 Program by: Trym Nygård 
 Last updated: April 2022

'''

# Import required libraries 
import matplotlib.pyplot as plt
import dataAugment as da
import segment as sg
import pandas as pd
import cv2 as cv
import func as fc
import numpy as np
import sys

FOLDER = "1000"
FRAME_NO = int(sys.argv[1]) #564 #554
FILE_NAME = 'F'+str(FRAME_NO)
WIN_SIZE = 60
SAMPLE_INTER = 2 # Every 3rd pixel 

#Load images
imgL = cv.imread('images/stereo_left_'+FOLDER+'/L'+str(FRAME_NO)+'.jpg',0)
img = cv.imread('images/stereo_left_'+FOLDER+'/L'+str(FRAME_NO)+'.jpg',0)
imgR = cv.imread('images/stereo_right_'+FOLDER+'/R'+str(FRAME_NO)+'.jpg',0)

# Convert to correct data format
filePath = "data/"+FOLDER+"/"
header = ['frame','id', 'x', 'y','w', 'h'] 
dfL = pd.read_csv(filePath+"LabelsLeft.csv", usecols=header, delim_whitespace=True) 
dfL = da.crop(dfL, 'frame', FRAME_NO)
dfL = da.scale(dfL,'x',1280)
dfL = da.scale(dfL,'y',818)
print(dfL)

#EYE = 1
#HEAD = 3
#CAUDAL FIN = 5  
dfEYE = da.mask(dfL, 'id',1)
dfEYE = dfEYE.sort_values('x')
dfCAUDAL = da.mask(dfL, 'id',5)

print(dfEYE)
print(dfCAUDAL)

xE = dfEYE['x'].iloc[0] 
yE = dfEYE['y'].iloc[0] 

xC = dfCAUDAL['x'].iloc[0] 
yC = dfCAUDAL['y'].iloc[0] 

offset = 50
xL = int(xE-offset)
yL = int(yC-offset)
M = xL+abs(int(xE-offset)-int(xC+offset))
N = yL+abs(int(yE+offset)-int(yC-offset))

cursorPos = list()
#this function will be called whenever the mouse is right-clicked
def mouse_callback(event, x, y, flags, params):
    global cursorPos,img

    #right-click event value is 2
    if event == cv.EVENT_LBUTTONDOWN and len(cursorPos) <= 1:

        #store the coordinates of the right-click event
        cursorPos.append([x, y])

        #this just verifies that the mouse data is being collected
        #you probably want to remove this later
        print(cursorPos)
        img = cv.circle(img, (x,y), radius=4, color=(0, 0, 255), thickness=-1) 
        cv.imshow("image",img)
        
        
#Disparity 
with open("data/experiments/"+FILE_NAME+".txt", "w") as c:

    #Measure length
    #*********************************************************************
    #set mouse callback function for window
    cv.namedWindow('image', cv.WINDOW_NORMAL)
    cv.resizeWindow('image', img.shape[0]*2, img.shape[1]*2)
    cv.setMouseCallback('image', mouse_callback)
    cv.imshow("image",img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    header = ['W','L'] 
    df = pd.read_csv("salmonSize.csv", usecols=header, delim_whitespace=True) 
    sizeData = df.to_numpy()
    yData = sizeData[:,0]
    xData = sizeData[:,1]
    coef = np.polyfit(xData,yData,1)
    poly1d_fn = np.poly1d(coef)
    start = cursorPos[0][0] #int(xH-int(wH/2))
    end = cursorPos[1][0] #int(xC+int(wC/2))
    
    _, _, tempE = fc.blockMatching(imgL,imgR,start,cursorPos[0][1],WIN_SIZE,WIN_SIZE, winFunc="False")
    _, _, tempC = fc.blockMatching(imgL,imgR,end,cursorPos[1][1],WIN_SIZE,WIN_SIZE, winFunc="False") 
    test = imgL
    pointE = sg.dis2point(start,int(cursorPos[0][1]),start-tempE)/10
    pointC = sg.dis2point(end,int(cursorPos[1][1]),end-tempC)/10
    length = np.sqrt((pointC[0]-pointE[0])**2+(pointC[1]-pointE[1])**2+(pointC[2]-pointE[2])**2)
    print("Lenght: ", round(length,3), " [CM]")
    print("Mass: ", round(poly1d_fn(length),3), "kg")
    with open("data/measurements.txt", "w") as m:
        measurements = str(round(length,3)) +' '+str(round(poly1d_fn(length),3))
        print(measurements, file=m, flush=True)

    contour,imgC,data,cX,cY,theta,res1,res2 = sg.countours(imgL,25,6,xL,yL,M,N)
    
    f, axarr = plt.subplots(2,2)
    axarr[1,0].imshow(imgC,cmap='gray')
    axarr[1,1].imshow(contour,cmap='gray')
    axarr[0,0].imshow(imgL,cmap='gray')
    axarr[0,1].imshow(imgR,cmap='gray')
    plt.show()
   
    
    #*********************************************************************

    for x in range (xL,M,SAMPLE_INTER):
        for y in range (yL,N,SAMPLE_INTER):
            if sg.inellipse(x,y,xL+cX,yL+cY,np.sqrt((xC-xE+10)**2+(yC-yE)**2),res2,theta):
                #TM_CCOEFF_NORMED, TM_CCORR_NORMED, POC,  TM_SQDIFF_NORMED
                winL, winR, newXr = fc.blockMatching(imgL,imgR,x,y,WIN_SIZE,WIN_SIZE, winFunc="False")       
                
                #Compute disparity with integer precition
                di  = int(x-newXr)
        
                #subShiftX = fc.computeCCinter2(winL,winR)
                subShiftX = fc.computeGradCorr(winL,winR,gradMethod="sobel") 

                point = sg.dis2point(x,y,di-subShiftX)

                point = point/1000 #Convert to meter
                positionStr = str(imgL[y,x]) +' '+str(point[0]) +' '+ str(point[1]) +' '+ str(point[2])

                #Filter out points that are clearly not valid (Mistakes in the stereo matching)
                # Depth can't be negative and the fish tank is maximum 2 meters in depth. 
                if point[2] > 0 and point[2] < 2:
                    print(positionStr, file=c, flush=True)

            


