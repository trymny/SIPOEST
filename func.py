'''
==================================================
 Additonal functions for phaseCorrelation.py
==================================================
 Info:

 Written by: Trym Nygård 
 Last updated: March 2022
 
'''

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import cv2 as cv
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RectBivariateSpline


def stereoImgResize(imgL,imgR, scale_percent,interMethod):
    width = int(imgL.shape[1] * scale_percent / 100)
    height = int(imgL.shape[0] * scale_percent / 100)
    dimL = (width, height)
    imgL = cv.resize(imgL, dimL, interpolation = interMethod)
    width = int(imgR.shape[1] * scale_percent / 100)
    height = int(imgR.shape[0] * scale_percent / 100)
    dimR = (width, height)
    imgR = cv.resize(imgR, dimR, interpolation = interMethod)

    return imgL, imgR

def stereoImgCrop(img,X,Y,N,M):
    win = img[int(Y-(N/2)):int(Y+(N/2)), int(X-(M/2)):int(X+(M/2))]
    #print("Window size: ", winL.shape)
    return win

def windowFunc(funcType, win):
    width,height = win.shape
    Ww=eval('np.'+funcType+'(width)')
    Wh=eval('np.'+funcType+'(height)')
    windowFilter=np.outer(Ww,Wh)
    win = win * windowFilter
    return win

def magMask(dft,param=0):
    for i in range(dft.shape[0]):
        maxValue = abs(np.max(dft[i]))
        if(maxValue <= param):
            dft[i] = 0.00001   # zeroes out row i
            #dft[:,i] = 0.00001 # zeroes out column i
    return dft

def radMask(dft,param=0.6):
    N = dft.shape[1]
    radius = param*((N)/(2))
    maskedDft = dft[int(dft.shape[0]/2-radius):int(dft.shape[0]/2+radius),int(dft.shape[0]/2-radius):int(dft.shape[0]/2+radius)]
    return maskedDft

def computeSID(dftL,dftR, radius=0.6, magThres=0, plot=False):
    ''' 
    Subspace identification (SID) method for sub-pixel estimation.
    SID is an extension to the Phase Correlation Method and developed by William Scott Hoge
    and based on the paper "A Subspace Identification Extension to the Phase Correlation Method"
    Pros:
        Accurate and robust for images with large shifts or rotations. 
        Consistent results even for images with low correlation.
        Can achieve up to 1/20th pixel accuracy
    Cons:
        SVD can be quite slow and may become unstable for very small images
        Can be sensitive to aliasing effects during image acquisition and edge effects caused by the DFT

    Preprocessing:
    A window function such as a Hamming or a Blackman window should be applied to each image to avoid edge
    effects.
    Masking out higher frequencies in the Normalized cross-power spectrum R should be performed 
    in order to reduce the impact of aliasing.

    How to use:
    plot = True 
        Subplot of the phase difference and the 1D unwrapped phase difference.
        Plot of Frequency Spectrum of the discrete Fourier transform.
    Radius 
    '''

    M = dftL.shape[0]
    N = dftL.shape[1]

    dftShiftL = np.fft.fftshift(dftL)
    dftShiftR = np.fft.fftshift(dftR)
    
    #Mask out spectral components that lie outside a radius from the central peak
    maskedDftL = radMask(dftShiftL,radius)
    maskedDftR = radMask(dftShiftR,radius)
    
    #Mask out spectral components for DFT's that have peaks with magnitudes less than a threshold value
    #maskedDftL = magMask(maskedDftL,magThres)
    #maskedDftR = magMask(maskedDftR,magThres)

    # Normalized cross-power spectrum 
    R = (maskedDftL*np.conjugate(maskedDftR))/(np.abs(maskedDftL*np.conjugate(maskedDftR)))  

    U,S,V  = np.linalg.svd(R) #Reduce from 2D to 1D 

    dominantV = V[np.argmax(S),:]
    dominantU = U[:,np.argmax(S)]

    A = []
    for i in range(len(V)):
        A.append(i)
    A = np.vstack([A, np.ones(len(A))]).T #Refered to as R in the paper

    angleV = np.angle(dominantV)  #Find the angle
    unwrapV = np.unwrap(angleV)

    fittedLineV, res,rank,s = np.linalg.lstsq(A,unwrapV,rcond=-1) #Equation 6 inv(R^TR)R^Tunwrap(v)
    muX = fittedLineV[0] #slope of the fitted line
    cX = fittedLineV[1] #abscissa of the fitted line
    subShiftX = muX * (M / (2*np.pi)) #translational shift

    angleU = np.angle(dominantU)  #Find the angle
    unwrapU = np.unwrap(angleU)
    fittedLineU, res,rank,s = np.linalg.lstsq(A,unwrapU,rcond=-1) #Equation 6 inv(R^TR)R^Tunwrap(v)
    muY = fittedLineU[0] #slope of the fitted line
    #cY = fittedLineU[1] #abscissa of the fitted line
    subShiftY = muY * (N / (2*np.pi)) #translational shift

    if plot: 
        plotDFT("Frequency Spectrum Left Image",maskedDftL)
        plotDFT("Frequency Spectrum Right Image",maskedDftR)

        temp = unwrapV.shape[0]
        fx= (np.linspace(-temp/2,temp/2-1,temp)/temp);    # FM = number of pixel in x direction 
        fx = fx *2 * np.pi                    # Multiply with 2pi to get the (spatial) frequency 
        fx = np.reshape(fx, (-1, 1))          # Go from shape (M,) to (M,1)

        fig, axarr = plt.subplots(nrows=1, ncols=2, figsize=(5, 3))
        axarr[0].set_title("Phase component")
        axarr[0].grid(True)
        axarr[0].plot(fx, angleV)
        
        x = np.linspace(0,temp,temp)
        axarr[1].set_title("Unwrapped Phase component")
        axarr[1].grid(True)
        axarr[1].plot(x, unwrapV)
        line, = axarr[1].plot(x, muX*x+cX, "--")
        line.set_label('Fitted line')
        axarr[1].text(0,max(unwrapV),"Estimated shift: "+str(round(subShiftX,3)),fontsize=12)
        axarr[1].text(max(x)/2,(max(unwrapV)+min(unwrapV))/2,"Slope: "+str(round(muY,7)))
        axarr[1].legend(fontsize=15, bbox_to_anchor=(1, 0.15))
        plt.show()
    
    
    return subShiftX,subShiftY

def computeGradCorr(img, img2, gradMethod = "hvdiff", polyDeg = 2, nPoints = 200, plot = False):
    M = img.shape[0]
 
    if gradMethod == "hvdiff":

        gHor = np.zeros_like(img.astype('float64'))
        gVer = np.zeros_like(img.astype('float64'))
        gHor2 = np.zeros_like(img2.astype('float64'))
        gVer2 = np.zeros_like(img2.astype('float64'))
       
        for i in range(1,img.shape[0]-1):
            for j in range(1,img.shape[1]-1):
                gHor[i,j] = (int(img[i+1,j])-int(img[i-1,j]))
                gVer[i,j] = (int(img[i,j+1])-int(img[i,j-1]))
                gHor2[i,j] = (int(img2[i+1,j])-int(img2[i-1,j]))
                gVer2[i,j] = (int(img2[i,j+1])-int(img2[i,j-1]))
        
    elif gradMethod == "sobel":
        gHor = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
        gVer = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)
        gHor2 = cv.Sobel(img2,cv.CV_64F,1,0,ksize=5)
        gVer2 = cv.Sobel(img2,cv.CV_64F,0,1,ksize=5)
    elif gradMethod == "scharr":
        gHor = cv.Scharr(img, cv.CV_64F, 1, 0) 
        gVer = cv.Scharr(img, cv.CV_64F, 0, 1) 
        gHor2 = cv.Scharr(img2, cv.CV_64F, 1, 0) 
        gVer2 = cv.Scharr(img2, cv.CV_64F, 0, 1) 
    elif gradMethod == "False":
        gHor = img
        gVer = img
        gHor2 = img2
        gVer2 = img2

    g = gHor - np.imag(gVer)
    g2 = gHor2 - np.imag(gVer2)

    dftG = (np.fft.fft2(g))
    dftG2 = (np.fft.fft2(g2))
    
    cc = np.fft.ifft2(dftG*np.conjugate(dftG2))
     
    for i in range(int(M/2)):
        cc = np.roll(cc, -1, axis=1)

    cc = cc.real
    #plotSurface(cc)
    
    idxMax = np.unravel_index(cc.argmax(), cc.shape) 
    
    dist = 1
    if (idxMax[1]+dist >= M):
        print("too big", idxMax[1]+dist )
        return 0
    if (idxMax[1]-dist <= 0):
        print("too small", idxMax[1]-dist )
        return 0

    y = np.array([cc[idxMax[0],idxMax[1]-dist], cc[idxMax[0],idxMax[1]], cc[idxMax[0],idxMax[1]+dist]])
    x = np.linspace(idxMax[1]-dist,idxMax[1]+dist,3).astype(int)

    fit = np.polyfit(x,y,polyDeg)
    p = np.poly1d(fit) 
    newX = np.linspace(idxMax[1]-dist,idxMax[1]+dist,nPoints)
    
    maxPeak = [newX[np.argmax(p(newX))],p(newX[np.argmax(p(newX))])]
    x_shift = int(M/2)-maxPeak[0]

    if plot:
         
        temp = M
        xx = np.linspace(0,temp,temp).astype(int)  
        fig, axarr = plt.subplots(1,1)
        axarr.grid(True)
        axarr.plot(xx,cc[idxMax[0],:] )
        axarr.plot(newX,p(newX))
        axarr.plot(x[0],y[0], marker="o", color="green")
        axarr.plot(x[1],y[1], marker="o", color="green")
        axarr.plot(x[2],y[2], marker="o", color="green")
        axarr.text(maxPeak[0],maxPeak[1],str(x_shift),horizontalalignment='right')
        plt.show()
         
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(img,cmap='gray')
        axarr[1].imshow(cv.addWeighted(gHor, 0.5, gVer, 0.5, 0),cmap='gray')
        f.suptitle(gradMethod)
        axarr[1].set_title(str(x_shift))
        plt.show()
        
    
    return x_shift

def computePC(dftL,dftR,  polyDeg = 2, nPoints = 200,  subpx = True, plot = False):

    M = dftL.shape[0]

    # Normalized cross-power spectrum 
    R = (dftL*np.conjugate(dftR))/(np.abs(dftL*np.conjugate(dftR)))  

    # Taking the inverse discrete Fourier transform to find the phase correlation
    PC = np.fft.ifft2(R)

    # Using the fourier shif theorem to move the maximum peak
    PC = np.fft.fftshift(PC)
    
    # Compute magnitude
    PC = abs(PC)

    # Using a 2d Gaussian distribution to smooth the phase correlation and to remove false peaks
    PC = cv.GaussianBlur(PC,(5,5),0)

    idxMax = np.unravel_index(PC.argmax(), PC.shape) # Extract indices of max peak from POC function 
    if subpx:
        dist = 1
        y = np.array([PC[idxMax[0],idxMax[1]-dist], PC[idxMax[0],idxMax[1]], PC[idxMax[0],idxMax[1]+dist]])
        x = np.linspace(idxMax[1]-dist,idxMax[1]+dist,3).astype(int)

        fit = np.polyfit(x,y,polyDeg)
        p = np.poly1d(fit) 
        newX = np.linspace(idxMax[1]-dist,idxMax[1]+dist,nPoints)
        
        maxPeak = [newX[np.argmax(p(newX))],p(newX[np.argmax(p(newX))])]
        xShift = int(M/2)-maxPeak[0]

        if plot:
            xx = np.linspace(0,M,M).astype(int)  
            fig, axarr = plt.subplots(1,1)
            axarr.grid(True)
            axarr.plot(xx, PC[idxMax[0],:] )
            axarr.plot(newX,p(newX))
            axarr.plot(x[0],y[0], marker="o", color="green")
            axarr.plot(x[1],y[1], marker="o", color="green")
            axarr.plot(x[2],y[2], marker="o", color="green")
            plt.show()

            #plotSurface(PC)

    else:
        yShift = int(PC.shape[0]/2-idxMax[0])       # Compute integer shift Y from the max peak in POC function
        xShift = int(PC.shape[1]/2-idxMax[1])       # Compute integer shift X from the max peak in POC function
        
        if plot:
            print("POC Max Peak: ", np.max(PC))
            plotSurface(PC)
            plt.figure()
            plt.imshow(PC,cmap='gray')
            plt.show()

    return xShift

def computeCCinter(imgL, imgR, plot=False):
    M = imgL.shape[0]
    dftG = (np.fft.fft2(imgL))
    dftG2 = (np.fft.fft2(imgR))
    cc = np.fft.ifft2(dftG*np.conjugate(dftG2))
    for i in range(int(M/2)):
            cc = np.roll(cc, -1, axis=1)
    for i in range(int(M/2)):
            cc = np.roll(cc, -1, axis=0)
    cc = cc.real
    idxMax = np.unravel_index(cc.argmax(), cc.shape)   
    dist = 4
    z = cc[idxMax[0]-dist:idxMax[0]+dist+1,idxMax[1]-dist:idxMax[1]+dist+1]
    x = np.linspace(idxMax[1]-dist,idxMax[1]+dist,z.shape[0])
    y = np.linspace(idxMax[0]-dist,idxMax[0]+dist,z.shape[1])
    X, Y = np.meshgrid(x, y)

    interp_spline = RectBivariateSpline(y, x, z, kx=4, ky=4)

    dx2, dy2 = 0.1, 0.1
    x2 = np.arange(x[0], x[-1], dx2)
    y2 = np.arange(y[0], y[-1], dy2)
    X2, Y2 = np.meshgrid(x2,y2)
    Z2 = interp_spline(y2, x2)

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X2, Y2, Z2,color='y',alpha=0.5)
        ax.plot(X.ravel(), Y.ravel(), z.ravel(), "ok")
        fig.tight_layout()
        plt.show()

    idxMax = np.unravel_index(Z2.argmax(), Z2.shape)  
    x_shift = int(M/2)-X2[idxMax[0],idxMax[1]]
    return x_shift
    
def parbola(y):
    dx = ((y[2])-(y[0]))/(2*(2*(y[1])-(y[2])-(y[0])))
    return dx

def computeCCinter2(imgL, imgR, plot=False):
    M = imgL.shape[0]
    dftG = (np.fft.fft2(imgL))
    dftG2 = (np.fft.fft2(imgR))
    cc = np.fft.ifft2(dftG*np.conjugate(dftG2))
    for i in range(int(M/2)):
            cc = np.roll(cc, -1, axis=1)

    cc = cc.real
    idxMax = np.unravel_index(cc.argmax(), cc.shape)   

    dist = 1
    y = np.array([cc[idxMax[0],idxMax[1]-dist], cc[idxMax[0],idxMax[1]], cc[idxMax[0],idxMax[1]+dist]])
    x = np.linspace(idxMax[1]-dist,idxMax[1]+dist,3).astype(int)

    x_shift = int(M/2)-parbola(y)-idxMax[1]

    if plot:
        temp = M
        xx = np.linspace(0,temp,temp).astype(int)  
        fig, axarr = plt.subplots(1,1)
        axarr.grid(True)
        axarr.plot(xx,cc[idxMax[0],:] )
        #axarr.plot(newX,p(newX))
        axarr.plot(x[0],y[0], marker="o", color="green")
        axarr.plot(x[1],y[1], marker="o", color="green")
        axarr.plot(x[2],y[2], marker="o", color="green")
        #axarr.text(maxPeak[0],maxPeak[1],str(x_shift),horizontalalignment='right')
        plt.show()
        
    return x_shift

def blockMatching(imgL, imgR,X_L, Y_L, M, N, method="TM_CCORR_NORMED", winFunc = "blackman"):

    # Creating a window around the feature
    winL = stereoImgCrop(imgL, X_L, Y_L, M, N)
    
    # Applying blackman window 
    if winFunc != "False":
        winL = windowFunc(winFunc,winL) #funcType: blackman, hanning,

    #**************NEEDS TO BE IMPROVED****************
    guess  = 155
    if(X_L-guess <= M/2):
        while(X_L-guess <= M/2):
            guess = guess-1
             
    x = X_L-guess 
    maxVal = []
    minVal = []
    xList = []
    
    while(x < X_L-150):
        winR = stereoImgCrop(imgR, x, Y_L, M, N) 
        
        if winFunc != "False":
            winR = windowFunc(winFunc,winR)

        if method == "TM_CCORR_NORMED":
            res = cv.matchTemplate(winR.astype(np.float32),winL.astype(np.float32),cv.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            maxVal.append(max_val)
        elif method == "TM_CCOEFF_NORMED":
            res = cv.matchTemplate(winR.astype(np.float32),winL.astype(np.float32),cv.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            maxVal.append(max_val)
        elif method == "TM_SQDIFF_NORMED":
            res = cv.matchTemplate(winR.astype(np.float32),winL.astype(np.float32),cv.TM_SQDIFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            minVal.append(min_val)
        elif method == "POC":
            dftL = np.fft.fft2(winL)
            dftR = np.fft.fft2(winR)
            POC = computePC(dftL,dftR)
            maxVal.append(np.max(POC))
        xList.append(x)
        x=x+1
    
    if method == "TM_SQDIFF_NORMED":
        #print("minVal: ", minVal[np.argmin(minVal)], " x: ", xList[np.argmin(minVal)])
        xShift = xList[np.argmin(minVal)]
    else:
        #print("maxVal: ", maxVal[np.argmax(maxVal)], " x: ", xList[np.argmax(maxVal)])
        xShift = xList[np.argmax(maxVal)]
    
    winR = stereoImgCrop(imgR, xShift, Y_L, M, N) 
   
    #f, axarr = plt.subplots(2,2)
    #axarr[0,0].imshow(winL,cmap='gray')
    #axarr[0,1].imshow(winR,cmap='gray')
    #plt.show()
    
    if winFunc != "False":
        winR = windowFunc(winFunc,winR)

    return winL, winR, xShift

def plotPhaseDifference(dftL,dftR,radius=0.6,rStepSize=2, cStepSize=2, aa=True):

    dftShiftL = np.fft.fftshift(dftL)
    dftShiftR = np.fft.fftshift(dftR)

    #Mask out spectral components that lie outside a radius from the central peak
    #maskedDftL = radMask(dftShiftL,radius)
    #maskedDftR = radMask(dftShiftR,radius)
    
    #Mask out spectral components for DFT's that have peaks with magnitudes less than a threshold value
    #maskedDftL = magMask(maskedDftL,magThres)
    #maskedDftR = magMask(maskedDftR,0)

    R = (dftShiftL*np.conjugate(dftShiftR))/(np.abs(dftShiftL*np.conjugate(dftShiftR)))  
    
    theta = np.arctan2(R.imag,R.real)  #Can also use theta = np.angle(R) 

    # Plot of phase component before removal of integer shift (figure 3)
    fig = plt.figure()
    xData, yData = np.mgrid[(-theta.shape[0]/2):(theta.shape[0]/2), (-(theta.shape[1])/2):(theta.shape[1]/2)]
    ax = fig.gca(projection='3d')
    ax.set_title('Phase differences')
    ax.plot_wireframe(xData,yData,theta,rstride=rStepSize, cstride=cStepSize,antialiased=aa)
    plt.show()

def plotSurface(func, wireframe=False,rStepSize=1, cStepSize=1, aa=True):
    #*************************Phase correlation surface plot*********************************
    xx, yy = np.mgrid[(-func.shape[0]/2):(func.shape[0]/2), (-(func.shape[1])/2):(func.shape[1]/2)]

    #Add fancy color mapping
    minn, maxx = func.min(), func.max()
    norm = matplotlib.colors.Normalize(minn, maxx)
    m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
    m.set_array([])
    fcolors = m.to_rgba(func)

    #create the figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    if not wireframe:
        ax.plot_surface(xx, yy, func , facecolors=fcolors,rstride=rStepSize, cstride=cStepSize,linewidth=0, antialiased=aa, shade=False)
    else:
        ax.plot_wireframe(xx, yy, func, rstride=rStepSize, cstride=cStepSize,antialiased=aa)

    plt.show()

def plotDFT(title, dft):
    M = dft.shape[0]
    fx= (np.linspace(-M/2,M/2-1,M)/M);    # FM = number of pixel in x direction 
    fx = fx *2 * np.pi                    # Multiply with 2pi to get the (spatial) frequency 
    fx = np.reshape(fx, (-1, 1))          # Go from shape (M,) to (M,1)
    fig = plt.figure()
    ax = fig.gca()
    ax.grid(True)
    ax.set_title(title)
    ax.plot(fx,abs(dft))
    plt.show()
