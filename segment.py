import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def ellipse_center(a):
        b,c,d,f,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[0]
        num = b*b-a*c
        x0=(c*d-b*f)/num
        y0=(a*f-b*d)/num
        return int(x0), int(y0)

def ellipse_axis_length(X,Y):
    
    #Find covariance matrix
    A = np.cov(X,Y) 

    #Compute SVD
    U, s, V = np.linalg.svd(A)
    
    eig1 = U[0][:] #Eigenvector 1
    eig2 = U[1][:] #Eigenvector 2

    Pointcloud = np.array([X,Y]).T

    center = np.mean( Pointcloud, axis=0 )  # the centre of all the points
    Pointcloud = Pointcloud - center  # shift the cloud to be centred at [0 0]

    # distances along the long axis t * v1
    Dist1 = np.dot( Pointcloud, eig1 )
    Lo1 = Dist1.min() * eig1+center
    Hi1 = Dist1.max() * eig1+center
    # and along the short axis t * v2 
    Dist2 = np.dot( Pointcloud, eig2 )
    Lo2 = Dist2.min() * eig2+center
    Hi2 = Dist2.max() * eig2+center


    dx = round(np.sqrt((Lo1[0]-Hi1[0])**2+(Lo1[1]-Hi1[1])**2),3)
    dy = round(np.sqrt((Lo2[0]-Hi2[0])**2+(Lo2[1]-Hi2[1])**2),3)

    return dx/2, dy/2,U

def inellipse(x, y, xc, yc,a, b, R):
    xr = x - xc
    yr = y - yc
    x0 = R[0,0]*xr +  R[0,1]*yr 
    y0 = R[1,0]*xr +  R[1,1]*yr
    p = x0**2 / a**2 + y0**2 / b**2

    if(p < 1):
        return True
    else:
        return False

def countours(img,threshold,dist,x,y,M,N):
    imgC = img[int(y):int(N), int(x):int(M)]

    alpha = 2.0
    beta = -0.0

    imgC = alpha * imgC + beta
    imgC = np.clip(imgC, 0, 255).astype(np.uint8)
    imgC =  cv.bilateralFilter(imgC,11,75,75)

    # Detect edges using Canny
    canny_output = cv.Canny(imgC, threshold, threshold * 2)
    # Find contours
    contours, hierarchy = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Draw contours
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    xData = np.array([])
    yData = np.array([])

    for i in range(len(contours)):
        d  = 0
        for k in range(len(contours[i])):
            if(k < len(contours[i])-1):
                d = d + np.sqrt((contours[i][k][0][0]-contours[i][k+1][0][0])**2+(contours[i][k][0][1]-contours[i][k+1][0][1])**2)
            avgD = d/len(contours[i])

        if(avgD > dist/10):
            for j in range(len(contours[i])):
                xData = np.append(xData,contours[i][j][0][0])
                yData = np.append(yData,contours[i][j][0][1])
                cv.drawContours(drawing, contours, i, (255, 255, 255), 2, cv.LINE_8, hierarchy, 0)

    X = xData.reshape(-1, 1)
    Y = yData.reshape(-1, 1)
    
    A = np.hstack([X**2, X * Y, Y**2, X, Y])
    b = np.ones_like(X)
    x = np.linalg.lstsq(A, b, rcond=False)[0].squeeze()

    cX,cY = ellipse_center(x)

    X_coord, Y_coord = np.meshgrid(np.linspace(0,500,500), np.linspace(0,500,500))
    Z_coord = x[0] * X_coord ** 2 + x[1] * X_coord * Y_coord + x[2] * Y_coord**2 + x[3] * X_coord + x[4] * Y_coord
    c = plt.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=2)
    plt.close()
    data= c.allsegs[0][0]
    #plt.plot(dat0[:,0],dat0[:,1])
    #plt.show()
    for i in range(len(data[:,0])):
        drawing = cv.circle(drawing, (int(data[i,0]),int(data[i,1])), radius=0, color=(255, 45, 255), thickness=-1)
    drawing = cv.circle(drawing, (cX,cY), radius=2, color=(0, 0, 255), thickness=-1)

    res1,res2,R = ellipse_axis_length((data[:,0]).astype(int),(data[:,1]).astype(int))

    return drawing,imgC,data,cX,cY,R,res1,res2

def dis2point(x,y,disp):
    
    vec_tmp = np.array([x,y,disp,1.0])
    vec_tmp = np.reshape(vec_tmp, (4,1))

    globQ = np.array([
                        np.array([1, 0, 0, -565.3385467529297]),
                       np.array([0, 1, 0, -410.1333084106445]),
                        np.array([0, 0, 0, 1289.887332633299]),
                        np.array([0, 0, 0.006655545399213341, -0.1802197918186371])
                    ])

    vec_tmp = globQ@vec_tmp
    vec_tmp = np.reshape(vec_tmp, (1,4))[0]
    x = vec_tmp[0]
    y = vec_tmp[1]
    z = vec_tmp[2]
    w = vec_tmp[3]

    point = [x,y,z]/w

    return point # this is the 3D point