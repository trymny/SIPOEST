import numpy as np
import plotly.graph_objects as go
import time
import pandas as pd
import matplotlib.pyplot as plt
import icpLib 
import sys


def PCA(X,Y,Z):

    #Find covariance matrix
    A = np.array([
                    np.array([np.var(X), np.cov(X,Y,bias=True)[0][1], np.cov(X,Z,bias=True)[0][1]]),
                    np.array([np.cov(X,Y,bias=True)[0][1], np.var(Y), np.cov(Y,Z,bias=True)[0][1]]),
                    np.array([np.cov(X,Z,bias=True)[0][1], np.cov(Y,Z,bias=True)[0][1], np.var(Z)])
                ])

    #Compute SVD
    U, S, Vt = np.linalg.svd(A)
    
    eig1 = Vt[0][:] #Eigenvector 1
    eig2 = Vt[1][:] #Eigenvector 2
    eig3 = Vt[2][:] #Eigenvector 3

    R = Vt
    
    # Special reflection case
    if np.linalg.det(R) < 0:
       print("Special reflection case")
       Vt[2,:] *= -1
       R = Vt

    std = (np.sqrt(S))
    #print("var: ",S)
    #print("std: ", std)

    Pointcloud = np.array([X,Y,Z]).T

    centre = np.mean( Pointcloud, axis=0 )  # the centre of all the points
    Pointcloud = Pointcloud - centre  # shift the pointcloud so that it's centred at [0,0]
    
    # Long axis distance
    points1 = np.dot( Pointcloud, eig1 )
    Lo1 = points1.min() * eig1+centre
    Hi1 = points1.max() * eig1+centre
    #dx = round(np.sqrt((Lo1[0]-Hi1[0])**2+(Lo1[1]-Hi1[1])**2+(Lo1[2]-Hi1[2])**2),3) #Length of the fish
    dx = std[0]*3.3

    # Short axis distance
    points2 = np.dot( Pointcloud, eig2 )
    Lo2 = points2.min() * eig2+centre
    Hi2 = points2.max() * eig2+centre
    #dy = round(np.sqrt((Lo2[0]-Hi2[0])**2+(Lo2[1]-Hi2[1])**2+(Lo2[2]-Hi2[2])**2),3)  #Height
    dy = std[1]*4.0

    # Depth axis distance
    points3 = np.dot( Pointcloud, eig3 )
    Lo3 = points3.min() * eig3+centre
    Hi3 = points3.max() * eig3+centre
    #dz = round(np.sqrt((Lo3[0]-Hi3[0])**2+(Lo3[1]-Hi3[1])**2+(Lo3[2]-Hi3[2])**2),3)  #Thickness
    dz = std[2]*4

    scale = np.array([dx,dy,dz])

    #print("Centre: ", centre)
    return Vt,S,R, scale,centre,np.array([Lo1, Hi1, Lo2, Hi2,Lo3, Hi3])

FRAME_NO = int(sys.argv[1]) #564 #554
FILE_NAME = 'F'+str(FRAME_NO)
data = np.loadtxt('data/experiments/'+FILE_NAME+'.txt')
#data = np.loadtxt('data/1000/trajectories/caudalFin1.txt')
#data = np.loadtxt('data/1000/trajectories/caudalFin2.txt')

s = 0
e = len(data[:,0])
print(len(data[:,0]))
intensity = data[s:e, 0]
X = data[s:e, 1]
Y = data[s:e, 2]
Z = data[s:e, 3]

tic = time.process_time()

#--------------------------PCA---------------------------
evecs,evals,rotMat,scale,centre,test = PCA(X,Y,Z)
#--------------------------------------------------------

#-------------------Define inital guess------------------
# Set of all spherical angles:
d = 1.5
u = np.linspace(0, np.pi,30)
v = np.linspace(d, np.pi+d, 30)
#u = np.linspace(0, 2*np.pi, 25)
#v = np.linspace(0, np.pi, 25)

# Cartesian coordinates that correspond to the spherical angles:
xx = scale[0]/2 * np.outer(np.cos(u), np.sin(v))
yy = scale[1]/2 * np.outer(np.sin(u), np.sin(v))
zz = scale[2]/2 * np.outer(np.ones_like(u), np.cos(v))

pc2 = np.array([xx.flatten(),yy.flatten(),zz.flatten()]).T 

H = np.identity(4)
H[0:3, 0:3] = rotMat
H[0:3, 3] = centre

pc2 = pc2@rotMat

xInit = pc2[:,0]+centre[0]
yInit = pc2[:,1]+centre[1]
zInit = pc2[:,2]+centre[2]
  
#--------------------ICP algorithm------------------------
pc1 = np.array([X,Y,Z]).T

T,_,_ = icpLib.icp(pc2,pc1,init_pose=H,max_iterations=200, tolerance=10e-15)
R = T[0:3,0:3]
t = T[0:3,3:4]

pc2 = pc2@R

xFitted = pc2[:,0]+t[0]
yFitted = pc2[:,1]+t[1]
zFitted = pc2[:,2]+t[2]

toc = time.process_time()

#----------------------------------------------------------

#--------------------Create plotly plot------_-------------
grayStyle = dict(size=1,color=intensity,colorscale='gray',opacity=0.8)
blueStyle = dict(size=1,color='blue',opacity=0.8)
redStyle = dict(size=1,color='red',opacity=0.8)
greenStyle = dict(size=1,color='green',opacity=0.8)

bluePoint = dict(size=5,color='blue',opacity=0.8)
redPoint = dict(size=5,color='red',opacity=0.8)
greenPoint = dict(size=5,color='green',opacity=0.8)
fig = go.Figure(data=[go.Scatter3d(x=X,y=Y,z=Z,mode='markers',marker=grayStyle)])

data = fig._data

icpPlot =go.Scatter3d(x=xFitted,y=yFitted,z=zFitted,mode='markers',marker=redStyle)
initGuessPlot =go.Scatter3d(x=xInit,y=yInit,z=zInit,mode='markers',marker=blueStyle)
lengthPoints = go.Scatter3d(x=test[0:2,0],y=test[0:2,1],z=test[0:2,2],mode='markers',marker=redPoint)
heightPoints = go.Scatter3d(x=test[2:4,0],y=test[2:4,1],z=test[2:4,2],mode='markers',marker=greenPoint)
depthPoints = go.Scatter3d(x=test[4:,0],y=test[4:,1],z=test[4:,2],mode='markers',marker=bluePoint)


data.append(icpPlot)
#data.append(initGuessPlot)
''' 
data.append(lengthPoints)
data.append(heightPoints)
data.append(depthPoints)
'''

fig = go.Figure(data=data)
fig.show()

#----------------------------------------------------------

#---------------------Print information--------------------
print("Comp time: ", round(toc - tic,3),"s")

volume = (4/3)*np.pi*(scale[0]/2)*(scale[1]/2)*(scale[2]/2)
#Density of salt water 1.03 g/cm??
density  = 1.03*1000 #kg/m??   

print("\nSize information extracted from fitted ellipsoid")
print("Volume: ", round(volume,5),"m??")
print("Length: ", round(scale[0]*100,3),"cm")
print("Mass: ", round(volume*density,3),"kg")

header = ['W','L'] 
df = pd.read_csv("salmonSize.csv", usecols=header, delim_whitespace=True) 
sizeData = df.to_numpy()

yData = sizeData[:,0]
xData = sizeData[:,1]
coef = np.polyfit(xData,yData,1)
poly1d_fn = np.poly1d(coef)

print("\nGROUND TRUTH (Linear regression based on physical size measurements of atlantic salmon)")
print("Ground truth mass: ", round(poly1d_fn(scale[0]*100),3), "kg")
print("Error: ", round(abs(volume*density-poly1d_fn(scale[0]*100)),3), "kg")


with open("data/result.txt", "w") as c:
    positionStr =  str(round(volume,5)) +' '+str(round(scale[0]*100,3)) +' '+ str(round(volume*density,3)) +' '+ str(round(poly1d_fn(scale[0]*100),3))
    print(positionStr, file=c, flush=True)

''' 
fig = plt.figure(figsize=(18, 16)) 
ax = fig.gca()
ax.set_ylabel('Weight [kg]',fontsize=20)
ax.set_xlabel('Length [cm]',fontsize=20)
ax.grid(True)
ax.plot(xData,yData, 'yo', markersize=14)
ax.plot(xData, poly1d_fn(xData), 'k') 
plt.show()
'''
#----------------------------------------------------------