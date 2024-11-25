#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import numpy as np
from BabelViscoFDTD.H5pySimple import ReadFromH5py,SaveToH5py
from BabelViscoFDTD.PropagationModel import PropagationModel

PModel=PropagationModel()


# ### Preamble
# Please consult first the example in the `1 - Flat source homgenous medium.ipynb` notebook for the basics how to run a simulation
# 
# 
# In this example we will cover the importance and flexibility of specifying the directivity of the particle displacement in the source
# 
# 
# # 1 - Source oriented in $Z$ direction
# We repeat the simulation as in `1 - Flat source homgenous medium.ipynb` example

# In[8]:


Frequency = 350e3  # Hz
MediumSOS = 1500 # m/s - water
MediumDensity=1000 # kg/m3

ShortestWavelength =MediumSOS / Frequency
SpatialStep =ShortestWavelength / 8.0 # A minimal step of 6 is recommnded
Amplitude=100e3/MediumDensity/MediumSOS #100 kPa

DimDomain =  np.array([0.05,0.05,0.1])  # in m, x,y,z

TxDiam = 0.03 # m, circular piston
TxPlaneLocation = 0.01  # m , in XY plane at Z = 0.01 m

PMLThickness = 12 # grid points for perect matching layer, HIGHLY RECOMMENDED DO NOT CHANGE THIS SIZE 
ReflectionLimit= 1.0000e-05 #reflection parameter for PML, IGHLY RECOMMENDED DO NOT CHANGE THIS VALUE

N1=int(np.ceil(DimDomain[0]/SpatialStep)+2*PMLThickness)
N2=int(np.ceil(DimDomain[1]/SpatialStep)+2*PMLThickness)
N3=int(np.ceil(DimDomain[2]/SpatialStep)+2*PMLThickness)
print('Domain size',N1,N2,N3)
TimeSimulation=np.sqrt(DimDomain[0]**2+DimDomain[1]**2+DimDomain[2]**2)/MediumSOS #time to cross one corner to another
TemporalStep=1e-7 # if this step is too coarse a warning will be generated (but simulation will continue,) 

MaterialMap=np.zeros((N1,N2,N3),np.uint32) # note the 32 bit size
MaterialList=np.zeros((1,5)) # one material in this examples
MaterialList[0,0]=MediumDensity # water density
MaterialList[0,1]=MediumSOS # water SoS
#all other parameters are set to 0 
COMPUTING_BACKEND=3 # 0 for CPU, 1 for CUDA, 2 for OpenCL, 3 for Metal
DefaultGPUDeviceName='M1' # ID of GPU


# In[9]:


def MakeCircularSource(DimX,DimY,SpatialStep,Diameter):
    #simple defintion of a circular source centred in the domain
    XDim=np.arange(DimX)*SpatialStep
    YDim=np.arange(DimY)*SpatialStep
    XDim-=XDim.mean()
    YDim-=YDim.mean()
    XX,YY=np.meshgrid(XDim,YDim)
    MaskSource=(XX**2+YY**2)<=(Diameter/2.0)**2
    return (MaskSource*1.0).astype(np.uint32)

SourceMask=MakeCircularSource(N1,N2,SpatialStep,TxDiam)
plt.imshow(SourceMask,cmap=plt.cm.gray);
plt.title('Circular source map')

SourceMap=np.zeros((N1,N2,N3),np.uint32)
LocZ=int(np.round(TxPlaneLocation/SpatialStep))+PMLThickness
SourceMap[:,:,LocZ]=SourceMask 

Ox=np.zeros((N1,N2,N3))
Oy=np.zeros((N1,N2,N3))
Oz=np.zeros((N1,N2,N3))
Oz[SourceMap>0]=1 #only Z has a value of 1


# In[10]:


LengthSource=4.0/Frequency #we will use 4 pulses
TimeVectorSource=np.arange(0,LengthSource+TemporalStep,TemporalStep)

PulseSource = np.sin(2*np.pi*Frequency*TimeVectorSource)
plt.figure()
plt.plot(TimeVectorSource*1e6,PulseSource)
plt.title('4-pulse signal')

#note we need expressively to arrange the data in a 2D array
PulseSource=np.reshape(PulseSource,(1,len(TimeVectorSource))) 


# In[11]:


SensorMap=np.zeros((N1,N2,N3),np.uint32)

SensorMap[PMLThickness:-PMLThickness,int(N2/2),PMLThickness:-PMLThickness]=1

plt.figure()
plt.imshow(SensorMap[:,int(N2/2),:].T,cmap=plt.cm.gray)
plt.title('Sensor map location');


# In[12]:


Sensor,LastMap,DictRMSValue,InputParam=PModel.StaggeredFDTD_3D_with_relaxation(
                                                         MaterialMap,
                                                         MaterialList,
                                                         Frequency,
                                                         SourceMap,
                                                         PulseSource,
                                                         SpatialStep,
                                                         TimeSimulation,
                                                         SensorMap,
                                                         Ox=Ox*Amplitude,
                                                         Oy=Oy*Amplitude,
                                                         Oz=Oz*Amplitude,
                                                         NDelta=PMLThickness,
                                                         ReflectionLimit=ReflectionLimit,
                                                         COMPUTING_BACKEND=COMPUTING_BACKEND,
                                                         USE_SINGLE=True,
                                                         SelRMSorPeak=2, #we select now only peak data
                                                         DT=TemporalStep,
                                                         QfactorCorrection=True,
                                                         SelMapsRMSPeakList=['Pressure'],
                                                         SelMapsSensorsList=['Vx','Vy','Vz','Pressure'],
                                                         DefaultGPUDeviceName=DefaultGPUDeviceName,
                                                         TypeSource=0)


# ### Ploting data

# In[13]:


RMSValue=DictRMSValue['Pressure']
RMSValue[:,:,LocZ-2:LocZ+2]=0 # we hide the values too close to the source 
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(RMSValue[:,int(N2/2),:].T/1e6,cmap=plt.cm.jet)
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(RMSValue[:,:,125].T/1e6,cmap=plt.cm.jet)
plt.colorbar()


# We calculate now peak and RMS values from

# In[14]:


#To remain compatible with Matlab (whcih uses a Fortran convention for arrays, the index need to be rebuilt)
MaxSensorPlane=np.zeros((N1,N3))
RMSSensorPlane=np.zeros((N1,N3))

ii,jj,kk=np.unravel_index(InputParam['IndexSensorMap']-1, SensorMap.shape, order='F')
assert(np.all(jj==N2/2))

for s in ['Vx','Vy','Vz','Pressure']:
    #We use the IndexSensorMap array that was used in the low level function to 
    for n, i,j,k in zip(range(len(InputParam['IndexSensorMap'])),ii,jj,kk):
        if i==int(N1/2) and k==int(N3/2):
            CentralPoint=n #we save this to later plot the time signal at the center
        MaxSensorPlane[i,k]=np.max(Sensor[s][n,:])
        RMSSensorPlane[i,k]=np.sqrt(1./len(Sensor[s][n,:])*np.sum(Sensor[s][n,:]**2))
    if 'Pressure' == s:
        #convert to MPa
        MaxSensorPlane/=1e6
        RMSSensorPlane/=1e6
    MaxSensorPlane[:,LocZ-2:LocZ+2]=0
    RMSSensorPlane[:,LocZ-2:LocZ+2]=0

    plt.figure(figsize=(14,8))
    plt.subplot(1,3,1)
    plt.imshow(MaxSensorPlane.T,cmap=plt.cm.jet)
    plt.title('Peak value')
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.imshow(RMSSensorPlane.T,cmap=plt.cm.jet)
    plt.title('RMS value')
    plt.colorbar()
    plt.subplot(1,3,3)
    plt.plot(Sensor['time']*1e6,Sensor[s][CentralPoint])
    plt.xlabel('time ($\mu$s)')
    plt.ylabel('$'+s[0]+'_'+s[1]+'$')
    plt.title('Time signal at central point')
    plt.suptitle('Plots for $'+s[0]+'_'+s[1]+'$')
    plt.tight_layout()


# ## 1.a - Changing particle direction in previous example - bad direction
# For purposes of illustration, we will change the particle displacmentdirection to $Y$. For many cases, this would be for the most an *erroneous* setting. We will repeat the simulation with all the other parameters as before.

# In[18]:


BadOx=np.zeros((N1,N2,N3))
BadOy=np.zeros((N1,N2,N3))
BadOz=np.zeros((N1,N2,N3))
BadOy[SourceMap>0]=1 #only Y has a value of 1


# In[19]:


Sensor,LastMap,DictRMSValue,InputParam=PModel.StaggeredFDTD_3D_with_relaxation(
                                                         MaterialMap,
                                                         MaterialList,
                                                         Frequency,
                                                         SourceMap,
                                                         PulseSource,
                                                         SpatialStep,
                                                         TimeSimulation,
                                                         SensorMap,
                                                         Ox=BadOx*Amplitude, #We use now the wrong directivity
                                                         Oy=BadOy*Amplitude, #We use now the wrong directivity
                                                         Oz=BadOz*Amplitude, #We use now the wrong directivity
                                                         NDelta=PMLThickness,
                                                         ReflectionLimit=ReflectionLimit,
                                                         COMPUTING_BACKEND=COMPUTING_BACKEND,
                                                         USE_SINGLE=True,
                                                         SelRMSorPeak=2, #we select now only peak data
                                                         DT=TemporalStep,
                                                         QfactorCorrection=True,
                                                         SelMapsRMSPeakList=['Pressure'],
                                                         SelMapsSensorsList=['Vx','Vy','Vz','Pressure'],
                                                         DefaultGPUDeviceName=DefaultGPUDeviceName,
                                                         TypeSource=0)


# ### Ploting data

# In[20]:


RMSValue=DictRMSValue['Pressure']/1e6
RMSValue[:,:,LocZ-2:LocZ+2]=0 # we hide the values too close to the source 
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(RMSValue[:,int(N2/2),:].T,cmap=plt.cm.jet)
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(RMSValue[:,:,125].T,cmap=plt.cm.jet)
plt.colorbar();


# We can see the pressure map got completely innacurate because the mistake in the directivity arrays. 
# 
# We calculate now peak and RMS values from sensor data

# In[21]:


#To remain compatible with Matlab (whcih uses a Fortran convention for arrays, the index need to be rebuilt)
MaxSensorPlane=np.zeros((N1,N3))
RMSSensorPlane=np.zeros((N1,N3))

ii,jj,kk=np.unravel_index(InputParam['IndexSensorMap']-1, SensorMap.shape, order='F')
assert(np.all(jj==N2/2))

for s in ['Vx','Vy','Vz','Pressure']:
    #We use the IndexSensorMap array that was used in the low level function to 
    for n, i,j,k in zip(range(len(InputParam['IndexSensorMap'])),ii,jj,kk):
        if i==int(N1/2) and k==int(N3/2):
            CentralPoint=n #we save this to later plot the time signal at the center
        MaxSensorPlane[i,k]=np.max(Sensor[s][n,:])
        RMSSensorPlane[i,k]=np.sqrt(1./len(Sensor[s][n,:])*np.sum(Sensor[s][n,:]**2))
    if 'Pressure' == s:
        #convert to MPa
        MaxSensorPlane/=1e6
        RMSSensorPlane/=1e6
    MaxSensorPlane[:,LocZ-2:LocZ+2]=0
    RMSSensorPlane[:,LocZ-2:LocZ+2]=0

    plt.figure(figsize=(14,8))
    plt.subplot(1,3,1)
    plt.imshow(MaxSensorPlane.T,cmap=plt.cm.jet)
    plt.title('Peak value')
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.imshow(RMSSensorPlane.T,cmap=plt.cm.jet)
    plt.title('RMS value')
    plt.colorbar()
    plt.subplot(1,3,3)
    plt.plot(Sensor['time']*1e6,Sensor[s][CentralPoint])
    plt.xlabel('time ($\mu$s)')
    plt.ylabel('$'+s[0]+'_'+s[1]+'$')
    plt.title('Time signal at central point')
    plt.suptitle('Plots for $'+s[0]+'_'+s[1]+'$')
    plt.tight_layout()


# -----
# We can clearly see this field was not as intended to model an flat circular source
# 
# # 2 - Rotated source
# We will now rotate the source 45 degrees. We will need to increase a bit the domain size.

# In[22]:


DimDomain =  np.array([0.08,0.05,0.08])  # in m, x,y,z

TxDiam = 0.03/np.sqrt(2) # m, circular piston , as it will be rotated 45 degrees, we will make it a bit shorter to 
TxPlaneLocation = 0.01  # m , in XY plane at Z = 0.01 m

PMLThickness = 12 # grid points for perect matching layer, HIGHLY RECOMMENDED DO NOT CHANGE THIS SIZE 
ReflectionLimit= 1.0000e-05 #reflection parameter for PML, IGHLY RECOMMENDED DO NOT CHANGE THIS VALUE

N1=int(np.ceil(DimDomain[0]/SpatialStep)+2*PMLThickness)
N2=int(np.ceil(DimDomain[1]/SpatialStep)+2*PMLThickness)
N3=int(np.ceil(DimDomain[2]/SpatialStep)+2*PMLThickness)
print('Domain size',N1,N2,N3)
TimeSimulation=np.sqrt(DimDomain[0]**2+DimDomain[1]**2+DimDomain[2]**2)/MediumSOS #time to cross one corner to another
TemporalStep=1e-7 # if this step is too coarse a warning will be generated (but simulation will continue,) 

MaterialMap=np.zeros((N1,N2,N3),np.uint32) # note the 32 bit size
MaterialList=np.zeros((1,5)) # one material in this examples
MaterialList[0,0]=MediumDensity # water density
MaterialList[0,1]=MediumSOS # water SoS
#all other parameters are set to 0 


# In[23]:


SourceMask=MakeCircularSource(N1,N2,SpatialStep,TxDiam).T
plt.figure(figsize=(8,5))
plt.subplot(1,2,1)
plt.imshow(SourceMask.T,cmap=plt.cm.gray);
plt.title('Circular source map')

SourceMap=np.zeros((N1,N2,N3),np.uint32) #note this time we will use float type, later we will convert it back to uint32
LocZ=int(np.round(TxPlaneLocation/SpatialStep))+PMLThickness
SourceMap[:,:,LocZ]=SourceMask 

plt.subplot(1,2,2)
plt.imshow(SourceMap[:,int(N2/2),:].T,cmap=plt.cm.gray);
plt.title('Central cut of 3D source map');


# We will rotate plane by plane using OpenCV

# In[24]:


from scipy import ndimage
for n in range(N2):
    SourceMap[:,n,:]=np.roll(np.roll(ndimage.rotate(SourceMap[:,n,:],45,mode='nearest',reshape=False),-10,axis=1),10,axis=0)
plt.imshow(SourceMap[:,int(N2/2),:].T,cmap=plt.cm.gray);


# Now we just create a vector for the particle displacement oriented in the right direction

# In[25]:


Ox=np.zeros((N1,N2,N3))
Oy=np.zeros((N1,N2,N3))
Oz=np.zeros((N1,N2,N3))
Vector45XZ=[-1,0,1]
Vector45XZ/=np.linalg.norm(Vector45XZ)
print('Vector45XZ',Vector45XZ)
Ox[SourceMap>0]=Vector45XZ[0]
Oy[SourceMap>0]=Vector45XZ[1]
Oz[SourceMap>0]=Vector45XZ[2]


# In[26]:


SensorMap=np.zeros((N1,N2,N3),np.uint32)

SensorMap[PMLThickness:-PMLThickness,int(N2/2),PMLThickness:-PMLThickness]=1

plt.figure()
plt.imshow(SensorMap[:,int(N2/2),:].T,cmap=plt.cm.gray)
plt.title('Sensor map location');


# In[27]:


Sensor,LastMap,DictRMSValue,InputParam=PModel.StaggeredFDTD_3D_with_relaxation(
                                                         MaterialMap,
                                                         MaterialList,
                                                         Frequency,
                                                         SourceMap,
                                                         PulseSource,
                                                         SpatialStep,
                                                         TimeSimulation,
                                                         SensorMap,
                                                         Ox=Ox*Amplitude, 
                                                         Oy=Oy*Amplitude, 
                                                         Oz=Oz*Amplitude, 
                                                         NDelta=PMLThickness,
                                                         ReflectionLimit=ReflectionLimit,
                                                         COMPUTING_BACKEND=COMPUTING_BACKEND,
                                                         USE_SINGLE=True,
                                                         SelRMSorPeak=2, #we select now only peak data
                                                         DT=TemporalStep,
                                                         QfactorCorrection=True,
                                                         SelMapsRMSPeakList=['Pressure'],
                                                         SelMapsSensorsList=['Vx','Vy','Vz','Pressure'],
                                                         DefaultGPUDeviceName=DefaultGPUDeviceName,
                                                         TypeSource=0)


# ### Ploting data

# In[28]:


RMSValue=DictRMSValue['Pressure']/1e6
for n in range(-2,3):
    for m in range(-2,3):
        RMSValue[np.roll(np.roll(SourceMap>0,n,axis=0),m,axis=2)]=0. #we turn off the values close the source 
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(RMSValue[:,int(N2/2),:].T,cmap=plt.cm.jet)
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(RMSValue[:,:,125].T,cmap=plt.cm.jet)
plt.colorbar();


# We can see a similar acoustic field as the first test above , just rotated 45 degrees.
# 
# We can see now the individual vector components of the displacements

# In[29]:


#To remain compatible with Matlab (whcih uses a Fortran convention for arrays, the index need to be rebuilt)
MaxSensorPlane=np.zeros((N1,N3))
RMSSensorPlane=np.zeros((N1,N3))

ii,jj,kk=np.unravel_index(InputParam['IndexSensorMap']-1, SensorMap.shape, order='F')
assert(np.all(jj==N2/2))

for s in ['Vx','Vy','Vz','Pressure']:
    #We use the IndexSensorMap array that was used in the low level function to 
    for n, i,j,k in zip(range(len(InputParam['IndexSensorMap'])),ii,jj,kk):
        if i==int(N1/2) and k==int(N3/2):
            CentralPoint=n #we save this to later plot the time signal at the center
        MaxSensorPlane[i,k]=np.max(Sensor[s][n,:])
        RMSSensorPlane[i,k]=np.sqrt(1./len(Sensor[s][n,:])*np.sum(Sensor[s][n,:]**2))
    if 'Pressure' == s:
        #convert to MPa
        MaxSensorPlane/=1e6
        RMSSensorPlane/=1e6
    MaxSensorPlane[:,LocZ-2:LocZ+2]=0
    RMSSensorPlane[:,LocZ-2:LocZ+2]=0

    plt.figure(figsize=(14,8))
    plt.subplot(1,3,1)
    plt.imshow(MaxSensorPlane.T,cmap=plt.cm.jet)
    plt.title('Peak value')
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.imshow(RMSSensorPlane.T,cmap=plt.cm.jet)
    plt.title('RMS value')
    plt.colorbar()
    plt.subplot(1,3,3)
    plt.plot(Sensor['time']*1e6,Sensor[s][CentralPoint])
    plt.xlabel('time ($\mu$s)')
    plt.ylabel('$'+s[0]+'_'+s[1]+'$')
    plt.title('Time signal at central point')
    plt.suptitle('Plots for $'+s[0]+'_'+s[1]+'$')
    plt.tight_layout()


# We can see now how the field is split mainly in X and Z components

# ### 2.a - Assigning again  bad direction
# We repeat the exercise of assigning a wrong orientation to the particles

# In[30]:


BadOx=np.zeros((N1,N2,N3))
BadOy=np.zeros((N1,N2,N3))
BadOz=np.zeros((N1,N2,N3))
BadOy[SourceMap>0]=1 #only Y has a value of 1


# In[31]:


SensorBad,LastMap,DictRMSValueBad,InputParam=PModel.StaggeredFDTD_3D_with_relaxation(
                                                         MaterialMap,
                                                         MaterialList,
                                                         Frequency,
                                                         SourceMap,
                                                         PulseSource,
                                                         SpatialStep,
                                                         TimeSimulation,
                                                         SensorMap,
                                                         Ox=BadOx*Amplitude, #We use now the wrong directivity
                                                         Oy=BadOy*Amplitude, #We use now the wrong directivity
                                                         Oz=BadOz*Amplitude, #We use now the wrong directivity
                                                         NDelta=PMLThickness,
                                                         ReflectionLimit=ReflectionLimit,
                                                         COMPUTING_BACKEND=COMPUTING_BACKEND,
                                                         USE_SINGLE=True,
                                                         DT=TemporalStep,
                                                         QfactorCorrection=True,
                                                         SelMapsRMSPeakList=['Pressure'],
                                                         SelMapsSensorsList=['Vx','Vy','Vz','Pressure'],
                                                         SensorSubSampling=2,
                                                         DefaultGPUDeviceName=DefaultGPUDeviceName,
                                                         TypeSource=0)


# In[32]:


RMSValueBad=DictRMSValueBad['Pressure']
for n in range(-2,3):
    for m in range(-2,3):
        RMSValueBad[np.roll(np.roll(SourceMap>0,n,axis=0),m,axis=2)]=0. #we turn off the values close the source 
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(RMSValueBad[:,int(N2/2),:].T,cmap=plt.cm.jet)
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(RMSValueBad[:,:,125].T,cmap=plt.cm.jet)
plt.colorbar();


# We can see again that the wrong assigment to the directivity produces an inacurate field

# In[33]:


#To remain compatible with Matlab (whcih uses a Fortran convention for arrays, the index need to be rebuilt)
MaxSensorPlane=np.zeros((N1,N3))
RMSSensorPlane=np.zeros((N1,N3))

for s in ['Vx','Vy','Vz']:
    #We use the IndexSensorMap array that was used in the low level function to 
    for n, index in enumerate( InputParam['IndexSensorMap']): 
        k=int(index/(N1*N2))
        j=int(index%(N1*N2))
        i=int(j%N1)
        j=int(j/N1)
        assert(j==N2/2) #all way up we specified the XZ plane at N2/2, this assert should pass
        if i==int(N1/2) and k==int(N3/2):
            CentralPoint=n #we save this to later plot the time signal at the center
        MaxSensorPlane[i,k]=np.max(SensorBad[s][n,:])
        RMSSensorPlane[i,k]=np.sqrt(1./len(SensorBad[s][n,:])*np.sum(SensorBad[s][n,:]**2))
        
    for n in range(-2,3):
        for m in range(-2,3):
            MaxSensorPlane[np.roll(np.roll(SourceMap[:,j,:]>0,n,axis=0),m,axis=1)]=0. 
            RMSSensorPlane[np.roll(np.roll(SourceMap[:,j,:]>0,n,axis=0),m,axis=1)]=0.

    plt.figure(figsize=(14,6))
    plt.subplot(1,3,1)
    plt.imshow(MaxSensorPlane.T,cmap=plt.cm.jet)
    plt.title('Peak value')
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.imshow(RMSSensorPlane.T,cmap=plt.cm.jet)
    plt.title('RMS value')
    plt.colorbar()
    plt.subplot(1,3,3)
    plt.plot(SensorBad['time']*1e6,SensorBad[s][CentralPoint])
    plt.xlabel('time ($\mu$s)')
    plt.ylabel('$'+s[0]+'_'+s[1]+'$')
    plt.title('Time signal at central point')
    plt.suptitle('Plots for $'+s[0]+'_'+s[1]+'$')
    plt.tight_layout()


# In[ ]:




