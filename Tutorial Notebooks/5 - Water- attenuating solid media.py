from matplotlib import pyplot as plt
import numpy as np
from BabelViscoFDTD.H5pySimple import ReadFromH5py,SaveToH5py
from BabelViscoFDTD.PropagationModel import PropagationModel
from scipy import ndimage
from skimage.draw import circle_perimeter,disk
from skimage.transform import rotate
from ipywidgets import  HBox, VBox, GridBox, Layout, Label, IntSlider, interact,\
    Checkbox,GridspecLayout,interactive, widgets, FloatSlider
from IPython.display import display
PModel=PropagationModel()
COMPUTING_BACKEND=3 # 0 for CPU, 1 for CUDA, 2 for OpenCL, 3 for Metal
DefaultGPUDeviceName='M1' # ID of GPU

Frequency=1e6 # Hz
WaterSOS = 1500 # m/s - water
WaterDensity=1000 # kg/m3

# Skull properties (solid media mimicking skull)
SolidLSOS = 2800  # Speed of sound in solid (skull) longitudinal waves (m/s)
SolidsSOS = 1026  # Speed of sound in solid (skull) shear waves (m/s)
SolidDensity = 1850  # Density of skull (kg/m^3)
SolidLAlpha = 2.7  # Longitudinal attenuation (Np/m)
SolidSAlpha = 3.0  # Shear attenuation (Np/m)

ShortestWavelength =SolidsSOS / Frequency
SpatialStep =ShortestWavelength / 8.0 # A minimal step of 6 is recommnded

DimDomain =  np.array([0.1,0.05,0.09])  # in m, x,y,z  # in m, x,y,z

TxRadius = 0.04 # m, 
TxDiameter=0.04
PMLThickness = 12 # grid points for perect matching layer, 
ReflectionLimit= 1.0000e-05 #reflection parameter for PML, 

Amplitude= 60e3/WaterSOS/WaterDensity

N1=int(np.ceil(DimDomain[0]/SpatialStep)+2*PMLThickness)
N2=int(np.ceil(DimDomain[1]/SpatialStep)+2*PMLThickness)
N3=int(np.ceil(DimDomain[2]/SpatialStep)+2*PMLThickness)
print('Domain size',N1,N2,N3)
TimeSimulation=np.sqrt(DimDomain[0]**2+DimDomain[1]**2+DimDomain[2]**2)/(np.mean([SolidsSOS,WaterSOS])) #time to cross one corner to another
TemporalStep=5e-8 # if this step is too coarse a warning will be generated (but simulation will continue,) 

MaterialMap=np.zeros((N1,N2,N3),np.uint32) # note the 32 bit size
MaterialList=np.zeros((2,5)) # two materials in this example
MaterialList[0,0]=WaterDensity # water density
MaterialList[0,1]=WaterSOS # water SoS

MaterialList[1,0]=SolidDensity # plastic density
MaterialList[1,1]=SolidLSOS # plastic long. SoS
MaterialList[1,2]=SolidsSOS # plastic shear SoS
MaterialList[1,3]=SolidLAlpha # plastic long. attenuation
MaterialList[1,4]=SolidSAlpha # plastic shear attenuation

# We define three regions: the first one is water, the second one is skull, the third one is water
skull_thickness = int(0.002 / SpatialStep)  # 2mm skull sheet
middle_start = int(N3 / 2) - int(skull_thickness / 2)
middle_end = middle_start + skull_thickness

MaterialMap[:, :, middle_start:middle_end] = 1

plt.figure()
plt.imshow(MaterialMap[:,int(N2/2),:].T,cmap=plt.cm.gray)
plt.title('Material map')
plt.xlabel('x')
plt.ylabel('z')

#all other parameters are set to 0 

def MakeFocusingSource(N1,N2,N3,SpatialStep,TxRadius,TxDiameter,Angle):
    #simple defintion of a focusing source centred in the domain, 
    #please note this is not a bullet-proof solution as it may not work for all cases
    XDim=np.arange(N1)*SpatialStep
    YDim=np.arange(N2)*SpatialStep
    ZDim=np.arange(N3)*SpatialStep
    XDim-=XDim[int(N1/2)]
    YDim-=YDim[int(N2/2)]
    ZDim-=ZDim[int(N3/2)]
    XX,YY,ZZ=np.meshgrid(YDim,XDim,ZDim)#note we have to invert this because how meshgrid works
    Depth=np.sqrt(TxRadius**2-(TxDiameter/2.0)**2)
    cX=int(N1/2)
    cZ=int(N3/2)
    
    MaskSource=np.zeros((N1,N2,N3),np.bool_)
    FillMask=np.zeros((N1,N2,N3))
    
    for n,y in enumerate(YDim):
        if np.abs(y)<TxRadius:
            cDiam=int(np.ceil(TxRadius*np.sin(np.arccos(y/TxRadius))/SpatialStep))
            rr, cc = circle_perimeter(cX,cZ,cDiam,shape=(N1,N3))
            MaskSource[rr,n,cc]=np.True_
            rr,cc=disk((cX,cZ),cDiam+1,shape=(N1,N3))
            FillMask[rr,n,cc]=1
            
    
    FillMask[ZZ<=-Depth]=0.
    #instead of rotating the arc, we rotate the mask that will negate the perimeter to be turned off
    if Angle!=0.:
        for n in range(N2):
            FillMask[:,n,:]=rotate(FillMask[:,n,:],Angle,preserve_range=True)
        
    MaskSource[FillMask!=0]=False
        
    #since the sphere mask is 0-centred, the orientation vectors in each point is straighforward
    OxOyOz=np.vstack((-XX.flatten(),-YY.flatten(),-ZZ.flatten())).T
    #and we just normalize
    OxOyOz/=np.tile( np.linalg.norm(OxOyOz,axis=1).reshape(OxOyOz.shape[0],1),[1,3])
    Ox=OxOyOz[:,1].reshape(XX.shape) 
    Oy=OxOyOz[:,0].reshape(XX.shape)
    Oz=OxOyOz[:,2].reshape(XX.shape)
    Ox[MaskSource==False]=0
    Oy[MaskSource==False]=0
    Oz[MaskSource==False]=0
    return MaskSource.astype(np.uint32),Ox,Oy,Oz


SourceMap,Ox,Oy,Oz=MakeFocusingSource(N1,N2,N3,SpatialStep,TxRadius,TxDiameter,-20)
XDim=(np.arange(N1)*SpatialStep-(PMLThickness+1)*SpatialStep)*100 #cm
XDim-=XDim.mean()
ZDim=(np.arange(N3)*SpatialStep-(PMLThickness+1)*SpatialStep)*100 #cm
plt.figure(figsize=(6,5))
plt.imshow(SourceMap[:,int(N2/2),:].T,cmap=plt.cm.gray,extent=[XDim.min(),XDim.max(),ZDim.max(),ZDim.min()]);

LengthSource=4.0/Frequency #we will use 4 pulses
TimeVectorSource=np.arange(0,LengthSource+TemporalStep,TemporalStep)

PulseSource = np.sin(2*np.pi*Frequency*TimeVectorSource)
plt.figure()
plt.plot(TimeVectorSource*1e6,PulseSource)
plt.title('4-pulse signal')

#note we need expressively to arrange the data in a 2D array
PulseSource=np.reshape(PulseSource,(1,len(TimeVectorSource))) 

SensorMap=np.zeros((N1,N2,N3),np.uint32)

SensorMap[PMLThickness:-PMLThickness,int(N2/2),PMLThickness:-PMLThickness]=1

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
                                                         DT=TemporalStep,
                                                         ReflectionLimit=ReflectionLimit,
                                                         COMPUTING_BACKEND=COMPUTING_BACKEND,
                                                         USE_SINGLE=True,
                                                         SelMapsRMSPeakList=['Sigmaxx',
                                                                            'Sigmayy','Sigmazz',
                                                                            'Sigmaxy','Sigmaxz',
                                                                            'Sigmayz','Pressure'],
                                                         DefaultGPUDeviceName=DefaultGPUDeviceName,
                                                         SelMapsSensorsList=['Vx','Vy','Vz','Sigmaxx',
                                                                            'Sigmayy','Sigmazz',
                                                                            'Sigmaxy','Sigmaxz',
                                                                            'Sigmayz','Pressure'],
                                                         SensorSubSampling=4,
                                                         TypeSource=0)



plt.figure(figsize=(18,8))
n=1
for k,RMSValue in DictRMSValue.items():
    for a in range(3):
        for m in range(-2,3):
            RMSValue[np.roll(SourceMap>0,m,axis=a)]=0. #we turn off the values close the source 
    plt.subplot(2,4,n)
    selP=RMSValue[:,int(N2/2),:].copy()
    if 'Sigma' in k:
        selP=selP/1e5 
    selP[selP==0]=1e-6;
    selP=20*np.log10(selP/np.max(selP))
    plt.imshow(selP.T,cmap=plt.cm.jet,vmin=-40,vmax=0)
    plt.colorbar()
    if 'Sigma' in k:
        plt.title('$\sigma_{'+k.split('Sigma')[1]+'}$')
    else:
        plt.title(k)
    n+=1


MaxSensorPlane=np.zeros((N1,N3))
RMSSensorPlane=np.zeros((N1,N3))

for s in ['Vx','Vy','Vz','Pressure']:
    #We use the IndexSensorMap array that was used in the low level function to 
    for n, index in enumerate( InputParam['IndexSensorMap']): 
        k=int(index/(N1*N2))
        j=int(index%(N1*N2))
        i=int(j%N1)
        j=int(j/N1)
        assert(j==int(N2/2)) #all way up we specified the XZ plane at N2/2, this assert should pass
        if i==int(N1/2) and k==int(N3/2):
            CentralPoint=n #we save this to later plot the time signal at the center
        MaxSensorPlane[i,k]=np.max(Sensor[s][n,:])
        RMSSensorPlane[i,k]=np.sqrt(1./len(Sensor[s][n,:])*np.sum(Sensor[s][n,:]**2))
        
    for a in range(2):
        for m in range(-2,3):
            RMSSensorPlane[np.roll(SourceMap[:,j,:]>0,m,axis=a)]=0. #we turn off the values close the source
            MaxSensorPlane[np.roll(SourceMap[:,j,:]>0,m,axis=a)]=0. #we turn off the values close the source

    plt.figure(figsize=(14,8))
    plt.subplot(1,3,1)
    plt.imshow(MaxSensorPlane.T)
    plt.title('Peak value')
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.imshow(RMSSensorPlane.T)
    plt.title('RMS value')
    plt.colorbar()
    plt.subplot(1,3,3)
    plt.plot(Sensor['time']*1e6,Sensor[s][CentralPoint])
    plt.xlabel('time ($\mu$s)')
    plt.ylabel('$'+s[0]+'_'+s[1]+'$')
    plt.title('Time signal at central point')
    plt.suptitle('Plots for $'+s[0]+'_'+s[1]+'$')
    plt.tight_layout()


class ShowTimeData(object):
    def __init__(self,N1,N2,N3,InputParam,SourceMap,Sensor,Skip=1):
        nL=Sensor['Vx'].shape[1]
        IndTime=np.arange(0,nL,Skip)
        SensorPlane={}
        for k in Sensor:
            if k=='time':
                continue
            SensorPlane[k]=np.zeros((N1,N3,len(IndTime)))
        BInd=[]
        for n, index in enumerate( InputParam['IndexSensorMap']): 
            k=int(index/(N1*N2))
            j=int(index%(N1*N2))
            i=int(j%N1)
            j=int(j/N1)
            BInd.append([i,j,k])
        BInd=np.array(BInd)
        for k in SensorPlane:
            for nt,et in enumerate(IndTime):
                SensorPlane[k][BInd[:,0],BInd[:,2],nt]=Sensor[k][:,et]
            if 'Sigma' in k:
                SensorPlane[k]/=1e5
        self.SensorPlane=SensorPlane
        self.fig=None
        
    def PlotTimePoint(self,timep=0):
        if self.fig:
            plt.close(self.fig)
        
        self.fig=plt.figure(figsize=(16,16))
        n=1
        for k in  self.SensorPlane:
            plt.subplot(4,3,n)
            plt.imshow(self.SensorPlane[k][:,:,timep].T,cmap=plt.cm.jet)
            plt.colorbar()
            n+=1
            if 'V' in k:
                plt.title('$V_'+k.split('V')[1]+'$')
            elif 'Sigma' in k:
                plt.title('$\sigma_{'+k.split('Sigma')[1]+'}$')
            else:
                plt.title(k)
        plt.show()
        
    def SelectTimePoint(self):
        Sel=IntSlider(description='Time point',value=0,
                        min=0,max=self.SensorPlane[list(self.SensorPlane.keys())[0]].shape[2]-1,step=1,continuous_update=False,
                        layout=Layout(width='75%'),style={'description_width': 'initial'})
        interact(self.PlotTimePoint,timep=Sel)
       
        

ShowTimeData(N1,N2,N3,InputParam,SourceMap,Sensor).SelectTimePoint()


# In[ ]:

MaterialMap=np.zeros((N1,N2,N3),np.uint32) # note the 32 bit size
MaterialList=np.zeros((2,5)) # two materials in this example
MaterialList[0,0]=WaterDensity # water density
MaterialList[0,1]=WaterSOS # water SoS

MaterialList[1,0]=SolidDensity # plastic density
MaterialList[1,1]=SolidLSOS # plastic long. SoS
MaterialList[1,2]=SolidsSOS # plastic shear SoS
MaterialList[1,3]=SolidLAlpha # plastic long. attenuation
MaterialList[1,4]=SolidSAlpha # plastic shear attenuation
#we define two - half spaces

MaterialMap[:,:,int(N3/2):int(N3/2)+40]=1

plt.figure()
plt.imshow(MaterialMap[:,int(N2/2),:].T,cmap=plt.cm.gray);

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
                                                         DT=TemporalStep,
                                                         ReflectionLimit=ReflectionLimit,
                                                         COMPUTING_BACKEND=COMPUTING_BACKEND,
                                                         USE_SINGLE=True,
                                                         SelMapsRMSPeakList=['Sigmaxx',
                                                                            'Sigmayy','Sigmazz',
                                                                            'Sigmaxy','Sigmaxz',
                                                                            'Sigmayz','Pressure'],
                                                         DefaultGPUDeviceName=DefaultGPUDeviceName,
                                                         SelMapsSensorsList=['Vx','Vy','Vz','Sigmaxx',
                                                                            'Sigmayy','Sigmazz',
                                                                            'Sigmaxy','Sigmaxz',
                                                                            'Sigmayz','Pressure'],
                                                         SensorSubSampling=4,
                                                         TypeSource=0)

# In[]:


plt.figure(figsize=(18,8))
n=1
for k,RMSValue in DictRMSValue.items():
    for a in range(3):
        for m in range(-2,3):
            RMSValue[np.roll(SourceMap>0,m,axis=a)]=0. #we turn off the values close the source 
    plt.subplot(2,4,n)
    selP=RMSValue[:,int(N2/2),:].copy()
    if 'Sigma' in k:
        selP=selP/1e5 
    selP[selP==0]=1e-6;
    selP=20*np.log10(selP/np.max(selP))
    plt.imshow(selP.T,cmap=plt.cm.jet,vmin=-40,vmax=0)
    plt.colorbar()
    if 'Sigma' in k:
        plt.title('$\sigma_{'+k.split('Sigma')[1]+'}$')
    else:
        plt.title(k)
    n+=1


# In[]:


ShowTimeData(N1,N2,N3,InputParam,SourceMap,Sensor).SelectTimePoint()


# Now the the attenuation is being considered, we can appreciate that in the "All_V" map that one single dominant  patterns appear in the far field water domain, indicating the longitudinal type conversion is highly dominating the transmission.  Also, in the temporal map plots we can see the amplitude of the transverse component being much more attenuated.
# 
# # Above critical angle
# The previous example used an acoustic beam hitting the block material with an inciden angle of 20 degrees. The critical angle for this material is around 38 degrees. To show only shear-type transmission through the solid, this time we run the simulation with an incident angle of 45 degrees.

# In[16]:


SourceMap,Ox,Oy,Oz=MakeFocusingSource(N1,N2,N3,SpatialStep,TxRadius,TxDiameter,-45)
XDim=(np.arange(N1)*SpatialStep-(PMLThickness+1)*SpatialStep)*100 #cm
XDim-=XDim.mean()
ZDim=(np.arange(N3)*SpatialStep-(PMLThickness+1)*SpatialStep)*100 #cm
plt.figure(figsize=(6,5))
plt.imshow(SourceMap[:,int(N2/2),:].T,cmap=plt.cm.gray);


# In[17]:


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
                                                         DT=TemporalStep,
                                                         ReflectionLimit=ReflectionLimit,
                                                         COMPUTING_BACKEND=COMPUTING_BACKEND,
                                                         USE_SINGLE=True,
                                                         SelMapsRMSPeakList=['Sigmaxx',
                                                                            'Sigmayy','Sigmazz',
                                                                            'Sigmaxy','Sigmaxz',
                                                                            'Sigmayz','Pressure'],
                                                         DefaultGPUDeviceName=DefaultGPUDeviceName,
                                                         SelMapsSensorsList=['Vx','Vy','Vz','Sigmaxx',
                                                                            'Sigmayy','Sigmazz',
                                                                            'Sigmaxy','Sigmaxz',
                                                                            'Sigmayz','Pressure'],
                                                         SensorSubSampling=4,
                                                         TypeSource=0)


# ### Ploting data

# In[18]:


plt.figure(figsize=(18,8))
n=1
for k,RMSValue in DictRMSValue.items():
    for a in range(3):
        for m in range(-2,3):
            RMSValue[np.roll(SourceMap>0,m,axis=a)]=0. #we turn off the values close the source 
    ax=plt.subplot(2,4,n)
    selP=RMSValue[:,int(N2/2),:].copy()
    if 'Sigma' in k:
        selP=selP/1e5 
    selP[selP==0]=1e-6;
    selP=20*np.log10(selP/np.max(selP))
    plt.imshow(selP.T,cmap=plt.cm.jet,vmin=-40,vmax=0)
    plt.colorbar()
    ax.set_aspect(1)
    if 'Sigma' in k:
        plt.title('$\sigma_{'+k.split('Sigma')[1]+'}$')
    else:
        plt.title(k)
    n+=1


# In[19]:


ShowTimeData(N1,N2,N3,InputParam,SourceMap,Sensor).SelectTimePoint()


# As in the previous notebook, the transmitted field in the far field is refracted mostly by the shear-mode conversion. However, because the attenuation is much higher, the coherence is less pronounced once the wave crosses to the far field region.

# In[ ]:




