#pip install pylibjpeg pylibjpeg-libjpeg (s.pixel_array error)

import sys
from PyQt5 import QtGui
from lungmask.utils import preprocess
import numpy as np
from numpy.core.fromnumeric import mean
from numpy.lib.function_base import average, median
from numpy.ma.core import exp
import pydicom
import os
import os.path
import matplotlib.pyplot as plt
from lungmask import mask
import SimpleITK as sitk
from pydicom import dcmread



from PyQt5 import QtCore
from PyQt5 import *;
from PyQt5.QtWidgets import QFileDialog, QLabel, QApplication,QSlider,QGridLayout,QDialog,QPushButton
from PyQt5.QtCore import *
from PyQt5.QtGui import *

PIC_HEIGHT=200
PIC_WIDTH=200
FIXED_INHALE=-950
FIXED_EXHALE=-856
INSPIRATION_AND_EXPIRATION_SLICES_MAX_DIFF_COUNT=80
CSV_FILE_PATH='data/data.csv'

class MainWindow(QDialog):
    def initVars(self):
        self.path = ''#'DICOM/GOLABADI.HASAN____36YM_95.4.31-R2391003_2016-07-21'
        self.patientFolderName=''
        self.voxelSize=.0
        self.inspirationPath=''
        self.expirationPath=''
        self.inspiration=[]
        self.expiration=[]
        self.segmented_in_lung_mask=[]
        self.segmented_ex_lung_mask=[]
        self.binary_in_filter=np.array([])
        self.binary_ex_filter=np.array([])
        self.delta=()
        self.voxels_inhale_HUs=np.array([])
        self.voxels_ex_HUs=np.array([])
        self.only_lungs_in=np.array([])
        self.only_lungs_ex=np.array([])
        self.inspirationVolume=.0
        self.expirationVolume=.0
        self.MLD_ins=0
        self.MLD_exp=0
        self.MLDR=0
        self.NDI=0
        self.NDE=0
        self.NDEI=0
        self.VDR=0
        self.T=0
        self.AT=0
        self.normal_vx_cnt=0
        self.emphysema_vx_cnt=0
        self.airTrapping_vx_cnt=0
        self.all_vx_cnt=0
        self.AVI=0
        self.inspirationImage=0
        self.expirationImage=0
        self.voxels_registered_inhale_HUs=0
        self.rvc=0
        self.z_in=0
        self.x_in=0
        self.y_in=0
        self.z_ex=0
        self.x_ex=0
        self.y_ex=0
        self.inrange_ex_voxels_p=0
        self.inrange_in_voxels_p=0
        self.abnormal_in_voxels_p=0
        self.abnormal_ex_voxels_p=0

    def redraw(self,idx):
        pixMap1=QtGui.QPixmap(self.save_image_in_actual_size(None,self.voxels_inhale_HUs[idx],"raw_in_image"+str(idx),True))
        self.pic11.setPixmap(pixMap1.scaledToHeight(PIC_HEIGHT))
        pixMap2=QtGui.QPixmap(self.save_image_in_actual_size(self.binary_in_filter[idx],self.voxels_inhale_HUs[idx],"masked_in_image"+str(idx),True))
        self.pic12.setPixmap(pixMap2.scaledToHeight(PIC_HEIGHT))
        pixMap3=QtGui.QPixmap(self.save_image_in_actual_size(None,self.voxels_ex_HUs[idx],"raw_ex_image"+str(idx),False))
        self.pic21.setPixmap(pixMap3.scaledToHeight(PIC_HEIGHT))
        pixMap4=QtGui.QPixmap(self.save_image_in_actual_size(self.binary_ex_filter[idx],self.voxels_ex_HUs[idx],"masked_ex_image"+str(idx),False))
        self.pic22.setPixmap(pixMap4.scaledToHeight(PIC_HEIGHT)) 
        pixMap5=QtGui.QPixmap(self.save_histogram(self.voxels_inhale_HUs[idx],"histogram_in"+str(idx)))
        self.pic13.setPixmap(pixMap5.scaledToHeight(PIC_HEIGHT)) 
        pixMap6=QtGui.QPixmap(self.save_histogram(self.voxels_ex_HUs[idx],"histogram_ex_"+str(idx)))
        self.pic23.setPixmap(pixMap6.scaledToHeight(PIC_HEIGHT)) 
        pixMap7=QtGui.QPixmap(self.save_prm_image_in_actual_size(self.binary_ex_filter[idx],self.voxels_ex_HUs[idx],self.voxels_registered_inhale_HUs[idx],"prm_image"+str(idx)))
        self.pic44.setPixmap(pixMap7.scaledToHeight(PIC_HEIGHT)) 


    def changeSlice(self):
        self.redraw(self.slider00.value())

    def generateHist(self):
        self.pic13.setPixmap(QtGui.QPixmap(self.save_histogram(self.voxels_inhale_HUs,"Inspiration Histogram")).scaledToHeight(PIC_HEIGHT))
        self.pic23.setPixmap(QtGui.QPixmap(self.save_histogram(self.voxels_ex_HUs,"Expiration Histogram")).scaledToHeight(PIC_HEIGHT))
    
    def draw3DSlow(self):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import numpy as np
        from skimage import measure

        p = self.z_in.transpose(0,2,1)
        verts, faces, normals, values = measure.marching_cubes_lewiner(p, 0.5)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        mesh = Poly3DCollection(verts[faces], alpha=0.1)
        face_color = [0.5, 0.5, 1]
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
        ax.set_xlim(0, p.shape[0])
        ax.set_ylim(0, p.shape[1])
        ax.set_zlim(0, p.shape[2])

        plt.show()

    def draw3D(self):
        from matplotlib import pyplot
        from mpl_toolkits.mplot3d import Axes3D

        fig=pyplot.figure()
        ax=Axes3D(fig)

        z1=self.z_in[0:-1:200]
        y1=self.y_in[0:-1:200]
        x1=self.x_in[0:-1:200]

        z2=self.z_ex[0:-1:200]
        y2=self.y_ex[0:-1:200]
        x2=self.x_ex[0:-1:200]


        ax.scatter(z1,y1,x1,cmap = 'plasma',s=1, alpha = 1, depthshade = False)
        ax.scatter(z2,y2,x2,cmap = 'plasma',s=1, alpha = 1, depthshade = False)
        pyplot.show()


    def get_voxels_hu(self,path):
        print("Reading Dicom directory:", path)
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(path)
        ds = dcmread(dicom_names[0])
        self.patientFolderName=str(ds.PatientName).replace('/','-').replace('.','-').replace('  ',' ')+"_"+str(ds.AcquisitionDate)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
        return image,sitk.GetArrayFromImage(image)


    def save_image(self,fig,name):

        plt.axis('off')
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                    hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(name,bbox_inches='tight',pad_inches = 0)
        plt.close(fig)


    def save_prm_image_in_actual_size(self,mask,ex,reg_in,name):
            if not os.path.exists('data'):
                os.mkdir('data')
            if not os.path.exists('data/'+self.patientFolderName):
                os.mkdir('data/'+self.patientFolderName)
            name='data/'+self.patientFolderName+'/'+name
            if os.path.exists(name+'.png'):
                return name
            plt.ioff()
            fig=plt.figure()
            plt.imshow(ex, cmap='gray')
            if mask is not None and ex is not None:
                x=[]
                y=[]
                c_x=[]
                c_y=[]
                colors=[]

                for i in range(mask.shape[0]):
                        for j in range(mask.shape[1]):
                            if mask[i,j]==True:
                                y.append(i)
                                x.append(j)
                                if reg_in[i,j]<FIXED_INHALE and ex[i,j]<FIXED_EXHALE:
                                    c_y.append(i)
                                    c_x.append(j)  
                                    colors.append('red')
                                    continue
                                elif reg_in[i,j]>FIXED_INHALE and ex[i,j]<FIXED_EXHALE:
                                    c_y.append(i)
                                    c_x.append(j)   
                                    colors.append('blue')
                                else:
                                    c_y.append(i)
                                    c_x.append(j)   
                                    colors.append('green')     
                plt.scatter(c_x, c_y, c=colors, alpha=0.25,s=1)
            self.save_image(fig,name)
            return name

    def save_image_in_actual_size(self,mask,hu,name,isInspiration):
        if not os.path.exists('data'):
            os.mkdir('data')
        if not os.path.exists('data/'+self.patientFolderName):
            os.mkdir('data/'+self.patientFolderName)
        name='data/'+self.patientFolderName+'/'+name
        if os.path.exists(name+'.png'):
            return name
        plt.ioff()
        fig=plt.figure()
        plt.imshow(hu, cmap='gray')

        if mask is not None and hu is not None:
            x=[]
            y=[]
            c_x=[]
            c_y=[]
            colors=[]

            for i in range(mask.shape[0]):
                    for j in range(mask.shape[1]):
                        if mask[i,j]==True:
                            y.append(i)
                            x.append(j)
                            if isInspiration:
                                if hu[i,j]>FIXED_INHALE:
                                    c_y.append(i)
                                    c_x.append(j)  
                                    colors.append('green')
                                    continue
                                else:
                                    c_y.append(i)
                                    c_x.append(j)   
                                    colors.append('red')      
                            else: #isExpiration
                                if hu[i,j]>FIXED_EXHALE:
                                    c_y.append(i)
                                    c_x.append(j)  
                                    colors.append('green')
                                    continue
                                else:
                                    c_y.append(i)
                                    c_x.append(j)   
                                    colors.append('red')      
            plt.scatter(c_x, c_y, c=colors, alpha=0.25,s=1)
        self.save_image(fig,name)
        return name

    def save_histogram(self,data,name):
        name='data/'+self.patientFolderName+'/'+name
        if os.path.exists(name+'.png'):
            return name
        plt.ioff()
        fig=plt.figure()
        figure = plt.gcf()
        figure.set_size_inches(200/100, 200/100)
        plt.xlim(xmin=-1000, xmax = 0)
        plt.gca().set_position([0, 0, 1, 1]) 
        f_data=data.flatten()

        threshold=FIXED_INHALE
        if 'ex' in name:
            threshold=FIXED_EXHALE

        plt.hist(f_data[f_data>round(threshold/10)*10], bins = 500,range=[-1000, 0], color = "green", lw=0) 
        plt.hist(f_data[f_data<round(threshold/10)*10], bins = 500,range=[-1000, 0], color = "red", lw=0) 
    
        plt.title(name) 
        plt.savefig(name,dpi=100)
        plt.close(fig)
        return name



    def detectInAndExFolders(self,path):
        for d in os.listdir(path):
            currentPath=path+'/'+d;
            if os.path.isfile(currentPath):
                continue
            slices = [pydicom.dcmread(currentPath+'/'+ s) for s in               
                    os.listdir(currentPath) if s.endswith('dcm')]
            if len(slices)<2:
                continue
            slices = [s for s in slices if 'SliceLocation' in s]
            slices.sort(key = lambda x: int(x.InstanceNumber))
            if 'INSP' in slices[0].ImageComments.upper() or 'INSP' in slices[0].SeriesDescription.upper():
                self.inspiration.append({'path':currentPath,'filesCount':len(slices)})
                self.inspiration.sort(key = lambda x: int(x['filesCount']),reverse = True )
                RefDs = slices[0]
                ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(slices))
                self.voxelSize = float(RefDs.PixelSpacing[0])* float(RefDs.PixelSpacing[1])* float(RefDs.SliceThickness)
                print(self.voxelSize)

            elif 'EXP' in slices[0].ImageComments.upper() or 'EXP' in slices[0].SeriesDescription.upper():
                self.expiration.append({'path':currentPath,'filesCount':len(slices)})
                self.expiration.sort(key = lambda x: int(x['filesCount']),reverse = True )
                ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(slices))
                self.voxelSize = float(RefDs.PixelSpacing[0])* float(RefDs.PixelSpacing[1])* float(RefDs.SliceThickness)
                print(self.voxelSize)

        for i, j in np.ndindex(len(self.inspiration), len(self.expiration)):
            if self.inspiration[i]['filesCount']>=self.expiration[j]['filesCount']-INSPIRATION_AND_EXPIRATION_SLICES_MAX_DIFF_COUNT and self.inspiration[i]['filesCount']<=self.expiration[j]['filesCount']+INSPIRATION_AND_EXPIRATION_SLICES_MAX_DIFF_COUNT:
                self.inspirationPath=self.inspiration[i]['path']
                self.expirationPath=self.expiration[j]['path']
                break

    
    def register(self,inspirationPath,expirationPath):
        in_segmentation=(self.binary_in_filter).astype('uint8')
        in_maskImage=sitk.GetImageFromArray(in_segmentation)
        in_maskImage.CopyInformation(self.inspirationImage)

        ex_segmentation=(self.binary_ex_filter).astype('uint8')
        ex_maskImage=sitk.GetImageFromArray(ex_segmentation)
        ex_maskImage.CopyInformation(self.expirationImage)        

        # Instantiate SimpleElastix
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetFixedImage(self.expirationImage)
        elastixImageFilter.SetFixedMask(ex_maskImage)
        elastixImageFilter.SetMovingImage(self.inspirationImage)
        elastixImageFilter.SetMovingMask(in_maskImage)
        elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("affine"))

        # Perform registration
        #elastixImageFilter.LogToConsoleOn()
        elastixImageFilter.Execute()
        insp_registered=elastixImageFilter.GetResultImage()

        #sitk.PrintParameterMap(elastixImageFilter.GetTransformParameterMap()[0])
        #image3=sitk.Transformix(image,elastixImageFilter.GetTransformParameterMap(),True)

        # Get the current working directory:
        # CurrentWorkingDir = os.getcwd()
        # TransformParameterMap = elastixImageFilter.GetParameterMap()
        # TransformixImFilt = sitk.TransformixImageFilter()
        # TransformixImFilt.LogToConsoleOn()
        # TransformixImFilt.LogToFileOn()
        # TransformixImFilt.SetOutputDirectory(CurrentWorkingDir)
        # TransformixImFilt.ComputeDeformationFieldOn() 
        # TransformixImFilt.ComputeSpatialJacobianOn()
        # TransformixImFilt.ComputeDeterminantOfSpatialJacobianOn()
        # TransformixImFilt.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
        # TransformixImFilt.SetMovingImage(image)
        # TransformixImFilt.Execute()
        # image3=TransformixImFilt.GetResultImage()


        # image1A=sitk.GetArrayFromImage(image).astype('int16')
        # image2A=sitk.GetArrayFromImage(image2).astype('int16')
        self.voxels_registered_inhale_HUs=sitk.GetArrayFromImage(insp_registered).astype('int16')
        return


    
    def preprocessDICOMImages(self,inspirationPath,expirationPath):
        print('loading inspiration images')
        self.inspirationImage,self.voxels_inhale_HUs = self.get_voxels_hu(inspirationPath) 
        self.segmented_in_lung_mask=mask.apply(self.voxels_inhale_HUs,model=mask.get_model('unet', 'LTRCLobes')).copy()
        in_mask=(self.segmented_in_lung_mask!=0)
        self.binary_in_filter= (in_mask==1)
        segmented_in_lungs=self.binary_in_filter*self.voxels_inhale_HUs
        self.only_lungs_in=segmented_in_lungs[segmented_in_lungs!=0]
        

        in_points = np.argwhere(self.binary_in_filter==1)
        self.x_in = [p[0] for p in in_points]
        self.y_in = [p[1] for p in in_points]
        self.z_in = [p[2] for p in in_points]
        centroid_inhale = [sum(self.x_in) / len(in_points), sum(self.y_in) / len(in_points),sum(self.z_in) / len(in_points)]

        print('loading expiration images')
        self.expirationImage,self.voxels_ex_HUs = self.get_voxels_hu(expirationPath) 
        self.segmented_ex_lung_mask=mask.apply(self.voxels_ex_HUs,model=mask.get_model('unet', 'LTRCLobes')).copy()
        ex_mask=(self.segmented_ex_lung_mask!=0)
        self.binary_ex_filter= (ex_mask==1)
        segmented_ex_lungs=self.binary_ex_filter*self.voxels_ex_HUs
        self.only_lungs_ex=segmented_ex_lungs[segmented_ex_lungs!=0]

        ex_points = np.argwhere(self.binary_ex_filter==1)
        self.x_ex = [p[0] for p in ex_points]
        self.y_ex = [p[1] for p in ex_points]
        self.z_ex = [p[2] for p in ex_points]
        centroid_exhale = [sum(self.x_ex) / len(ex_points), sum(self.y_ex) / len(ex_points),sum(self.z_ex) / len(ex_points)]

        self.register(inspirationPath,expirationPath)

        self.delta=tuple(np.subtract(centroid_exhale, centroid_inhale))
        print(self.delta)

    def processImages(self):
        f_only_lungs_in=np.array(self.only_lungs_in).flatten()
        self.inspirationVolume=len(f_only_lungs_in)*self.voxelSize
        f_only_lungs_ex=np.array(self.only_lungs_ex).flatten()
        self.expirationVolume=len(f_only_lungs_ex)*self.voxelSize
        
        self.abnormal_in_voxels_p=np.count_nonzero(f_only_lungs_in[f_only_lungs_in<FIXED_INHALE])/np.count_nonzero(f_only_lungs_in)
        self.abnormal_ex_voxels_p=np.count_nonzero(f_only_lungs_ex[f_only_lungs_ex<FIXED_EXHALE])/np.count_nonzero(f_only_lungs_ex)
        
        normal_in_voxels=f_only_lungs_in[np.bitwise_and(f_only_lungs_in<0,f_only_lungs_in>FIXED_INHALE)]
        normal_ex_voxels=f_only_lungs_ex[np.bitwise_and(f_only_lungs_ex<0,f_only_lungs_ex>FIXED_EXHALE)]
        self.MLD_ins=mean(self.only_lungs_in)
        print('MLD_ins='+str(self.MLD_ins))
        self.MLD_exp=mean(self.only_lungs_ex)
        print('MLD_exp='+str(self.MLD_exp))
        self.MLDR=self.MLD_exp/self.MLD_ins
        print('MLDR='+str(self.MLDR))
        self.NDI=normal_in_voxels.sum()/len(self.only_lungs_in)
        self.NDE=normal_ex_voxels.sum()/len(self.only_lungs_ex)
        self.NDEI=self.NDE/self.NDI
        print('NDE/I='+str(self.NDEI))
        X=np.percentile(f_only_lungs_in,90)
        Y=median(f_only_lungs_in)
        X_ex=np.percentile(f_only_lungs_ex,90)
        D=abs(X-X_ex)

        inrange_ex_voxels=f_only_lungs_ex[np.bitwise_and(f_only_lungs_ex<=FIXED_EXHALE,f_only_lungs_ex>=FIXED_INHALE)]
        inrange_in_voxels=f_only_lungs_in[np.bitwise_and(f_only_lungs_in<=FIXED_EXHALE,f_only_lungs_in>=FIXED_INHALE)]
        
        self.inrange_ex_voxels_p=np.count_nonzero(inrange_ex_voxels)/np.count_nonzero(f_only_lungs_ex)
        self.inrange_in_voxels_p=np.count_nonzero(inrange_in_voxels)/np.count_nonzero(f_only_lungs_in)


        greater_than_min_ex_voxels=f_only_lungs_ex[f_only_lungs_ex>=FIXED_INHALE]
        greater_than_min_in_voxels=f_only_lungs_in[f_only_lungs_in>=FIXED_INHALE]

        self.rvc=np.count_nonzero(inrange_ex_voxels)/np.count_nonzero(greater_than_min_ex_voxels)-np.count_nonzero(inrange_in_voxels)/np.count_nonzero(greater_than_min_in_voxels)




        self.T=np.zeros((3))
        less_than_T=np.zeros((3))
        self.AT=np.zeros((3))
        for i in range(3):
            self.T[i]=X-(i-1)*(X-Y)-(1-D/343)*(X-Y)/3
            less_than_T[i]=(np.count_nonzero(f_only_lungs_ex<self.T[i]))
            self.AT[i]=(less_than_T[i]/len(f_only_lungs_ex))
            print('AT {0} = {1}'.format((i+1),self.AT[i]))

        self.normal_vx_cnt=0
        self.emphysema_vx_cnt=0
        self.airTrapping_vx_cnt=0
        for p in np.argwhere(self.binary_ex_filter==1):
            x,y,z=p
            if self.voxels_registered_inhale_HUs[x,y,z]<FIXED_INHALE and self.voxels_ex_HUs[x,y,z]<FIXED_EXHALE:
                self.emphysema_vx_cnt=self.emphysema_vx_cnt+1
            elif self.voxels_registered_inhale_HUs[x,y,z]>FIXED_INHALE and self.voxels_ex_HUs[x,y,z]<FIXED_EXHALE:
                self.airTrapping_vx_cnt=self.airTrapping_vx_cnt+1
            else:
                self.normal_vx_cnt=self.normal_vx_cnt+1


        self.all_vx_cnt=self.normal_vx_cnt+self.emphysema_vx_cnt+self.airTrapping_vx_cnt
        
        print('NORMAL VX: {0}, EMPHYSEMA VX: {1}, AIR TRAPPING VX: {2}'.format(self.normal_vx_cnt/self.all_vx_cnt,self.emphysema_vx_cnt/self.all_vx_cnt,self.airTrapping_vx_cnt/self.all_vx_cnt))

        self.MVLA_insp=average(f_only_lungs_in)
        self.MVLA_exp=average(f_only_lungs_ex)
        self.VDR=100*(self.inspirationVolume-self.expirationVolume)/self.inspirationVolume
        self.AVI=(self.MVLA_exp-self.MVLA_insp)/self.VDR

        print('MVLA_insp: {0}, MVLA_ex: {1}, AVI: {2}'.format(self.MVLA_insp,self.MVLA_exp,self.AVI))
        newFile=False
        
        if not os.path.exists('data'):
            os.mkdir('data')

        if not os.path.isfile(CSV_FILE_PATH):
            newFile=True
            
        self.file = open(CSV_FILE_PATH, 'a')
        if newFile:
            self.file.write("Patient Name,P Inhale -950,P Exhale -856,MLD_ins,MLD_exp,E/I Ratio,NDE,NDI,NDEI,VDR,AVI,AT0,AT1,AT2,normal_vx_cnt,emphysema_vx_cnt,airTrapping_vx_cnt,RVC\n")
        self.file.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                self.patientFolderName,
                "{:.2f}".format(self.abnormal_in_voxels_p),
                "{:.2f}".format(self.abnormal_ex_voxels_p),
                "{:.2f}".format(self.MLD_ins),
                "{:.2f}".format(self.MLD_exp),
                "{:.2f}".format(self.MLDR),
                "{:.2f}".format(self.NDE),
                "{:.2f}".format(self.NDI),
                "{:.2f}".format(self.NDEI),
                "{:.2f}".format(self.VDR),
                "{:.2f}".format(self.AVI),
                "{:.2f}".format(self.AT[0]),
                "{:.2f}".format(self.AT[1]),
                "{:.2f}".format(self.AT[2]),
                "{:.2f}".format(self.normal_vx_cnt/self.all_vx_cnt),
                "{:.2f}".format(self.emphysema_vx_cnt/self.all_vx_cnt),
                "{:.2f}".format(self.airTrapping_vx_cnt/self.all_vx_cnt),
                "{:.2f}".format(self.rvc)
                ))
        self.file.flush()
        self.file.close()
    
    def process(self,patientDICOMfolderPath):
        self.initVars()
        self.path=patientDICOMfolderPath
        self.detectInAndExFolders(self.path)
        if(self.inspirationPath!='' and self.expirationPath!=''):
            self.preprocessDICOMImages(self.inspirationPath,self.expirationPath)
            self.processImages()
            if not self.isBatch:
                self.preloadImages()
                self.slider00.setMaximum(min(len(self.voxels_inhale_HUs),len(self.voxels_ex_HUs))-1)
                self.redraw(0)
            self.ATValLabel15.setText(
            ("AT[1]= {}\nAT[2]= {}\nAT[3]= {}\n"
            "NORMAL VX={}\nEMPHYSEMA VX= {}\nAIR TRAPPING VX= {}")\
            .format(
                "{:.2f}".format(self.AT[0]),
                "{:.2f}".format(self.AT[1]),
                "{:.2f}".format(self.AT[2]),
                "{:.2f}".format(self.normal_vx_cnt/self.all_vx_cnt),
                "{:.2f}".format(self.emphysema_vx_cnt/self.all_vx_cnt),
                "{:.2f}".format(self.airTrapping_vx_cnt/self.all_vx_cnt),
                ))
            self.NDEValLabel15.setText(
            ("NDE={}\nNDI={}\nNDE/I={}\nVDR={}\nAVI={}")\
            .format(
                "{:.2f}".format(self.NDE),
                "{:.2f}".format(self.NDI),
                "{:.2f}".format(self.NDEI),
                "{:.2f}".format(self.VDR),
                "{:.2f}".format(self.AVI),
                ))
            self.MLDValLabel14.setText(
            ("MLD-I={}\nMLD-E={}\nMLD E/I Ratio={}")\
            .format(
                "{:.2f}".format(self.MLD_ins),
                "{:.2f}".format(self.MLD_exp),
                "{:.2f}".format(self.MLDR),
                ))


    def batch(self):
        self.isBatch=True
        from pathlib import Path
        curdir = Path(str(QFileDialog.getExistingDirectory(self, "Select Directory")))
        directories_in_curdir = [f for f in curdir.iterdir() if f.is_dir()]
        isDicom=False
        for dir in directories_in_curdir:
            try:
                dirP = Path(dir)
                inner_dirs=[f for f in dirP.iterdir() if f.is_dir()]
                for i in inner_dirs:
                    if 'SR_1' in str(i.parts[-1]):
                        isDicom=True
                        break
                if isDicom:
                    self.process(str(dir))
            except:
                print("Skipping Bad Folder:"+dir)
                    
        

    def single(self):
        self.isBatch=False
        self.path = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.process(self.path)

            
            
    def preloadImages(self):
        for idx in range(min(len(self.voxels_inhale_HUs),len(self.voxels_ex_HUs))):
            self.save_image_in_actual_size(None,self.voxels_inhale_HUs[idx],"raw_in_image"+str(idx),True)
            self.save_image_in_actual_size(self.binary_in_filter[idx],self.voxels_inhale_HUs[idx],"masked_in_image"+str(idx),True)
            self.save_image_in_actual_size(None,self.voxels_ex_HUs[idx],"raw_ex_image"+str(idx),False)
            self.save_image_in_actual_size(self.binary_ex_filter[idx],self.voxels_ex_HUs[idx],"masked_ex_image"+str(idx),False)
            self.save_histogram(self.voxels_inhale_HUs[idx],"histogram_in"+str(idx))
            self.save_histogram(self.voxels_ex_HUs[idx],"histogram_ex_"+str(idx))
            self.save_prm_image_in_actual_size(self.binary_ex_filter[idx],self.voxels_ex_HUs[idx],self.voxels_registered_inhale_HUs[idx],"prm_image"+str(idx))
            print('preloading files',idx)    

      
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lung analyzer")
        parent_layout = QGridLayout()
        self.slider00 = QSlider(Qt.Horizontal)
        self.btnHist01= QPushButton(self)
        self.btn3D02= QPushButton(self)
        histLabel03=QLabel(self)
        self.btnBatch04= QPushButton(self)
        self.btnBatch04.setStyleSheet("background-color: blue")
        self.btnSingle14= QPushButton(self)
        self.btnSingle14.setStyleSheet("background-color: blue")
        MLDLabel04=QLabel(self)
        MLDLabel04.setStyleSheet("background-color: #caf1de")
        NDE05=QLabel(self)
        NDE05.setStyleSheet("background-color: #fef8dd")
        AT06=QLabel(self)
        AT06.setStyleSheet("background-color: #f7d8ba")
        inspirationLabel10 = QLabel(self)
        self.pic11 = QLabel(self)
        self.pic12 = QLabel(self)
        self.pic13 = QLabel(self)
        self.MLDValLabel14= QLabel(self)
        self.MLDValLabel14.setStyleSheet("background-color: #caf1de")
        self.NDEValLabel15= QLabel(self)
        self.NDEValLabel15.setStyleSheet("background-color: #fef8dd")
        self.ATValLabel15= QLabel(self)
        self.ATValLabel15.setStyleSheet("background-color: #f7d8ba")
        expirationLabel20 = QLabel(self)
        self.pic21 = QLabel(self)
        self.pic22 = QLabel(self)
        self.pic23 = QLabel(self)
        self.pic44 = QLabel(self)
        overallLabel30 = QLabel(self)
        parent_layout.addWidget(self.slider00, 0, 0)
        parent_layout.addWidget(self.btnHist01,0,1)
        parent_layout.addWidget(self.btn3D02,0,2)
        parent_layout.addWidget(histLabel03,0,3)
        parent_layout.addWidget(self.btnBatch04,3,4)
        parent_layout.addWidget(self.btnSingle14,2,4)
        parent_layout.addWidget(inspirationLabel10, 1, 0)
        parent_layout.addWidget(self.pic11, 1, 1)
        parent_layout.addWidget(self.pic12, 1, 2)
        parent_layout.addWidget(self.pic13, 1, 3)
        parent_layout.addWidget(expirationLabel20, 2, 0)
        parent_layout.addWidget(self.pic21, 2, 1)
        parent_layout.addWidget(self.pic22, 2, 2)
        parent_layout.addWidget(self.pic23, 2, 3)
        parent_layout.addWidget(MLDLabel04,3,1)
        parent_layout.addWidget(NDE05,3,2)
        parent_layout.addWidget(AT06,3,3)
        parent_layout.addWidget(overallLabel30, 4, 0)
        parent_layout.addWidget(self.MLDValLabel14,4,1)
        parent_layout.addWidget(self.NDEValLabel15,4,2)
        parent_layout.addWidget(self.ATValLabel15,4,3)    
        parent_layout.addWidget(self.pic44,4,4)     
        self.slider00.setFocusPolicy(Qt.StrongFocus)
        self.slider00.setTickPosition(QSlider.TicksBothSides)
        self.slider00.setMinimum(0)
        self.slider00.setTickInterval(1)
        self.slider00.setSingleStep(1)
        self.slider00.valueChanged.connect(self.changeSlice)
        self.btnHist01.clicked.connect(self.generateHist)
        self.btnHist01.setText("Generate Histogram")
        self.btn3D02.setText("3D")
        self.btn3D02.clicked.connect(self.draw3D)
        histLabel03.setText("Histograms")
        histLabel03.setFont(QFont('Comfortaa', 14))
        histLabel03.setAlignment(QtCore.Qt.AlignCenter)
        self.btnBatch04.setText("Batch Patients Processing")
        self.btnBatch04.clicked.connect(self.batch)
        self.btnSingle14.setText("Single Patient Processing")
        self.btnSingle14.clicked.connect(self.single)
        MLDLabel04.setText("Mean Lung Dose (MLD)")
        MLDLabel04.setFont(QFont('Comfortaa', 14))
        MLDLabel04.setAlignment(QtCore.Qt.AlignCenter)
        NDE05.setText("Normal Density")
        NDE05.setFont(QFont('Comfortaa', 14))
        NDE05.setAlignment(QtCore.Qt.AlignCenter)
        AT06.setText("Parametric Response Map")
        AT06.setFont(QFont('Comfortaa', 14))
        AT06.setAlignment(QtCore.Qt.AlignCenter)
        self.MLDValLabel14.setFont(QFont('Comfortaa', 12))
        self.MLDValLabel14.width=PIC_WIDTH
        inspirationLabel10.setText("Inspiration")
        inspirationLabel10.setFont(QFont('Comfortaa', 24))
        inspirationLabel10.setStyleSheet("background-color: #36B7F3")
        self.NDEValLabel15.setFont(QFont('Comfortaa', 12))
        self.NDEValLabel15.width=PIC_WIDTH
        self.ATValLabel15.setFont(QFont('Comfortaa', 12))
        self.ATValLabel15.width=PIC_WIDTH
        expirationLabel20.setText("Expiration")
        expirationLabel20.setFont(QFont('Comfortaa', 24))
        expirationLabel20.setStyleSheet("background-color: #E86570")
        overallLabel30.setText("Overall")
        overallLabel30.setFont(QFont('Comfortaa', 24))
        overallLabel30.setStyleSheet("background-color: #f7d8ba")
        self.setLayout(parent_layout)
        self.setStyleSheet("background-color:white")

if __name__ == '__main__':

    app = QApplication(sys.argv)
    dialog = MainWindow()
    dialog.setWindowState(dialog.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
    dialog.activateWindow()
    dialog.show()


    sys.exit(app.exec_())