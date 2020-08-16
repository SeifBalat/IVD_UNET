import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.image as img

#read the 3d image
img_nii = nib.load('B:/Masterarbeit/New life/Dataset/IVDM3SegChallenge2018_TrainingData_Labeled/Training/01/01_opp.nii', mmap=False)
image_data = img_nii.dataobj
#Axial view
img_iterator = image_data.shape[2]
for i in range(0,img_iterator):
  img.imsave('B:/Masterarbeit/New life/Dataset/Test/train/opp/image'+str(i)+'.png',image_data[:,:,i])
#Coronal view
img_iterator = image_data.shape[1]
for i in range(0,img_iterator):
  img.imsave('B:/Masterarbeit/New life/Dataset/Test/train/opp/image'+str(i)+'.png',image_data[:,i,:])
#Sagittal view
  img_iterator = image_data.shape[0]
for i in range(0,img_iterator):
  img.imsave('B:/Masterarbeit/New life/Dataset/Test/train/opp/image'+str(i)+'.png',image_data[i,:,:])