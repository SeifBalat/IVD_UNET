import nrrd
import matplotlib.pyplot as plt
import matplotlib.image as img

#read the 3d image
image_nrrd = ('B:/Masterarbeit/Thesis/Dataset/Dataset-3D/D0030100301_3D.nrrd')
readdata, header = nrrd.read(image_nrrd)


print('*'*40)
print('Sliceing Sagittal')
image_data = readdata.shape[0]
for i in range(0,image_data):
  img.imsave('B:/Masterarbeit/Thesis/Dataset/Dataset-2D/Train/D0030100301_3D/Data/Sagittal/image'+str(i)+'.png',readdata[i,:,:])

print('*'*40)
print('Sliceing Coronal')
image_data = readdata.shape[1]
for i in range(0,image_data):
  img.imsave('B:/Masterarbeit/Thesis/Dataset/Dataset-2D/Train/D0030100301_3D/Data/Coronal/image'+str(i)+'.png',readdata[:,i,:])

print('*'*40)
print('Sliceing Axial')
image_data = readdata.shape[2]
for i in range(0,image_data):
  img.imsave('B:/Masterarbeit/Thesis/Dataset/Dataset-2D/Train/D0030100301_3D/Data/Axial/image'+str(i)+'.png',readdata[:,:,i])