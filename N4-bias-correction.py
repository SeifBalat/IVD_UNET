

import SimpleITK as sitk

def N4():
    print("N4 bias correction runs.")
    inputImage = sitk.ReadImage("B:/Masterarbeit/Thesis/Dataset/Last-Update/D0040100402_3D.nrrd")
    # maskImage = sitk.ReadImage("06-t1c_mask.nii.gz")
    maskImage = sitk.OtsuThreshold(inputImage,0,1,200)
    sitk.WriteImage(maskImage, "B:/Masterarbeit/Thesis/Dataset/Last-Update/D0040100402_3D.seg.nrrd")

    inputImage = sitk.Cast(inputImage,sitk.sitkFloat32)

    corrector = sitk.N4BiasFieldCorrectionImageFilter();

    output = corrector.Execute(inputImage,maskImage)
    sitk.WriteImage(output,"B:/Masterarbeit/Thesis/Dataset/Last-Update/01_correct.nrrd")
    print("Finished N4 Bias Field Correction.....")

if __name__=='__main__':
   N4()