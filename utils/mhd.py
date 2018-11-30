import SimpleITK as sitk
import numpy as np


def load_oct_image(filename, device_type):
    """
    loads an .mhd file containing an OCT image using simple_itk
    :param filename: name of the image to be loaded
    :param device_type: oct devise vendor name
    :return: int32 3D image with voxels range 0-255
    """

    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)
    ct_scan = ct_scan.astype(np.int32)

    if "Spectralis" in device_type:
        ct_scan = (ct_scan.astype(np.float32) / (2 ** 16) * 255.).astype(np.int32)

    assert np.max(ct_scan) < 257

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing


def load_oct_seg(filename):
    """
    loads an .mhd file containing an OCT mask using simple_itk
    :param filename: name of the oct mask
    :return:
    """
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)
    ct_scan = ct_scan.astype(np.int8)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing