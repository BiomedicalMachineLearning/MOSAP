from skimage import measure
from scipy import ndimage as ndi
# import cv2
import SimpleITK as sitk
import matplotlib.pyplot as plt
from typing import Any, Union, Optional
from scipy.optimize import minimize
from pathlib import Path
from PIL import Image
import numpy as np

#run internally
def start_plot():
    """Setup data for plotting
    Invoked when StartEvent happens at the beginning of registration.
    """
    global metric_values, multires_iterations

    metric_values = []
    multires_iterations = []

def end_plot():
    """Cleanup the data and figures
    """
    global metric_values, multires_iterations

    del metric_values
    del multires_iterations
    # Close figure, we don't want to get a duplicate of the plot latter on.
    plt.close()


def update_plot(registration_method):
    """Plot metric value after each registration iteration
    Invoked when IterationEvent happens.
    """
    global metric_values, multires_iterations

    metric_values.append(registration_method.GetMetricValue())
    # Clear the output area (wait=True, to reduce flickering), and plot current data
    # clear_output(wait=True)
    # Plot the similarity metric values
    plt.plot(metric_values, 'r')
    plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
    plt.xlabel('Iteration Number', fontsize=12)
    plt.ylabel('Metric', fontsize=12)
    plt.show()


def update_multires_iterations():
    """Update the index in the metric values list that corresponds to a change in registration resolution
    Invoked when the sitkMultiResolutionIterationEvent happens.
    """
    global metric_values, multires_iterations
    multires_iterations.append(len(metric_values))

def plot_metric(title='Plot of registration metric vs iterations'):
    """Plots the mutual information over registration iterations
    Parameters
    ----------
    title : str
    Returns
    -------
    fig : matplotlib figure
    """
    global metric_values, multires_iterations

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('Iteration Number', fontsize=12)
    ax.set_ylabel('Mutual Information Cost', fontsize=12)
    ax.plot(metric_values, 'r')
    ax.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*',
            label='change in resolution')
    ax.legend()
    return fig

def affine_registration_slides(fixed_ref_img, moving_img, plot_registration_progress=True):
    initial_transform = sitk.CenteredTransformInitializer(fixed_ref_img, moving_img, sitk.Euler2DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    affine_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    affine_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=75)
    affine_method.SetMetricSamplingStrategy(affine_method.RANDOM)
    affine_method.SetMetricSamplingPercentage(0.15)

    affine_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    affine_method.SetOptimizerAsGradientDescent(learningRate=1, numberOfIterations=300, convergenceMinimumValue=1e-6,
                                                convergenceWindowSize=20)
    affine_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    affine_method.SetShrinkFactorsPerLevel(shrinkFactors=[16, 8, 4, 3, 2, 1])
    affine_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[10, 4, 3, 2, 1, 0])
    affine_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    affine_method.SetInitialTransform(initial_transform, inPlace=False)

    # Connect all of the observers so that we can perform plotting during registration.
    if plot_registration_progress:
        affine_method.AddCommand(sitk.sitkStartEvent, start_plot)
        affine_method.AddCommand(sitk.sitkEndEvent, end_plot)
    affine_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations)
    affine_method.AddCommand(sitk.sitkIterationEvent, lambda: update_plot(affine_method))

    affine_transform = affine_method.Execute(sitk.Cast(fixed_ref_img, sitk.sitkFloat32),
                                             sitk.Cast(moving_img, sitk.sitkFloat32))
    return affine_transform


def bspline_registration_slides(fixed_ref_img, moving_img, plot_registration_progress=True):
    bspline_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    bspline_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    bspline_method.SetMetricSamplingStrategy(bspline_method.RANDOM)
    bspline_method.SetMetricSamplingPercentage(0.15)

    bspline_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    bspline_method.SetOptimizerAsGradientDescent(learningRate=1, numberOfIterations=75, convergenceMinimumValue=1e-6,
                                                 convergenceWindowSize=10)
    bspline_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    bspline_method.SetShrinkFactorsPerLevel(shrinkFactors=[2, 1])
    bspline_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[1, 0])
    bspline_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    transformDomainMeshSize = [8] * moving_img.GetDimension()
    initial_transform = sitk.BSplineTransformInitializer(fixed_ref_img, transformDomainMeshSize)
    bspline_method.SetInitialTransform(initial_transform, inPlace=False)

    # Connect all of the observers so that we can perform plotting during registration.
    if plot_registration_progress==True:
        bspline_method.AddCommand(sitk.sitkStartEvent, start_plot)
        bspline_method.AddCommand(sitk.sitkEndEvent, end_plot)
    bspline_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations)
    bspline_method.AddCommand(sitk.sitkIterationEvent, lambda: update_plot(bspline_method))

    bspline_transform = bspline_method.Execute(sitk.Cast(fixed_ref_img, sitk.sitkFloat32),
                                               sitk.Cast(moving_img, sitk.sitkFloat32))
    return bspline_transform

def inverse_transform_point(xform, p):
    """
    Returns the inverse-transform of a point.

    :param sitk.Transform xform: The transform to invert
    :param (float,float)|[float|float]|np.ndarray p: The point to find inverse for
    :return np.ndarray, bool: The point and whether the operation succeeded or not
    """

    def fun(x):
        return np.linalg.norm(xform.TransformPoint(x) - p)

    p = np.array(p)
    res = minimize(fun, p, method='Powell')
    return res.x, res.success

def inverse_transform_multiple_points(xform, points):
    """
    Perform points transformation using xform as a transformation model
    """
    transformed_points = list()
    for point in points:
        result, state = inverse_transform_point(xform, point)
        transformed_points.append(result)
    return transformed_points

def sitk_transform_rgb(moving_rgb_img, fixed_rgb_img, transform, interpolator = sitk.sitkLanczosWindowedSinc):
    """Applies a Simple ITK transform (e.g. Affine, B-spline) to an RGB image
    
    The transform is applied to each channel
    
    Parameters
    ----------
    moving_rgb_img : Pillow Image 
        This image will be transformed to produce the output image
    fixed_rgb_img : Pillow Image
        This reference image provides the output information (spacing, size, and direction) of the output image
    transform : SimpleITK transform
        Generated from image registration
    interpolator : SimpleITK interpolator
    
    Returns
    -------
    rgb_transformed : Pillow Image
        Transformed moving image 
    """
    transformed_channels = []
    r_moving, g_moving, b_moving, = moving_rgb_img.convert('RGB').split()
    r_fixed, g_fixed, b_fixed = fixed_rgb_img.convert('RGB').split()
    for moving_img, fixed_img in [(r_moving, r_fixed), (g_moving, g_fixed), (b_moving, b_fixed)]:
        moving_img_itk = get_itk_from_pil(moving_img)
        fixed_img_itk = get_itk_from_pil(fixed_img)
        transformed_img = sitk.Resample(moving_img_itk, fixed_img_itk, transform, 
                            interpolator, 0.0, moving_img_itk.GetPixelID())
        transformed_channels.append(get_pil_from_itk(transformed_img))
    rgb_transformed = Image.merge('RGB', transformed_channels)
    return rgb_transformed    

def save_transformation_model(transform_model, fn:Union[str, Path]):
    sitk.WriteTransform(transform_model, fn)

def read_transformation_model(fn:str):
    transformer = sitk.ReadTransform(fn)
    return transformer
# bspline_transform = sitk.ReadTransform('BCC_Skin1_r2_registration_bspline_transform.tfm')

def overlay_pil_imgs(foreground, background, best_loc = (0,0), alpha=0.5):
    """ overlay two images to visualize the registration """
    newimg1 = Image.new('RGBA', size=background.size, color=(0, 0, 0, 0))
    newimg1.paste(foreground, best_loc)
    newimg1.paste(background, (0, 0))

    newimg2 = Image.new('RGBA', size=background.size, color=(0, 0, 0, 0))
    newimg2.paste(background, (0, 0))
    newimg2.paste(foreground, best_loc)
    result = Image.blend(newimg1, newimg2, alpha=alpha)
    return result

def get_itk_from_pil(pil_img):
    """Converts Pillow image into ITK image
    """
    return sitk.GetImageFromArray(np.array(pil_img))

def get_pil_from_itk(itk_img):
    """Converts ITK image into Pillow Image
    """
    return Image.fromarray(sitk.GetArrayFromImage(itk_img).astype(np.uint8))