#ifndef DLIBITKCONVERT_H
#define DLIBITKCONVERT_H

#include "resnet.h"
#include "struct.h"
#include "itkImage.h"
#include "itkOpenCVImageBridge.h"
#include "itkImageFileReader.h"

// includes from OpenCV
#include "cv.h"

#if CV_VERSION_MAJOR > 2
#include "opencv2/opencv.hpp" // cv::imwrite
#include <opencv2/core/core.hpp>
#endif

template<typename inputType, typename outputType>
//typedef float inputType;
//typedef float outputType;
void ExtractImageFrom3D(typename itk::Image<inputType, 3>::Pointer Image3D, typename itk::Image<outputType, 2>::Pointer Image2D, unsigned int slice)
{
	// 3D region info
	Image3DType::RegionType region3D;
	Image3DType::IndexType index3D;
	index3D[0] = 0;
	index3D[1] = 0;
	index3D[2] = slice;
	Image3DType::SizeType size3D;
	size3D[0] = Image3D->GetLargestPossibleRegion().GetSize()[0];
	size3D[1] = Image3D->GetLargestPossibleRegion().GetSize()[1];
	size3D[2] = 1;
	region3D.SetIndex(index3D);
	region3D.SetSize(size3D);

	// 2D region info
	Image2DType::RegionType region2D;
	Image2DType::IndexType start;
	start[0] = 0;
	start[1] = 0;

	Image2DType::SizeType size;
	size[0] = Image3D->GetLargestPossibleRegion().GetSize()[0];
	size[1] = Image3D->GetLargestPossibleRegion().GetSize()[1];

	region2D.SetSize(size);;
	region2D.SetIndex(start);

	// iterate over the slice
	typedef itk::ImageRegionIterator<typename itk::Image<inputType, 3>> Iterator3DType;
	typedef itk::ImageRegionIterator<typename itk::Image<outputType, 2>> Iterator2DType;

	Iterator3DType it3D(Image3D, region3D);
	Iterator2DType it2D(Image2D, region2D);
	while (!it3D.IsAtEnd())
	{
		it2D.Set(it3D.Value());
		++it3D;
		++it2D;
	}
};

template<typename inputType>
//typedef float inputType;
void ConvertToCVImage(typename itk::Image<inputType, 2>::Pointer ITK2DImageInput, cv::Mat& cvImageOutput)
{
	typedef itk::Image<inputType, 2> inputImageType;

	// cast and rescale image from float to unsigned char
	typedef itk::RescaleIntensityImageFilter<inputImageType, LabelImage2DType> RescaleFilter2DType;// note that label image is equivalent to unsigned char image

	RescaleFilter2DType::Pointer rescaleFilter = RescaleFilter2DType::New();
	rescaleFilter->SetInput(ITK2DImageInput);

	if (!std::is_same<typename inputType, unsigned char>::value && !std::is_same<typename inputType, unsigned int>::value)
	{
		rescaleFilter->SetOutputMinimum(0);
		rescaleFilter->SetOutputMaximum(255);
	}
	else
	{
		itk::MinimumMaximumImageCalculator<inputImageType>::Pointer minMaxCal = itk::MinimumMaximumImageCalculator<inputImageType>::New();
		minMaxCal->SetImage(ITK2DImageInput);
		minMaxCal->Compute();
		rescaleFilter->SetOutputMinimum(minMaxCal->GetMinimum());
		rescaleFilter->SetOutputMaximum(minMaxCal->GetMaximum());
	}
	rescaleFilter->Update();

	// cast image from itk to opencv
	cv::Mat img = itk::OpenCVImageBridge::ITKImageToCVMat< LabelImage2DType >(rescaleFilter->GetOutput()); // label image 2d type is same as unsigned char 2d type

	cvImageOutput = img.clone();
};


#endif // !DLIBITKCONVERT_H
