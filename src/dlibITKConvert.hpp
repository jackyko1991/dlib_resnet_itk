#ifndef DLIBITKCONVERT_H
#define DLIBITKCONVERT_H

#include "resnet.h"
#include "struct.h"
#include "typedef.h"
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

template<typename inputImageType>
void randomly_crop_image(
	typename itk::Image<inputImageType,3>::Pointer image,
	LabelImage3DType::Pointer label, // label must use unsigned char type
	sample& outputSample,
	dlib::rand& rnd)
{
	// Randomly select a layer from the input image
	int rndSliceNum = -1;
	while (rndSliceNum < 0)
	{
		rndSliceNum = rnd.get_integer(image->GetLargestPossibleRegion().GetSize()[2] - 1);
	}

	Image2DType::Pointer image2D = Image2DType::New();
	LabelImage2DType::Pointer label2D = LabelImage2DType::New();
	Image2DType::RegionType region2D;
	Image2DType::IndexType start;

	start[0] = 0;
	start[1] = 0;

	Image2DType::SizeType size;
	size[0] = image->GetLargestPossibleRegion().GetSize()[0];
	size[1] = image->GetLargestPossibleRegion().GetSize()[1];

	region2D.SetSize(size);
	region2D.SetIndex(start);

	image2D->SetRegions(region2D);
	image2D->Allocate();

	label2D->SetRegions(region2D);
	label2D->Allocate();

	ExtractImageFrom3D<inputImageType, inputImageType>(image, image2D, rndSliceNum);
	//std::cout << "extract image from 3D finish" << std::endl;
	ExtractImageFrom3D<LabelImage3DType::PixelType, LabelImage3DType::PixelType>(label, label2D, rndSliceNum);
	//std::cout << "extract label from 3D finish" << std::endl;

	// convert the itk 2D image to openCV mat then to dlib matrix
	cv::Mat imageCV, labelCV;
	ConvertToCVImage<inputImageType>(image2D, imageCV);
	//std::cout << "convert image from itk to opencv finish" << std::endl;
	ConvertToCVImage<LabelImage2DType::PixelType>(label2D, labelCV);
	//std::cout << "convert label from itk to opencv finish" << std::endl;

	// Convert openCV mat to dlib matrix
	cv::Mat imageCVBGR;
	cv::cvtColor(imageCV, imageCVBGR, CV_GRAY2BGR);

	dlib::cv_image<dlib::bgr_pixel> imageDlibCV(imageCVBGR);
	dlib::matrix<dlib::bgr_pixel> imageDlib;
	dlib::assign_image(imageDlib, imageDlibCV);

	dlib::cv_image<unsigned char> labelDlibCV(labelCV);
	dlib::matrix<unsigned char> labelDlibUC;
	dlib::assign_image(labelDlibUC, labelDlibCV);

	// cast label image from unsigned char to uint16_t
	dlib::matrix<uint16_t> labelDlib = dlib::matrix_cast<uint16_t>(labelDlibUC);

	const auto rect = make_random_cropping_rect_resnet(imageDlib, rnd);

	const dlib::chip_details chip_details(rect, dlib::chip_dims(227, 227));

	// Crop the input image.
	dlib::extract_image_chip(imageDlib, chip_details, outputSample.image, dlib::interpolate_bilinear());

	// Crop the labels correspondingly. However, note that here bilinear
	// interpolation would make absolutely no sense - you wouldn't say that
	// a bicycle is half-way between an aeroplane and a bird, would you?
	dlib::extract_image_chip(labelDlib, chip_details, outputSample.label, dlib::interpolate_nearest_neighbor());

	// Also randomly flip the input image and the labels.
	if (rnd.get_random_double() > 0.5)
	{
		outputSample.image = dlib::fliplr(outputSample.image);
		outputSample.label = dlib::fliplr(outputSample.label);
	}

	// And then randomly adjust the colors.
	dlib::apply_random_color_offset(outputSample.image, rnd);

	//dlib::image_window my_window1(outputSample.image, "Image");
	//dlib::image_window my_window2(255* outputSample.label, "Label");

	//my_window1.wait_until_closed();
	//my_window2.wait_until_closed();
}

#endif // !DLIBITKCONVERT_H
