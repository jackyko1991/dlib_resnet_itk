#ifndef TYPEDEF_H
#define TYPEDEF_H

#include "itkImage.h"
#include "itkImageFileReader.h"
#include <itkRescaleIntensityImageFilter.h>
#include <itkImageRegionIterator.h>


typedef itk::Image<float, 3> Image3DType;
typedef itk::Image<unsigned char, 3> LabelImage3DType;
typedef itk::Image<float, 2> Image2DType;
typedef itk::Image<unsigned char, 2> LabelImage2DType;

typedef itk::ImageFileReader<Image3DType> ImageReaderType;
typedef itk::ImageFileReader<LabelImage3DType> LabelReaderType;

#endif