#ifndef ACCURACY_H
#define ACCURACY_H

#include "resnet.h"
#include "struct.h"
#include "dlibITKConvert.hpp"

#include "itkImage.h"
#include "itkOpenCVImageBridge.h"
#include "itkImageFileReader.h"

// Calculate the per-pixel accuracy on a dataset whose file names are supplied as a parameter.
double calculate_accuracy(anet_type& anet, const std::vector<image_info>& dataset);

#endif // ACCURACY_H
