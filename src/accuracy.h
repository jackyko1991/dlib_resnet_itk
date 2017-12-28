#ifndef ACCURACY_H
#define ACCURACY_H

#include "resnet.h"
#include "struct.h"
#include "dlibITKConvert.h"

#include "itkImage.h"
#include "itkOpenCVImageBridge.h"
#include "itkImageFileReader.h"

double calculate_accuracy(anet_type& anet, const std::vector<image_info>& dataset);

#endif // ACCURACY_H
