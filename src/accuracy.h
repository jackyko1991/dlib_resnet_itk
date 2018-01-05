#ifndef ACCURACY_H
#define ACCURACY_H

#include "resnet.h"
#include "struct.h"
#include "typedef.h"
#include "dlibITKConvert.hpp"

#include "itkImage.h"
#include "itkOpenCVImageBridge.h"
#include "itkImageFileReader.h"

#include <dlib/opencv.h>
#include "dlib/data_io.h"
#include "dlib/image_transforms.h"
#include "dlib/dir_nav.h"
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>

// Calculate the per-pixel accuracy on a dataset whose file names are supplied as a parameter.
double calculate_accuracy(net_type& net, std::vector<image_info>& dataset);

#endif // ACCURACY_H
