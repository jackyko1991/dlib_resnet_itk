#ifndef STRUCT_H
#define STRUCT_H
#include "iostream"
#include <dlib/matrix.h>

// The name of input images and assocaited labels in the learning dataset
struct image_info
{
	std::string image_filename;
	std::string label_filename;
};

// A single smaple. A mini-batch comprises man of these
struct sample
{
	dlib::matrix<dlib::bgr_pixel> image;
	dlib::matrix<uint16_t> label;
};

#endif