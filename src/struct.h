#ifndef STRUCT_H
#define STRUCT_H
#include "iostream"
#include <dlib/matrix.h>
#include "QString"

// The name of input images and assocaited labels in the learning dataset
struct image_info
{
	QString image_filename;
	QString label_filename;
};

// A single smaple. A mini-batch comprises man of these
struct sample
{
	dlib::matrix<float> image;
	dlib::matrix<uint16_t> label;
};

#endif