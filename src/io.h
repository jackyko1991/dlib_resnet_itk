#ifndef IO_H
#define IO_H

#include "struct.h"
#include "typedef.h"
#include "QFileInfo"
#include <QDirIterator>
#include <QDebug>

#include "dlib/image_transforms.h"

std::vector<image_info> get_image_listing(
	QString dataFolder,
	QString imageFileName,
	QString labelFileName);

dlib::rectangle make_random_cropping_rect_resnet(
	const dlib::matrix<float>& img,
	dlib::rand& rnd);



#endif // !IO_H
