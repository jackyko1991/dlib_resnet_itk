#ifndef IO_H
#define IO_H

#include "struct.h"
#include "QFileInfo"
#include <QDirIterator>

std::vector<image_info> get_train_image_listing(QString dataFolder);
std::vector<image_info> get_test_image_listing(QString dataFolder);
std::vector<image_info> get_image_listing(
	QString dataFolder,
	QString imageFileName = "image.nii.gz",
	QString labelFileName = "label.nii.gz");

#endif // !IO_H
