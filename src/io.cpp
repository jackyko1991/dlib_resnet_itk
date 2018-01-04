#include "io.h"

// Read the list of training image files
std::vector<image_info> get_train_image_listing(QString dataFolder)
{
	return get_image_listing(dataFolder + "/training");
}

// Read the list of testing image files
std::vector<image_info> get_test_image_listing(QString dataFolder)
{
	return get_image_listing(dataFolder + "/testing");
}

// Read the list of image and label pairs
std::vector<image_info> get_image_listing(
	QString dataFolder,
	QString imageFileName,
	QString labelFileName
)
{
	std::vector<image_info> results;

	QFileInfo dataFolderInfo(dataFolder);

	if (!dataFolderInfo.isDir())
	{
		std::cerr << dataFolder.toStdString() << " not exist!!!" << std::endl;
		return results;
	}

	//std::cout << dataFolder << std::endl;

	QDirIterator dataFolderIterator(dataFolder, QDir::Dirs | QDir::NoDotAndDotDot, QDirIterator::Subdirectories);

	while (dataFolderIterator.hasNext())
	{
		dataFolderIterator.next();
		image_info image_info;

		image_info.image_filename = dataFolderIterator.fileInfo().absoluteFilePath() + QString("/") + imageFileName;
		image_info.label_filename = dataFolderIterator.fileInfo().absoluteFilePath() + QString("/") + labelFileName;

		std::cout << image_info.image_filename.toStdString() << std::endl;
		std::cout << image_info.label_filename.toStdString() << std::endl;

		// check if both image file and label file exists
		if (QFileInfo::exists(image_info.image_filename) &&
			QFileInfo::exists(image_info.label_filename))
		{
			results.push_back(image_info);
		}
		else
		{
			std::cerr << dataFolder.toStdString() << " does not contain valid files, i.e. " << imageFileName.toStdString() << " and " << labelFileName.toStdString() << std::endl;
		}
	}

	return results;
}

dlib::rectangle make_random_cropping_rect_resnet(
	const dlib::matrix<dlib::bgr_pixel>& img,
	dlib::rand& rnd
)
{
	// figure out what rectangle we want to crop from the image
	double mins = 0.466666666, maxs = 0.875;
	auto scale = mins + rnd.get_random_double()*(maxs - mins);
	auto size = scale*std::min(img.nr(), img.nc());
	dlib::rectangle rect(size, size);
	// randomly shift the box around
	dlib::point offset(rnd.get_random_32bit_number() % (img.nc() - rect.width()),
		rnd.get_random_32bit_number() % (img.nr() - rect.height()));
	return dlib::move_rect(rect, offset);
}