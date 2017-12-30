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

	QDirIterator dataFolderIterator(dataFolder);

	while (dataFolderIterator.hasNext())
	{
		image_info image_info;

		image_info.image_filename = dataFolderIterator.filePath() + QString("/") + imageFileName;
		image_info.label_filename = dataFolderIterator.filePath() + QString("/") + labelFileName;

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