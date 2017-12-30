#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkOpenCVImageBridge.h"

#include <dlib/opencv.h>
#include "dlib/data_io.h"
#include "dlib/image_transforms.h"
#include "dlib/dir_nav.h"
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include "iterator"
#include "thread"
#include "iostream"

#include "typedef.h"
#include "resnet.h"
#include "struct.h"
#include "accuracy.h"
#include "io.h"

#include "QString"

// includes from OpenCV
#include "cv.h"

#if CV_VERSION_MAJOR > 2
#include "opencv2/opencv.hpp" // cv::imwrite
#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
#endif

//
//
//dlib::rectangle make_random_cropping_rect_resnet(
//	const dlib::matrix<dlib::bgr_pixel>& img,
//	dlib::rand& rnd
//)
//{
//	// figure out what rectangle we want to crop from the image
//	double mins = 0.466666666, maxs = 0.875;
//	auto scale = mins + rnd.get_random_double()*(maxs - mins);
//	auto size = scale*std::min(img.nr(), img.nc());
//	dlib::rectangle rect(size, size);
//	// randomly shift the box around
//	dlib::point offset(rnd.get_random_32bit_number() % (img.nc() - rect.width()),
//		rnd.get_random_32bit_number() % (img.nr() - rect.height()));
//	return dlib::move_rect(rect, offset);
//}
//
//
//void randomly_crop_image(
//	Image3DType::Pointer image,
//	LabelImage3DType::Pointer label,
//	sample& outputSample,
//	dlib::rand& rnd
//)
//{
//	// Randomly select a layer from the input image
//	int rndSliceNum = -1;
//	while (rndSliceNum < 0)
//	{
//		rndSliceNum = rnd.get_integer(image->GetLargestPossibleRegion().GetSize()[2] - 1);
//	}
//
//	Image2DType::Pointer image2D = Image2DType::New();
//	LabelImage2DType::Pointer label2D = LabelImage2DType::New();
//	Image2DType::RegionType region2D;
//	Image2DType::IndexType start;
//
//	start[0] = 0;
//	start[1] = 0;
//
//	Image2DType::SizeType size;
//	size[0] = image->GetLargestPossibleRegion().GetSize()[0];
//	size[1] = image->GetLargestPossibleRegion().GetSize()[1];
//
//	region2D.SetSize(size);
//	region2D.SetIndex(start);
//
//	image2D->SetRegions(region2D);
//	image2D->Allocate();
//
//	label2D->SetRegions(region2D);
//	label2D->Allocate();
//
//	ExtractImageFrom3D<Image3DType::PixelType, Image3DType::PixelType>(image, image2D, rndSliceNum);
//	//std::cout << "extract image from 3D finish" << std::endl;
//	ExtractImageFrom3D<LabelImage3DType::PixelType, LabelImage3DType::PixelType>(label, label2D, rndSliceNum);
//	//std::cout << "extract label from 3D finish" << std::endl;
//
//	// convert the itk 2D image to openCV mat then to dlib matrix
//	cv::Mat imageCV, labelCV;
//	ConvertToCVImage<Image2DType::PixelType>(image2D, imageCV);
//	//std::cout << "convert image from itk to opencv finish" << std::endl;
//	ConvertToCVImage<LabelImage2DType::PixelType>(label2D, labelCV);
//	//std::cout << "convert label from itk to opencv finish" << std::endl;
//
//	// Convert openCV mat to dlib matrix
//	cv::Mat imageCVBGR;
//	cv::cvtColor(imageCV, imageCVBGR, CV_GRAY2BGR);
//	
//	dlib::cv_image<dlib::bgr_pixel> imageDlibCV(imageCVBGR);
//	dlib::matrix<dlib::bgr_pixel> imageDlib;
//	dlib::assign_image(imageDlib, imageDlibCV);
//
//	dlib::cv_image<unsigned char> labelDlibCV(labelCV);
//	dlib::matrix<unsigned char> labelDlibUC;
//	dlib::assign_image(labelDlibUC, labelDlibCV);
//
//	// cast label image from unsigned char to uint16_t
//	dlib::matrix<uint16_t> labelDlib = dlib::matrix_cast<uint16_t>(labelDlibUC);
//
//	const auto rect = make_random_cropping_rect_resnet(imageDlib, rnd);
//
//	const dlib::chip_details chip_details(rect, dlib::chip_dims(227, 227));
//
//	// Crop the input image.
//	dlib::extract_image_chip(imageDlib, chip_details, outputSample.image, dlib::interpolate_bilinear());
//
//	// Crop the labels correspondingly. However, note that here bilinear
//	// interpolation would make absolutely no sense - you wouldn't say that
//	// a bicycle is half-way between an aeroplane and a bird, would you?
//	dlib::extract_image_chip(labelDlib, chip_details, outputSample.label, dlib::interpolate_nearest_neighbor());
//
//	// Also randomly flip the input image and the labels.
//	if (rnd.get_random_double() > 0.5)
//	{
//		outputSample.image = dlib::fliplr(outputSample.image);
//		outputSample.label = dlib::fliplr(outputSample.label);
//	}
//
//	// And then randomly adjust the colors.
//	dlib::apply_random_color_offset(outputSample.image, rnd);
//
//	//dlib::image_window my_window1(outputSample.image, "Image");
//	//dlib::image_window my_window2(255* outputSample.label, "Label");
//
//	//my_window1.wait_until_closed();
//	//my_window2.wait_until_closed();
//}



int main()
{
	QString dataFolder("D:/Projects/dlib_resnet_itk/data_crop");
	QString trainerState("D:/Projects/dlib_resnet_itk/trainer_state.dat");

	// training parameter define
	const double intial_learning_rate = 0.1;
	const double weight_deacy = 0.0001;
	const double momentum = 0.9;

	std::cout << "111" << std::endl;

	net_type net;

	std::cout << "222" << std::endl;

	dlib::dnn_trainer<net_type> trainer(net, dlib::sgd(weight_deacy, momentum));
	trainer.be_verbose();
	trainer.set_learning_rate(intial_learning_rate);
	trainer.set_synchronization_file(trainerState.toStdString(), std::chrono::minutes(10));
	// This threshold is probably excessively large.
	trainer.set_iterations_without_progress_threshold(5000);
	// Since the progress threshold is so large might as well set the batch normalization
	// stats window to something big too.
	set_all_bn_running_stats_window_sizes(net, 1000);

	std::cout << "aaa" << std::endl;

	// Output training parameters.
	std::cout << trainer << std::endl;
	
	// prepare data for training
	const auto trainListing = get_train_image_listing(dataFolder);
	std::cout << "Images in training dataset: " << trainListing.size() << std::endl;
	if (trainListing.size() == 0)
	{
		std::cout << "Training dataset folder is empty." << std::endl;
		return 1;
	}

	std::vector<dlib::matrix<dlib::bgr_pixel>> images;
	std::vector<dlib::matrix<uint16_t>> labels;

	//// Start a bunch of threads that read images from disk and pull out random crops.  It's
	//// important to be sure to feed the GPU fast enough to keep it busy.  Using multiple
	//// thread for this kind of data preparation helps us do that.  Each thread puts the
	//// crops into the data queue.

	//dlib::pipe<sample> data(200);

	//auto f = [&data, &trainListing](time_t seed)
	//{
	//	dlib::rand rnd(time(0) + seed);
	//	dlib::matrix<dlib::bgr_pixel> input_image;
	//	dlib::matrix<uint16_t> index_label_image;
	//	sample temp;
	//	while (data.is_enabled())
	//	{
	//		// Pick a random input image.
	//		int rnd_pick = rnd.get_random_32bit_number() % trainListing.size();
	//		const image_info& image_info = trainListing[rnd_pick];

	//		// Load the input image.
	//		ImageReaderType::Pointer imageReader = ImageReaderType::New();
	//		imageReader->SetFileName(image_info.image_filename);
	//		imageReader->Update();

	//		// Load the label image.
	//		LabelReaderType::Pointer labelReader = LabelReaderType::New();
	//		labelReader->SetFileName(image_info.label_filename);
	//		labelReader->Update();

	//		//std::cout << image_info.image_filename << std::endl;
	//		//std::cout << image_info.label_filename << std::endl;

	//		// Randomly pick a part of the image.
	//		randomly_crop_image(imageReader->GetOutput(), labelReader->GetOutput(), temp, rnd);

	//		// Push the result to be used by the trainer.
	//		data.enqueue(temp);
	//	}
	//};
	//std::thread data_loader1([f]() { f(1); });
	//std::thread data_loader2([f]() { f(2); });
	//std::thread data_loader3([f]() { f(3); });
	//std::thread data_loader4([f]() { f(4); });

	//std::cout << "Training start" << std::endl;

	//// The main training loop.  Keep making mini-batches and giving them to the trainer.
	//// We will run until the learning rate has dropped by a factor of 1e-6.
	//int iterCount = 0;
	//
	//while (trainer.get_learning_rate() >= 1e-6)
	//{
	//	images.clear();
	//	labels.clear();

	//	// make a 30-image mini-batch
	//	sample temp;
	//	while (images.size() < 15)
	//	{
	//		data.dequeue(temp);

	//		images.push_back(std::move(temp.image));
	//		labels.push_back(std::move(temp.label));

	//		//dlib::image_window my_window1(outputSample.image, "Image");
	//		//dlib::image_window my_window2(255* outputSample.label, "Label");

	//		//my_window1.wait_until_closed();
	//		//my_window2.wait_until_closed();
	//	}

	//	trainer.train_one_step(images, labels);

	//	if (iterCount % 25 == 0 && iterCount != 0)
	//	{
	//		trainer.get_net();
	//		net.clean();

	//		// Make a copy of the network to use it for inference.
	//		anet_type anet = net;

	//		std::cout << "Testing the network..." << std::endl;

	//		// Find the accuracy of the newly trained network on both the training and the validation sets.
	//		std::cout << "Train accuracy: " << calculate_accuracy(anet, get_train_image_listing(dataFolder.string())) << std::endl;
	//	}

	//	iterCount++;
	//}

	//// Training done, tell threads to stop and make sure to wait for them to finish before
	//// moving on.
	//data.disable();
	//data_loader1.join();
	//data_loader2.join();
	//data_loader3.join();
	//data_loader4.join();

	//// also wait for threaded processing to stop in the trainer.
	//trainer.get_net();

	//net.clean();
	//std::cout << "Saving network..." << std::endl;
	//dlib::serialize("D:/Projects/dlib_resnet/net.dnn") << net;

	return 0;
}