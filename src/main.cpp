//#include "itkImage.h"
//#include "itkImageFileReader.h"
//#include "itkOpenCVImageBridge.h"
//
//#include <dlib/opencv.h>
//#include "dlib/data_io.h"
//#include "dlib/image_transforms.h"
//#include "dlib/dir_nav.h"
//#include <dlib/gui_widgets.h>
//#include <dlib/image_io.h>
//#include "iterator"
//#include "thread"
#include "iostream"
//
//#include "typedef.h"
//#include "resnet.h"
//#include "struct.h"
//#include "accuracy.h"
//#include "io.h"
//#include "dlibITKConvert.hpp"
#include "train.h"

#include "QString"
#include "QCommandLineParser"

//// includes from OpenCV
//#include "cv.h"
//
//#if CV_VERSION_MAJOR > 2
//#include "opencv2/opencv.hpp" // cv::imwrite
//#include <opencv2/core/core.hpp>
//#endif

int main(int argc, char *argv[])
{
	QCoreApplication app(argc, argv);
	QCoreApplication::setApplicationName("DNN-ResNet");
	QCoreApplication::setApplicationVersion("0.1");

	QCommandLineParser parser;
	parser.setApplicationDescription("ResNet Trainer and Evaluation with ITK image i/o");
	parser.addHelpOption();
	parser.addVersionOption();

	parser.addPositionalArgument("train", "Run in train mode");
	parser.addPositionalArgument("eval", "Run in evaluation mode");

	parser.parse(QCoreApplication::arguments());

	const QStringList args = parser.positionalArguments();
	const QString mode = args.isEmpty() ? QString() : args.first();

	if (args.size() < 1)
	{
		parser.showHelp(1);
		return 0;
	}

	if (mode == "train")
	{
		parser.clearPositionalArguments();
		parser.addPositionalArgument("train", "Train the network", "train [train_options]");

		parser.addOption(QCommandLineOption("train_dir","Directory containing training data","dir"));
		parser.addOption(QCommandLineOption("test_dir", "Directory containing testing data", "dir"));
		parser.addOption(QCommandLineOption("train_state", "Path to trainer state file", "path"));

		parser.parse(QCoreApplication::arguments());
		const QStringList trainArgs = parser.positionalArguments();

		if (args.size() < 2)
		{
			parser.showHelp(1);
			return 0;
		}


		// Process actual command line arguments
		parser.process(app);

		// check if everything has set correctly
		if (!parser.isSet("train_dir"))
		{
			std::cerr << "Train directory not set properly" << std::endl;

			parser.showHelp(1);
			return 0;
		}

		if (!parser.isSet("train_state"))
		{
			std::cerr << "Trainer state path not set properly" << std::endl;

			parser.showHelp(1);
			return 0;
		}

		// start training
		trainDNN train;
		train.SetTrainDirectory(parser.value("train_dir"));
		train.SetTrainStatePath(parser.value("train_state"));
		train.Train();
		
	}
	if (mode == "eval")
	{
		std::cout << "evaluation mode" << std::endl;
	}

	//net_type net;

	//
	//// prepare data for training
	//const auto trainListing = get_train_image_listing(dataFolder);
	//std::cout << "Images in training dataset: " << trainListing.size() << std::endl;
	//if (trainListing.size() == 0)
	//{
	//	std::cout << "Training dataset folder is empty." << std::endl;
	//	return 1;
	//}

	//std::vector<dlib::matrix<dlib::bgr_pixel>> images;
	//std::vector<dlib::matrix<uint16_t>> labels;

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
	//		imageReader->SetFileName(image_info.image_filename.toStdString());
	//		imageReader->Update();

	//		// Load the label image.
	//		LabelReaderType::Pointer labelReader = LabelReaderType::New();
	//		labelReader->SetFileName(image_info.label_filename.toStdString());
	//		labelReader->Update();

	////		//std::cout << image_info.image_filename << std::endl;
	////		//std::cout << image_info.label_filename << std::endl;

	//		// Randomly pick a part of the image.
	//		randomly_crop_image<Image3DType::PixelType>(imageReader->GetOutput(), labelReader->GetOutput(), temp, rnd);

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

	//	// make a n-image mini-batch
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

	//	//if (iterCount % 25 == 0 && iterCount != 0)
	//	//{
	//	//	trainer.get_net();
	//	//	net.clean();

	//	//	// Make a copy of the network to use it for inference.
	//	//	anet_type anet = net;

	//	//	std::cout << "Testing the network..." << std::endl;

	//	//	// Find the accuracy of the newly trained network on both the training and the validation sets.
	//	//	std::cout << "Train accuracy: " << calculate_accuracy(anet, get_train_image_listing(dataFolder)) << std::endl;
	//	//}

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