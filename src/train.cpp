#include "train.h"

trainDNN::trainDNN(QObject* parent)
{
	
}

trainDNN::~trainDNN()
{

}

void trainDNN::SetTrainDirectory(QString trainDir)
{
	m_trainDir = trainDir;
}

void trainDNN::SetTestDirectory(QString testDir)
{
	m_testDir = testDir;
}

void trainDNN::SetTrainStatePath(QString trainStatePath)
{
	m_trainStatePath = trainStatePath;
}

void trainDNN::SetInitialLearningRate(double initialLearningRate)
{
	m_initialLearningRate = initialLearningRate;
}

void trainDNN::SetDecayWeight(double decayWeight)
{
	m_decayWeight = decayWeight;
}

void trainDNN::SetMomentum(double momentum)
{
	m_momentum = momentum;
}

void trainDNN::SetNumberOfDataLoaders(unsigned int numOfDataLoaders)
{
	m_numOfDataLoaders = numOfDataLoaders;
}

void trainDNN::Train()
{
	net_type net;

	dlib::dnn_trainer<net_type> trainer(net, dlib::sgd(m_decayWeight, m_momentum));
	if (m_trainerVerbose)
		trainer.be_verbose();
	trainer.set_learning_rate(m_initialLearningRate);
	trainer.set_synchronization_file(m_trainStatePath.toStdString(), std::chrono::minutes(10));
	// This threshold is probably excessively large.
	trainer.set_iterations_without_progress_threshold(m_progressThreshold);
	// Since the progress threshold is so large might as well set the batch normalization
	// stats window to something big too.
	set_all_bn_running_stats_window_sizes(net, m_bnStatsWindow);

	// Output training parameters.
	std::cout << trainer << std::endl;

	// prepare data for training
	const auto trainListing = get_image_listing(m_trainDir);
	std::cout << "Images in training dataset = " << trainListing.size() << std::endl;
	if (trainListing.size() == 0)
	{
		std::cout << "Training dataset folder is empty." << std::endl;
		emit TrainDataFolderEmpty();
		return;
	}

	std::vector<dlib::matrix<dlib::bgr_pixel>> images;
	std::vector<dlib::matrix<uint16_t>> labels;

	// Start a bunch of threads that read images from disk and pull out random crops.  It's
	// important to be sure to feed the GPU fast enough to keep it busy.  Using multiple
	// thread for this kind of data preparation helps us do that.  Each thread puts the
	// crops into the data queue.

	dlib::pipe<sample> data(m_dataQueueSize);

	auto f = [&data, &trainListing](time_t seed)
	{
		dlib::rand rnd(time(0) + seed);
		dlib::matrix<dlib::bgr_pixel> input_image;
		dlib::matrix<uint16_t> index_label_image;
		sample temp;
		while (data.is_enabled())
		{
			// Pick a random input image.
			int rnd_pick = rnd.get_random_32bit_number() % trainListing.size();
			const image_info& image_info = trainListing[rnd_pick];

			// Load the input image.
			ImageReaderType::Pointer imageReader = ImageReaderType::New();
			imageReader->SetFileName(image_info.image_filename.toStdString());
			imageReader->Update();

			// Load the label image.
			LabelReaderType::Pointer labelReader = LabelReaderType::New();
			labelReader->SetFileName(image_info.label_filename.toStdString());
			labelReader->Update();

			//std::cout << image_info.image_filename.toStdString() << std::endl;
			//std::cout << image_info.label_filename.toStdString() << std::endl;

			// Randomly pick a part of the image.
			randomly_crop_image<Image3DType::PixelType>(imageReader->GetOutput(), labelReader->GetOutput(), temp, rnd);

			// Push the result to be used by the trainer.
			data.enqueue(temp);
		}
	};

	std::vector<std::thread*> dataLoaderThreads;

	std::cout << "Number of dataloaders = "<< m_numOfDataLoaders << std::endl;

	for (int i = 0; i < m_numOfDataLoaders; i++)
	{
		std::thread* dataLoader = new std::thread([f, i]() {f(i); });
		dataLoaderThreads.push_back(dataLoader);
	}

	std::cout << "Training start" << std::endl;

	// The main training loop.  Keep making mini-batches and giving them to the trainer.
	// We will run until the learning rate has dropped by a factor of 1e-6.
	int iterCount = 0;
	
	while (trainer.get_learning_rate() >= 1e-6)
	{
		images.clear();
		labels.clear();

		// make a n-image mini-batch
		sample temp;
		while (images.size() < 15)
		{
			data.dequeue(temp);

			images.push_back(std::move(temp.image));
			labels.push_back(std::move(temp.label));

			//dlib::image_window my_window1(outputSample.image, "Image");
			//dlib::image_window my_window2(255* outputSample.label, "Label");

			//my_window1.wait_until_closed();
			//my_window2.wait_until_closed();
		}

		trainer.train_one_step(images, labels);

		if (iterCount % m_testInterval == 0 && iterCount != 0 && !m_testDir.isEmpty())
		{
			trainer.get_net();
			net.clean();

			// Make a copy of the network to use it for inference.
			//anet_type anet = net;

			std::cout << "Testing the network..." << std::endl;

			// Find the accuracy of the newly trained network on both the training and the validation sets.
			//std::cout << "Test accuracy = " << calculate_accuracy(anet, get_image_listing(m_testDir)) << std::endl;
		}

		iterCount++;
	}

	// Training done, tell threads to stop and make sure to wait for them to finish before
	// moving on.
	data.disable();

	while (!dataLoaderThreads.empty())
	{
		dataLoaderThreads.front()->join();
		delete dataLoaderThreads.front();
		dataLoaderThreads.erase(dataLoaderThreads.begin());
	}

	// also wait for threaded processing to stop in the trainer.
	trainer.get_net();

	net.clean();
	std::cout << "Saving network..." << std::endl;
	dlib::serialize(m_networkPath.toStdString()) << net;
}

