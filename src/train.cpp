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

	//std::cout << net << std::endl;
	//layer<3>(net).get_output();
	//// Or to print the prelu parameter for layer 7 we can say:
	//std::cout << "prelu param: " << layer<1>(net).layer_details().get_initial_param_value() << std::endl;
	//system("pause");

	dlib::dnn_trainer<net_type> trainer(net, dlib::sgd(m_decayWeight, m_momentum));
	if (m_trainerVerbose)
		trainer.be_verbose();
	trainer.set_learning_rate(m_initialLearningRate);
	//trainer.set_learning_rate_shrink_factor(0.1);
	trainer.set_synchronization_file(m_trainStatePath.toStdString(), std::chrono::minutes(10));
	// This threshold is probably excessively large.
	trainer.set_iterations_without_progress_threshold(m_progressThreshold);
	// Since the progress threshold is so large might as well set the batch normalization
	// stats window to something big too.
	set_all_bn_running_stats_window_sizes(net, m_bnStatsWindow);

	// Output training parameters.
	std::cout << trainer << std::endl;

	// prepare data for training
	const auto trainListing = get_image_listing(m_trainDir, m_imageName, m_labelName);
	std::cout << "Images in training dataset = " << trainListing.size() << std::endl;
	if (trainListing.size() == 0)
	{
		std::cout << "Training dataset folder is empty." << std::endl;
		emit TrainDataFolderEmpty();
		return;
	}

	std::vector<dlib::matrix<float>> images;
	std::vector<dlib::matrix<uint16_t>> labels;

	// Start a bunch of threads that read images from disk and pull out random crops.  It's
	// important to be sure to feed the GPU fast enough to keep it busy.  Using multiple
	// thread for this kind of data preparation helps us do that.  Each thread puts the
	// crops into the data queue.

	dlib::pipe<sample> data(m_dataQueueSize);

	auto f = [&data, &trainListing](time_t seed)
	{
		dlib::rand rnd(time(0) + seed);
		dlib::matrix<float> input_image;
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

			// perform normalization on the images
			itk::NormalizeImageFilter<Image3DType, Image3DType>::Pointer normalFilter = itk::NormalizeImageFilter<Image3DType, Image3DType>::New();
			normalFilter->SetInput(imageReader->GetOutput());
			normalFilter->Update();

			// resample to isotropic image
			Image3DType::Pointer imageITK = Image3DType::New();
			LabelImage3DType::Pointer labelITK = LabelImage3DType::New();

			//if (imageReader->GetOutput()->GetSpacing()[0] != imageReader->GetOutput()->GetSpacing()[1] ||
			//	imageReader->GetOutput()->GetSpacing()[1] != imageReader->GetOutput()->GetSpacing()[2] ||
			//	imageReader->GetOutput()->GetSpacing()[0] != imageReader->GetOutput()->GetSpacing()[2])
			if (imageReader->GetOutput()->GetSpacing()[0] != imageReader->GetOutput()->GetSpacing()[1])
			{
				Image3DType::SpacingType oldSpacing = imageReader->GetOutput()->GetSpacing();
				Image3DType::SizeType oldSize = imageReader->GetOutput()->GetLargestPossibleRegion().GetSize();

				// using min voxel spacing
				Image3DType::SpacingType newSpacing;
				double minSpacing = 9999999999999.9;
				for (int i = 0; i < 3; i++)
				{
					if (imageReader->GetOutput()->GetSpacing()[i] < minSpacing)
						minSpacing = imageReader->GetOutput()->GetSpacing()[i];
				}

				for (int i = 0; i < 3; i++)
				{
					newSpacing[i] = minSpacing;
				}

				Image3DType::SizeType newSize;
				for (int i = 0; i < 3; i++)
				{
					newSize[i] = ceil(oldSpacing[i] * oldSize[i] / newSpacing[i]);
				}

				// linear interpolator for image
				typedef itk::LinearInterpolateImageFunction< Image3DType> LinearInterpolatorType;
				LinearInterpolatorType::Pointer linInterpolator = LinearInterpolatorType::New();

				itk::ResampleImageFilter<Image3DType, Image3DType>::Pointer imageResampler = itk::ResampleImageFilter<Image3DType, Image3DType>::New();
				imageResampler->SetInput(normalFilter->GetOutput());
				imageResampler->SetOutputSpacing(newSpacing);
				imageResampler->SetSize(newSize);
				imageResampler->SetInterpolator(linInterpolator);
				imageResampler->SetOutputOrigin(normalFilter->GetOutput()->GetOrigin());
				imageResampler->SetOutputDirection(normalFilter->GetOutput()->GetDirection());
				imageResampler->Update();
				imageITK->Graft(imageResampler->GetOutput());
				imageITK->SetMetaDataDictionary(imageResampler->GetOutput()->GetMetaDataDictionary());

				// nearest neighbour interpolator for label
				typedef itk::NearestNeighborInterpolateImageFunction< LabelImage3DType> NNInterpolatorType;
				NNInterpolatorType::Pointer nnInterpolator = NNInterpolatorType::New();

				itk::ResampleImageFilter<LabelImage3DType, LabelImage3DType>::Pointer labelResampler = itk::ResampleImageFilter<LabelImage3DType, LabelImage3DType>::New();
				labelResampler->SetInput(labelReader->GetOutput());
				labelResampler->SetOutputSpacing(newSpacing);
				labelResampler->SetSize(newSize);
				labelResampler->SetInterpolator(nnInterpolator);
				labelResampler->SetOutputOrigin(normalFilter->GetOutput()->GetOrigin());
				labelResampler->SetOutputDirection(normalFilter->GetOutput()->GetDirection());
				labelResampler->Update();
				labelITK->Graft(labelResampler->GetOutput());
				labelITK->SetMetaDataDictionary(labelResampler->GetOutput()->GetMetaDataDictionary());
			}
			else
			{
				imageITK->Graft(normalFilter->GetOutput());
				imageITK->SetMetaDataDictionary(normalFilter->GetOutput()->GetMetaDataDictionary());
				labelITK->Graft(labelReader->GetOutput());
				labelITK->SetMetaDataDictionary(normalFilter->GetOutput()->GetMetaDataDictionary());
			}

			// Randomly pick a part of the image.
			randomly_crop_image<Image3DType::PixelType>(imageITK, labelITK, temp, rnd);

			//std::cout << image_info.image_filename.toStdString() << std::endl;
			//std::cout << image_info.label_filename.toStdString() << std::endl;

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
		while (images.size() < m_batchSize)
		{
			data.dequeue(temp);

			images.push_back(std::move(temp.image));
			labels.push_back(std::move(temp.label));

			//dlib::image_window my_window1(outputSample.image, "Image");
			//dlib::image_window my_window2(255* outputSample.label, "Label");

			//my_window1.wait_until_closed();
			//my_window2.wait_until_closed();
		}

		// save one image out to check if it is correct
		dlib::save_jpeg(images.at(0), "./output/image_train.jpg");
		dlib::save_jpeg(labels.at(0) * 255, "./output/ground_truth_train.jpg");

		trainer.train_one_step(images, labels);
		std::cout << "Finish train step " << trainer.get_train_one_step_calls() << std::endl;

		if (trainer.get_train_one_step_calls() % m_testInterval == 0 && iterCount != 0 && !m_testDir.isEmpty())
		{
			std::cout << "Clean the network..." << std::endl;
			trainer.get_net();
			net.clean();

			std::cout << "Testing the network..." << std::endl;

			 //Find the accuracy of the newly trained network on both the training and the validation sets.
			std::cout << "Test accuracy = " << calculate_accuracy(net, get_image_listing(m_testDir,m_imageName, m_labelName)) << std::endl;
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

void trainDNN::SetBatchSize(unsigned int batchSize)
{
	m_batchSize = batchSize;
}

void trainDNN::SetTestInterval(unsigned int testInterval)
{
	m_testInterval = testInterval;
}

void trainDNN::SetDataQueueSize(unsigned int dataQueueSize)
{
	m_dataQueueSize = dataQueueSize;
}

void trainDNN::SetNetworkPath(QString networkPath)
{
	m_networkPath = networkPath;
}

void trainDNN::SetImageName(QString imageName)
{
	m_imageName = imageName;
}

void trainDNN::SetLabelName(QString labelName)
{
	m_labelName = labelName;
}