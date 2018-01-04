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

void trainDNN::SetMomemtum(double momentum)
{
	m_momentum = momentum;
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
	trainer.set_iterations_without_progress_threshold(5000);
	// Since the progress threshold is so large might as well set the batch normalization
	// stats window to something big too.
	set_all_bn_running_stats_window_sizes(net, 1000);

	// Output training parameters.
	std::cout << trainer << std::endl;
}

