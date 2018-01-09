#ifndef TRAIN_H
#define TRAIN_H

#include "QObject"
#include "QThread"
#include "resnet.h"
#include "struct.h"
#include "typedef.h"
#include "io.h"
#include "accuracy.h"
#include "dlibITKConvert.hpp"

#include "dlib/data_io.h"
//#include <dlib/gui_widgets.h>
//#include <dlib/image_io.h>

#include "itkNormalizeImageFilter.h"

class trainDNN: public QObject
{
	Q_OBJECT
public:
	explicit trainDNN(QObject* parent = 0);
	~trainDNN();

	void SetTrainDirectory(QString trainDir);
	void SetTestDirectory(QString testDir);
	void SetTrainStatePath(QString trainStatePath);
	void SetNetworkPath(QString networkPath);
	void SetInitialLearningRate(double initialLearningRate);
	void SetDecayWeight(double decayWeight);
	void SetMomentum(double momentum);
	void SetTrainerVerbose(bool verbose);
	void SetIterationsWithoutProgressThreshold(unsigned int progressThreshold);
	void SetBatchNormalizationStatsWindow(unsigned int bnStatsWindow);
	void SetDataQueueSize(unsigned int dataQueueSize);
	void SetNumberOfDataLoaders(unsigned int numOfDataLoaders);
	void SetImageName(QString imageName);
	void SetLabelName(QString labelName);
	void SetTestInterval(unsigned int testInterval);
	void SetBatchSize(unsigned int batchSize);

	void Train();

public slots:

signals:
	void TrainDataFolderEmpty();

private:
	QString m_trainDir;
	QString m_testDir;
	QString m_trainStatePath = "trainer_state.dat";
	QString m_networkPath = "net.dnn";
	QString m_imageName = "image.nii.gz";
	QString m_labelName = "label.nii.gz";

	// Training parameters
	double m_initialLearningRate = 0.1;
	double m_decayWeight = 0.001;
	double m_momentum = 0.9;
	unsigned int m_progressThreshold = 5000;
	unsigned int m_bnStatsWindow = 1000;
	unsigned int m_dataQueueSize = 200;
	unsigned int m_numOfDataLoaders = QThread::idealThreadCount(); // leave one thread for system usage
	unsigned int m_testInterval = 50;
	unsigned int m_batchSize = 15;

	bool m_trainerVerbose = true;
};

#endif