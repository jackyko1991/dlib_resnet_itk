#ifndef TRAIN_H
#define TRAIN_H

#include "QObject"
#include "resnet.h"

class trainDNN: public QObject
{
	Q_OBJECT
public:
	explicit trainDNN(QObject* parent = 0);
	~trainDNN();

	void SetTrainDirectory(QString trainDir);
	void SetTestDirectory(QString testDir);
	void SetTrainStatePath(QString trainStatePath);
	void SetInitialLearningRate(double initialLearningRate);
	void SetDecayWeight(double decayWeight);
	void SetMomemtum(double momentum);
	void SetTrainerVerbose(bool verbose);

	void Train();

public slots:

signals:

private:
	QString m_trainDir;
	QString m_testDir;
	QString m_trainStatePath;

	// Training parameters
	double m_initialLearningRate = 0.1;
	double m_decayWeight = 0.0001;
	double m_momentum = 0.9;

	bool m_trainerVerbose = true;
};

#endif