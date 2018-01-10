#include "iostream"
#include "train.h"

#include "QString"
#include "QCommandLineParser"

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
		parser.addOption(QCommandLineOption("network", "Path to save the trained network", "path"));
		parser.addOption(QCommandLineOption("lr", "Initial learning rate (default = 0.1)", "double"));
		parser.addOption(QCommandLineOption("decay_weight", "Decay weight (default = 0.001)", "double"));
		parser.addOption(QCommandLineOption("momentum", "Momentum (default = 0.9)", "double"));
		parser.addOption(QCommandLineOption("verbose", "Trainer verbose print out (default = true)", "bool"));
		parser.addOption(QCommandLineOption("prog_thres","Iterations without progress threshold (default = 5000)", "unsigned int"));
		parser.addOption(QCommandLineOption("bn_window", "Batch normalization statistics window size (default = 1000)", "unsigned int"));
		parser.addOption(QCommandLineOption("queue_size", "Max size of data queue (default = 200)", "unsigned int"));
		parser.addOption(QCommandLineOption("threads", "Max number of data loader threads (default = number of cpu threads)", "unsigned int"));
		parser.addOption(QCommandLineOption("image_name", "Image filename (default = image.nii.gz)", "string"));
		parser.addOption(QCommandLineOption("label_name", "Label filename (default = label.nii.gz)", "string"));
		parser.addOption(QCommandLineOption("interval", "Test interval (default = 50)", "unsigned int"));
		parser.addOption(QCommandLineOption("batch_size", "Batch size (default = 15)","unsigned int"));

		parser.parse(QCoreApplication::arguments());
		const QStringList trainArgs = parser.positionalArguments();

		if (args.size() < 2)
		{
			parser.showHelp(1);
			return 0;
		}


		// Process actual command line arguments
		parser.process(app);

		// check if everything has set correctly and parse values to trainer
		trainDNN train;

		if (parser.isSet("train_dir"))
		{
			train.SetTrainDirectory(parser.value("train_dir"));
		}
		else
		{
			std::cerr << "Train directory not set properly" << std::endl;
			return 0;
		}

		if (parser.isSet("test_dir"))
		{
			train.SetTestDirectory(parser.value("test_dir"));
		}

		if (parser.isSet("train_state"))
		{
			train.SetTrainStatePath(parser.value("train_state"));
		}

		if (parser.isSet("network"))
		{
			train.SetNetworkPath(parser.value("network"));
		}

		if (parser.isSet("image_name"))
		{
			train.SetImageName(parser.value("image_name"));
		}

		if (parser.isSet("label_name"))
		{
			train.SetLabelName(parser.value("label_name"));
		}

		if (parser.isSet("lr"))
		{
			if (parser.value("lr").toDouble())
			{
				train.SetInitialLearningRate(parser.value("lr").toDouble());
			}
			else
			{
				std::cerr << "Learning rate shoule be a numeric number" << std::endl;
				return 0;
			}
		}

		if (parser.isSet("decay_weight"))
		{
			if (parser.value("decay_weight").toDouble())
			{
				train.SetDecayWeight(parser.value("decay_weight").toDouble());
			}
			else
			{
				std::cerr << "Decay weight shoule be a numeric number" << std::endl;
				return 0;
			}
		}

		if (parser.isSet("momentum"))
		{
			if (parser.value("momentum").toDouble())
			{
				train.SetMomentum(parser.value("momentum").toDouble());
			}
			else
			{
				std::cerr << "Momentum shoule be a numeric number" << std::endl;
				return 0;
			}
		}

		if (parser.isSet("threads"))
		{
			if (parser.value("threads").toInt() && parser.value("threads").toInt() > 0)
			{
				train.SetNumberOfDataLoaders(parser.value("threads").toInt());
			}
			else
			{
				std::cerr << "Number of threads should be an integer" << std::endl;
				return 0;
			}
		}

		if (parser.isSet("interval"))
		{
			if (parser.value("interval").toInt() && parser.value("interval").toInt() > 0)
			{
				train.SetTestInterval(parser.value("interval").toInt());
			}
			else
			{
				std::cerr << "Test interval shoule be an integer" << std::endl;
				return 0;
			}
		}

		if (parser.isSet("batch_size"))
		{
			if (parser.value("batch_size").toInt() && parser.value("batch_size").toInt() > 0)
			{
				train.SetBatchSize(parser.value("batch_size").toInt());
			}
			else
			{
				std::cerr << "Batch size shoule be an integer" << std::endl;
				return 0;
			}
		}

		if (parser.isSet("queue_size"))
		{
			if (parser.value("queue_size").toInt() && parser.value("queue_size").toInt() > 0)
			{
				train.SetDataQueueSize(parser.value("queue_size").toInt());
			}
			else
			{
				std::cerr << "Data queue size shoule be an integer" << std::endl;
				return 0;
			}
		}

		// connect signals emitted from the training class
		QObject::connect(&train, &trainDNN::TrainDataFolderEmpty, []() {exit(0); });

		// start training
		train.Train();

		
	}
	//if (mode == "eval")
	//{
	//	std::cout << "evaluation mode" << std::endl;

	//	parser.clearPositionalArguments();
	//	parser.addPositionalArgument("train", "Train the network", "train [train_options]");

	//	parser.addOption(QCommandLineOption("train_dir", "Directory containing training data", "dir"));
	//	parser.addOption(QCommandLineOption("test_dir", "Directory containing testing data", "dir"));
	//	parser.addOption(QCommandLineOption("train_state", "Path to trainer state file", "path"));
	//	parser.addOption(QCommandLineOption("network", "Path to save the trained network", "path"));
	//	parser.addOption(QCommandLineOption("lr", "Initial learning rate (default = 0.1)", "double"));
	//	parser.addOption(QCommandLineOption("decay_weight", "Decay weight (default = 0.001)", "double"));
	//	parser.addOption(QCommandLineOption("momentum", "Momentum (default = 0.9)", "double"));
	//	parser.addOption(QCommandLineOption("verbose", "Trainer verbose print out (default = true)", "bool"));
	//	parser.addOption(QCommandLineOption("prog_thres", "Iterations without progress threshold (default = 5000)", "unsigned int"));
	//	parser.addOption(QCommandLineOption("bn_window", "Batch normalization statistics window size (default = 1000)", "unsigned int"));
	//	parser.addOption(QCommandLineOption("queue_size", "Max size of data queue (default = 200)", "unsigned int"));
	//	parser.addOption(QCommandLineOption("threads", "Max number of data loader threads (default = number of cpu threads)", "unsigned int"));
	//	parser.addOption(QCommandLineOption("image_name", "Image filename (default = image.nii.gz)", "string"));
	//	parser.addOption(QCommandLineOption("label_name", "Label filename (default = label.nii.gz)", "string"));
	//	parser.addOption(QCommandLineOption("interval", "Test interval (default = 50)", "unsigned int"));
	//	parser.addOption(QCommandLineOption("batch_size", "Batch size (default = 15)", "unsigned int"));

	//	parser.parse(QCoreApplication::arguments());
	//	const QStringList trainArgs = parser.positionalArguments();

	//	if (args.size() < 2)
	//	{
	//		parser.showHelp(1);
	//		return 0;
	//	}


	//	// Process actual command line arguments
	//	parser.process(app);

	//	// check if everything has set correctly and parse values to trainer
	//	trainDNN train;

	//}


	return app.exec();
}