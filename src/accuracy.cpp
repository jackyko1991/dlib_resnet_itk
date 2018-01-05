#include "accuracy.h"

double calculate_accuracy(net_type& net, std::vector<image_info>& dataset)
{
	// Make a copy of the network to use it for testing.
	anet_type anet = net;

	int num_right = 0;
	int num_wrong = 0;

	dlib::matrix<dlib::bgr_pixel> image;
	dlib::matrix<uint16_t> label;
	dlib::matrix<uint16_t> net_output;

	for (const auto& image_info : dataset)
	{
		// Load the input image.
		ImageReaderType::Pointer imageReader = ImageReaderType::New();
		imageReader->SetFileName(image_info.image_filename.toStdString());
		imageReader->Update();

		// Load the label image.
		LabelReaderType::Pointer labelReader = LabelReaderType::New();
		labelReader->SetFileName(image_info.label_filename.toStdString());
		labelReader->Update();

		// Randomly select a layer from the input image
		dlib::rand rnd(time(0));

		int rndSliceNum = -1;
		while (rndSliceNum < 0)
		{
			rndSliceNum = rnd.get_integer(imageReader->GetOutput()->GetLargestPossibleRegion().GetSize()[2] - 1);
		}

		Image2DType::Pointer image2D = Image2DType::New();
		LabelImage2DType::Pointer label2D = LabelImage2DType::New();
		Image2DType::RegionType region2D;
		Image2DType::IndexType start;

		start[0] = 0;
		start[1] = 0;

		Image2DType::SizeType size;
		size[0] = imageReader->GetOutput()->GetLargestPossibleRegion().GetSize()[0];
		size[1] = imageReader->GetOutput()->GetLargestPossibleRegion().GetSize()[1];

		region2D.SetSize(size);
		region2D.SetIndex(start);

		image2D->SetRegions(region2D);
		image2D->Allocate();

		label2D->SetRegions(region2D);
		label2D->Allocate();

		ExtractImageFrom3D<Image3DType::PixelType, Image3DType::PixelType>(imageReader->GetOutput(), image2D, rndSliceNum);
		ExtractImageFrom3D<LabelImage3DType::PixelType, LabelImage3DType::PixelType>(labelReader->GetOutput(), label2D, rndSliceNum);
	
		// convert the itk 2D image to openCV mat then to dlib matrix
		cv::Mat imageCV, labelCV;
		ConvertToCVImage<Image2DType::PixelType>(image2D, imageCV);
		ConvertToCVImage<LabelImage2DType::PixelType>(label2D, labelCV);

		// Convert openCV mat to dlib matrix
		cv::Mat imageCVBGR;
		cv::cvtColor(imageCV, imageCVBGR, CV_GRAY2BGR);

		dlib::cv_image<dlib::bgr_pixel> imageDlibCV(imageCVBGR);
		dlib::matrix<dlib::bgr_pixel> imageDlib;
		dlib::assign_image(imageDlib, imageDlibCV);

		image = imageDlib;

		dlib::cv_image<unsigned char> labelDlibCV(labelCV);
		dlib::matrix<unsigned char> labelDlibUC;
		dlib::assign_image(labelDlibUC, labelDlibCV);

		// cast label image from unsigned char to uint16_t
		dlib::matrix<uint16_t> labelDlib = dlib::matrix_cast<uint16_t>(labelDlibUC);
		label = labelDlib;

		// Create predictions for each pixel. At this point, the type of each prediction
		// is an index (a value between 0 and 20). Note that the net may return an image
		// that is not exactly the same size as the input.
		const dlib::matrix<uint16_t> temp = anet(image);

		//// Convert the indexes to RGB values.
		//rgb_label_image_to_index_label_image(rgb_label_image, index_label_image);

		// Crop the net output to be exactly the same size as the input.
		const dlib::chip_details chip_details(
			dlib::centered_rect(temp.nc() / 2, temp.nr() / 2, image.nc(), image.nr()),
			dlib::chip_dims(image.nr(), image.nc())
		);
		dlib::extract_image_chip(temp, chip_details, net_output, dlib::interpolate_nearest_neighbor());

		// Compare the predicted values to the ground-truth values.
		for (int r = 0; r < label.nr(); ++r)
		{
			for (int c = 0; c < label.nc(); ++c)
			{
				const uint16_t truth = label(r, c);
				if (truth != dlib::loss_multiclass_log_per_pixel_::label_to_ignore)
				{
					const uint16_t prediction = net_output(r, c);
					if (prediction == truth)
					{
						++num_right;
					}
					else
					{
						++num_wrong;
					}
				}
			}
		}
	}

	// Return the accuracy estimate.
	return num_right / static_cast<double>(num_right + num_wrong);
	//return 0;
}
