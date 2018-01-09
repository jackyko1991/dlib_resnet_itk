#include "accuracy.h"

double calculate_accuracy(net_type& net, std::vector<image_info>& dataset)
{
	// Make a copy of the network to use it for testing.
	anet_type anet = net;

	int num_right = 0;
	int num_wrong = 0;

	dlib::matrix<float> image;
	dlib::matrix<uint16_t> label;
	//dlib::matrix<uint16_t> net_output;

	//dlib::image_window win;

	for (const auto& image_info : dataset)
	{
		// Load the input image.
		ImageReaderType::Pointer imageReader = ImageReaderType::New();
		imageReader->SetFileName(image_info.image_filename.toStdString());
		imageReader->Update();

		// perform normalization on the images
		itk::NormalizeImageFilter<Image3DType, Image3DType>::Pointer normalFilter = itk::NormalizeImageFilter<Image3DType, Image3DType>::New();
		normalFilter->SetInput(imageReader->GetOutput());
		normalFilter->Update();

		// Load the label image.
		LabelReaderType::Pointer labelReader = LabelReaderType::New();
		labelReader->SetFileName(image_info.label_filename.toStdString());
		labelReader->Update();

		// resample to isotropic image
		Image3DType::Pointer imageITK = Image3DType::New();
		LabelImage3DType::Pointer labelITK = LabelImage3DType::New();

		if (imageReader->GetOutput()->GetSpacing()[0] != imageReader->GetOutput()->GetSpacing()[1] ||
			imageReader->GetOutput()->GetSpacing()[1] != imageReader->GetOutput()->GetSpacing()[2] || 
			imageReader->GetOutput()->GetSpacing()[0] != imageReader->GetOutput()->GetSpacing()[2])
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

		//std::cout << "===========================================" << std::endl;
		//std::cout << "image" << std::endl;
		//normalFilter->GetOutput()->Print(std::cout);
		//std::cout << "image after resample" << std::endl;
		//imageITK->Print(std::cout);

		// Randomly select a layer from the input image
		dlib::rand rnd(time(0));

		int rndSliceNum = -1;
		while (rndSliceNum < 0)
		{
			rndSliceNum = rnd.get_integer(labelITK->GetLargestPossibleRegion().GetSize()[2] - 1);
		}
		//std::cout << image_info.image_filename.toStdString() << std::endl;
		//std::cout << "Selected Slice: " << rndSliceNum << std::endl;

		Image2DType::Pointer image2D = Image2DType::New();
		LabelImage2DType::Pointer label2D = LabelImage2DType::New();
		Image2DType::RegionType region2D;
		Image2DType::IndexType start;

		start[0] = 0;
		start[1] = 0;

		Image2DType::SizeType size;
		size[0] = imageITK->GetLargestPossibleRegion().GetSize()[0];
		size[1] = imageITK->GetLargestPossibleRegion().GetSize()[1];

		region2D.SetSize(size);
		region2D.SetIndex(start);

		image2D->SetRegions(region2D);
		image2D->Allocate();

		label2D->SetRegions(region2D);
		label2D->Allocate();

		ExtractImageFrom3D<Image3DType::PixelType, Image3DType::PixelType>(imageITK, image2D, rndSliceNum);
		ExtractImageFrom3D<LabelImage3DType::PixelType, LabelImage3DType::PixelType>(labelITK, label2D, rndSliceNum);

		// convert the itk 2D image to openCV mat then to dlib matrix
		cv::Mat imageCV, labelCV;
		ConvertToCVImage<Image2DType::PixelType, float>(image2D, imageCV);
		ConvertToCVImage<LabelImage2DType::PixelType, unsigned char>(label2D, labelCV);

		// Convert openCV mat to dlib matrix
		//cv::Mat imageCVBGR;
		//cv::cvtColor(imageCV, imageCVBGR, CV_GRAY2BGR);

		dlib::cv_image<float> imageDlibCV(imageCV);
		dlib::matrix<float> imageDlib;
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
		dlib::matrix<uint16_t> net_output;
		dlib::extract_image_chip(temp, chip_details, net_output, dlib::interpolate_nearest_neighbor());

		// visualize the result
		// convert net_output to opencv mat
		//cv::Mat net_output_CV = dlib::toMat(net_output);

		//cv::Mat matDst(cv::Size(imageCV.cols * 3, imageCV.rows), imageCV.type(), cv::Scalar::all(0));
		//cv::Mat matRoi = matDst(cv::Rect(0,0, imageCV.cols, imageCV.rows));
		//imageCV.copyTo(matRoi);
		//matRoi = matDst(cv::Rect(imageCV.cols, 0, imageCV.cols, imageCV.rows));
		//double labelMin, labelMax;
		//cv::minMaxLoc(labelCV, &labelMin, &labelMax);
		//labelCV = 255 * labelCV; // for better visualization
		//labelCV.copyTo(matRoi);
		//matRoi = matDst(cv::Rect(imageCV.cols*2, 0, imageCV.cols, imageCV.rows));
		//double net_outputMin, net_outputMax;
		//cv::minMaxLoc(net_output_CV, &net_outputMin, &net_outputMax);
		//std::cout << "output min: " << net_outputMin << ", max: " << net_outputMax << std::endl;
		//net_output_CV = 255 * net_output_CV;
		//net_output_CV.copyTo(matRoi); // for better visualization

		////imshow("result", matDst);
		////cv::waitKey(0.1);
		/*imwrite("./test.jpg", matDst);*/

		// use dlib window to visualize image
		//win.set_image(imageDlib);
		//Sleep(500); // in microseconds

		dlib::save_jpeg(imageDlib,"./output/image.jpg");
		dlib::save_jpeg(label*255,"./output/ground_truth.jpg");
		dlib::save_jpeg(net_output *255,"./output/output.jpg");

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
