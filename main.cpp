#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>

#include <algorithm>
#include <iostream>
#include <math.h>
#include <fstream>
#include <vector>

#include "mlp.h"

/*
 * formatInputDataFromImage
 * 
 * Given a greyscale input image, produce training data. Each row of training data
 * consists of INPUT_LAYER_SIZE values, which are the pixel values from the input
 * image within a WINDOW_WIDTH x WINDOW_WIDTH window.
 */
void formatInputDataFromImage(cv::Mat& trainDataImage, cv::Mat& trainingData)
{
	int height = trainDataImage.rows;
	int width = trainDataImage.cols;
	int numTrainingPoints = width * height;

	int trainDataIndex = 0;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int windowIdx = 0;
			for (int dy = -WINDOW_RADIUS; dy <= WINDOW_RADIUS; dy++)
			{
				for (int dx = -WINDOW_RADIUS; dx <= WINDOW_RADIUS; dx++)
				{
					int srcX = x + dx;
					int srcY = y + dy;
					if (srcX < 0 || srcX >= width || srcY < 0 || srcY >= height)
						trainingData.at<float>(trainDataIndex, windowIdx) = 0;
					else
						trainingData.at<float>(trainDataIndex, windowIdx) = trainDataImage.at<unsigned char>(srcY, srcX) / 255.0;

					windowIdx++;
				}
			}

			trainDataIndex++;
		}
	}
}

/*
 * formatTrainingClassesFromImage
 * 
 * Given a greyscale ground truth input image, produce labels for the training
 * data. These labels correspond to the central pixel in a window.
 */
void formatTrainingClassesFromImage(cv::Mat& trainGroundTruthImage, cv::Mat& trainingClasses)
{
	int height = trainGroundTruthImage.rows;
	int width = trainGroundTruthImage.cols;
	int numTrainingPoints = width * height;

	int trainDataIndex = 0;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			float truth = trainGroundTruthImage.at<float>(y, x);

			trainingClasses.at<float>(trainDataIndex) = (truth > 0.1) ? 1.0 : 0.0;

			trainDataIndex++;
		}
	}
}

int main(int argc, char** argv)
{
	std::string fileNameTrainData = "train_data.tif";
	std::string fileNameTrainGroundTruth = "train_ground_truth.tif";
	std::string fileNameTestData = "train_test_data.tif";

	// Load the specified image
	cv::Mat trainDataImage = cv::imread(fileNameTrainData, CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat trainGroundTruthImage = cv::imread(fileNameTrainGroundTruth, CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat trainTestDataImage = cv::imread(fileNameTestData, CV_LOAD_IMAGE_GRAYSCALE);

	cv::namedWindow("Input", CV_WINDOW_NORMAL);
	cv::imshow("Input", trainDataImage);
	cv::namedWindow("Train", CV_WINDOW_NORMAL);
	cv::imshow("Train", trainGroundTruthImage);
	cv::namedWindow("Test", CV_WINDOW_NORMAL);
	cv::imshow("Test", trainTestDataImage);

	CV_Assert(trainDataImage.depth() == CV_8U);
	CV_Assert(trainDataImage.channels() == 1);
	CV_Assert(trainGroundTruthImage.depth() == CV_8U);
	CV_Assert(trainGroundTruthImage.channels() == 1);
	CV_Assert(trainTestDataImage.depth() == CV_8U);
	CV_Assert(trainTestDataImage.channels() == 1);

	int height = trainDataImage.rows;
	int width = trainDataImage.cols;
	int numTrainingPoints = width * height;

	cv::Mat output(height, width, CV_32FC1);

	// Construct the training data
	cv::Mat trainingData(numTrainingPoints, INPUT_LAYER_SIZE, CV_32FC1);
	cv::Mat trainingClasses(numTrainingPoints, 1, CV_32FC1);

	formatInputDataFromImage(trainDataImage, trainingData);
	formatTrainingClassesFromImage(trainGroundTruthImage, trainingClasses);

	cv::Mat testData(numTrainingPoints, INPUT_LAYER_SIZE, CV_32FC1);
	formatInputDataFromImage(trainTestDataImage, testData);

	CvANN_MLP* pMlp = createMlp();
	
	trainMlp(*pMlp, trainingData, trainingClasses);

	classifyData(*pMlp, testData, output);
	
	delete pMlp;

	// Write the output file
	cv::imwrite("train_output1.png", output);

	cv::namedWindow("Output", CV_WINDOW_NORMAL);
	cv::imshow("Output", output);
	cv::waitKey(0); // Wait for a keystroke in the window

	return 0;
}
