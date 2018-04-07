#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>

#include <algorithm>
#include <iostream>
#include <math.h>

#include "mlp.h"

namespace
{
	const int WINDOW_RADIUS = 1;
	const int WINDOW_WIDTH = 2 * WINDOW_RADIUS + 1;
	const int INPUT_LAYER_SIZE = WINDOW_WIDTH * WINDOW_WIDTH;
	const int HIDDEN_LAYER_SIZE = 2 * INPUT_LAYER_SIZE / 3 + 1;		// The number of hidden neurons should be 2 / 3 the size of the input layer, plus the size of the output layer.
}

/**
 * createMlp - Creates a Multilayer perceptron neural network
 * 
 * @return A pointer to a new MLP neural network
 */
CvANN_MLP* createMlp()
{
	std::cout << "Creating the neural network" << std::endl;

	// Define the layers of the neural network
	cv::Mat layers = cv::Mat(4, 1, CV_32SC1);
	layers.row(0) = cv::Scalar(INPUT_LAYER_SIZE);		// Input is a 3x3 square of pixels, with the pixel to classify at the center

	layers.row(1) = cv::Scalar(HIDDEN_LAYER_SIZE);		// The number of hidden neurons should be 2 / 3 the size of the input layer, plus the size of the output layer.
	layers.row(2) = cv::Scalar(HIDDEN_LAYER_SIZE);

	layers.row(3) = cv::Scalar(1);		// Output is the category of the center pixel

	// Create the neural network
	CvANN_MLP* pMlp = new CvANN_MLP;
	pMlp->create(layers);
	
	return pMlp;
}

/**
 * trainMlp - Trains a Multilayer perceptron neural network
 */
void trainMlp(CvANN_MLP& mlp, cv::Mat& trainingData, cv::Mat& trainingClasses)
{
	// Define the termination criteria for the training
	CvTermCriteria criteria;
	criteria.max_iter = 200;
	criteria.epsilon = 0.00001f;
	criteria.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;

	// Define the training method (back propagation)
	CvANN_MLP_TrainParams params;
	params.train_method = CvANN_MLP_TrainParams::BACKPROP;
	params.bp_dw_scale = 0.05f;
	params.bp_moment_scale = 0.05f;
	params.term_crit = criteria;

	// Train the neural network, assigning weights based on the training data
	std::cout << "Training the neural network" << std::endl;
	mlp.train(trainingData, trainingClasses, cv::Mat(), cv::Mat(), params);
}

/**
 * classifyData - Classifies data using a trained MLP neural network
 */
void classifyData(CvANN_MLP& mlp, cv::Mat& testData, cv::Mat& output)
{
	// Use the neural network to predict based on new data
	CV_Assert(testData.rows == output.rows * output.cols);

	std::cout << "Using the trained neural network to classify pixels" << std::endl;
	int i = 0;
	for (int y = 0; y < output.rows; y++)
	{
		for (int x = 0; x < output.cols; x++)
		{
			cv::Mat response(1, 1, CV_32FC1);
			cv::Mat sample = testData.row(i);

			mlp.predict(sample, response);

			float r = response.at<float>(0, 0);
			output.at<float>(y, x) = r;
			i++;
		}
	}
}
