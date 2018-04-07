#include <opencv2/core.hpp>
#include <opencv2/ml.hpp>

namespace
{
	extern const int WINDOW_RADIUS;
	extern const int WINDOW_WIDTH;
	extern const int INPUT_LAYER_SIZE;
	extern const int HIDDEN_LAYER_SIZE;
}

/**
 * createMlp - Creates a Multilayer perceptron neural network
 * 
 * @return A pointer to a new MLP neural network
 */
CvANN_MLP* createMlp();


/**
 * trainMlp - Trains a Multilayer perceptron neural network
 */
void trainMlp(CvANN_MLP& mlp, cv::Mat& trainingData, cv::Mat& trainingClasses);


/**
 * classifyData - Classifies data using a trained MLP neural network
 */
void classifyData(CvANN_MLP& mlp, cv::Mat& testData, cv::Mat& output);
