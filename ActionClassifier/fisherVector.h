#include "ActionClassifier.h"
#include "css.h"

using cv::EM;
using cv::TermCriteria;
using cv::noArray;
using cv::Point;

class GMM{
  public:
    EM * em_model;
	Mat probs;
	vector<Mat> covs;
	Mat means;
	Mat x;
	Mat weights;
	int k;

	GMM(int k);
	void learn(Mat samples);
	double assigment(int i,int j);
	Point gamma_ex(int cls);
    Point gamma_var(int cls);

};

extern vector<float>* getFisherVector(Mat * image,int k);
extern vector<float> *extractFisherVector(GMM * gmm);