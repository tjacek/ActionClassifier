#include "ActionClassifier.h"

class SVMClassifier:public Classifier{   
  public:
    SVMClassifier();
    void learn(Labels labels, Dataset* trainingData);
    float predict(DepthImage img);

  private:
    CvSVM * svm;
    CvSVMParams params;
	vector<FeatureExtractor*> extractors;
};

SVMClassifier::SVMClassifier(){
  svm=new CvSVM();
  params.kernel_type=CvSVM::RBF;
  params.svm_type=CvSVM::C_SVC;
  params.gamma=0.50625000000000009;
  params.C=312.50000000000000;
  params.term_crit=cvTermCriteria(CV_TERMCRIT_ITER,100,0.000001);
}
 
float SVMClassifier::predict(DepthImage image){
  ImageDescriptor desc=getImageDescriptor(image, extractors);
  return svm->predict(desc);
}

void SVMClassifier::learn(Labels labels,Dataset* trainingData){
	this->extractors=trainingData->extractors;
    bool res=svm->train(trainingData->toMat(),*labels,cv::Mat(),cv::Mat(),params);
}

