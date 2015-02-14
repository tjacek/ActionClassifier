#include "ActionClassifier.h"

typedef map<string,int> Categories;

extern void evaluate(string imageDir,string categoryFile);
extern Classifier * buildClassifier(ImageList trainingSet,Categories categories);
extern Categories readCategories(string name);
extern Labels getLabels(ImageList imageList,Categories categories );
extern Labels getPredictedLabels(ImageList imageList,Classifier * classifier);
extern int truePositives(Labels  trueLabels,Labels predictedLabels);

class SplitedImages{
  public:
    ImageList training;
    ImageList test;
    SplitedImages(ImageList fullSet);
};

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