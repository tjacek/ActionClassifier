#include "ActionClassifier.h"

extern void addCentroidExtractor(Dataset * dataset);
extern void addLinearStdExtractor(Dataset * dataset);

class CentroidExtractor:public FeatureExtractor{
  public:
    FeatureVector getFeatures(DepthImage image);
};

class LinearStdExtractor:public FeatureExtractor{
  public:
    FeatureVector getFeatures(DepthImage image);
};