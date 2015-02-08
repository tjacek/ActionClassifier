#include "ActionClassifier.h"

extern void addCentroidExtractor(Dataset * dataset);

class CentroidExtractor:public FeatureExtractor{
  public:
    FeatureVector getFeatures(DepthImage image);
};