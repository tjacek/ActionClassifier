#pragma once
#include "ActionClassifier.h"
#include "io.h"

extern void addPointCloudExtractor(Dataset * dataset);
extern void addShapeContextExtractor(Dataset * dataset);
extern void addShapeContext3DExtractor(Dataset * dataset);

class PointCloudExtractor:public FeatureExtractor{
  public:
    int numberOfFeatures();
    string featureName(int i);
    FeatureVector getFeatures(DepthImage image);
};

class ShapeContextExtractor:public FeatureExtractor{
  public:
    int numberOfFeatures();
    string featureName(int i);
    FeatureVector getFeatures(DepthImage image);
};

class ShapeContext3DExtractor:public FeatureExtractor{
  public:
    int numberOfFeatures();
    string featureName(int i);
    FeatureVector getFeatures(DepthImage image);
};