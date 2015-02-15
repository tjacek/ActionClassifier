#pragma once
#include "ActionClassifier.h"
#include "io.h"

extern void addCentroidExtractor(Dataset * dataset);
extern void addLinearStdExtractor(Dataset * dataset);
extern void addPcaExtractor(Dataset * dataset);

class CentroidExtractor:public FeatureExtractor{
  public:
    int numberOfFeatures();
    string featureName(int i);
	FeatureVector getFeatures(DepthImage image);
};

class LinearStdExtractor:public FeatureExtractor{
  public:
    int numberOfFeatures();
    string featureName(int i);
    FeatureVector getFeatures(DepthImage image);
};

class PcaExtractor:public FeatureExtractor{
  public:
    int numberOfFeatures();
    string featureName(int i);
    FeatureVector getFeatures(DepthImage image);
};