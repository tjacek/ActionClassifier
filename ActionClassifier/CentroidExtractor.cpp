#include "features.h"

void addCentroidExtractor(Dataset * dataset){
  CentroidExtractor* centroidExtractor=new CentroidExtractor();
  dataset->registerExtractor(centroidExtractor);
}

FeatureVector CentroidExtractor::getFeatures(DepthImage image){
  FeatureVector features=new vector<float>();
  features->push_back(0.0);
  features->push_back(0.0);
  return features;
};