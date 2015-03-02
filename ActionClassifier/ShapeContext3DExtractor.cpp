#include "features.h"
#include "ShapeContext.h"

void addShapeContext3DExtractor(Dataset * dataset){
  ShapeContext3DExtractor* extractor=new ShapeContext3DExtractor();
  dataset->registerExtractor(extractor);
}

int ShapeContext3DExtractor::numberOfFeatures(){
  return 20;
}

string ShapeContext3DExtractor:: featureName(int i){
  string str="shape_context3D" + intToString(i);
  return str;
}

FeatureVector ShapeContext3DExtractor::getFeatures(DepthImage image){
  PointCloud pointCloud(image.image);
  Histogram3D * hist=getShapeContext3D(200,pointCloud);
  return hist->toVector();
}