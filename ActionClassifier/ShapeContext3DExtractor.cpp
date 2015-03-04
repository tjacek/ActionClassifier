#include "features.h"
#include "ShapeContext.h"
#include "transform.h"

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
	cout << image.name <<"\n";
  PointCloud pointCloud(cleanEdge(&image.image));

  Histogram3D * hist=getShapeContext3D(500,pointCloud);
  return hist->toVector();
}