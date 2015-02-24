#include "features.h"
#include "ShapeContext.h"

void addShapeContextExtractor(Dataset * dataset){
  ShapeContextExtractor* extractor=new ShapeContextExtractor();
  dataset->registerExtractor(extractor);
}

int ShapeContextExtractor::numberOfFeatures(){
  return 20;
}

string ShapeContextExtractor:: featureName(int i){
  string str="shape_context" + intToString(i);
  return str;
}

FeatureVector ShapeContextExtractor::getFeatures(DepthImage image){
/*	OnlineHistogram * hist= getShapeContext(200,&image.image);*/
  FeatureVector fullVect= new vector<double>();
  vector<Mat> projImages= projection(&image.image);
  for(int i=0;i<projImages.size();i++){
    Mat m=projImages.at(i);
	OnlineHistogram * hist= getShapeContext(200,&m);
	FeatureVector newVect=hist->toVector();
	fullVect->insert(fullVect->end(), newVect->begin(), newVect->end());
  }
  cout << "@& "<< fullVect->size();
  return fullVect;
}