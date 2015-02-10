#include "features.h"
#include "utils.h"

void addLinearStdExtractor(Dataset * dataset){
  LinearStdExtractor* extractor=new LinearStdExtractor();
  dataset->registerExtractor(extractor);
}

FeatureVector LinearStdExtractor::getFeatures(DepthImage image){
  StdVector stdVector;  
  for(int i=0;i<image->rows;i++){
    const uchar* row_i = image->ptr<uchar>(i);
	Sample sample;
	for(int j = 0; j < image->cols; j++){
	  sample.push_back((float) row_i[j]);
	}
	float std=standardDeviation(sample);
	stdVector.push_back(std);
  }
  Histogram  hist(&stdVector);
  return hist.getFeatures();
}
