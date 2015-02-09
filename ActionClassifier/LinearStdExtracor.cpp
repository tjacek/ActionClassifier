#include "features.h"
#include "utils.h"

void addLinearStdExtractor(Dataset * dataset){
  LinearStdExtractor* extractor=new LinearStdExtractor();
  dataset->registerExtractor(extractor);
}

FeatureVector LinearStdExtractor::getFeatures(DepthImage image){
  StdVector stdVector;  
  for(int i=0;i<image->rows;i++){
    const float* row_i = image->ptr<float>(i);
	Sample sample;
	for(int j = 0; j < image->cols; j++){
	  sample.push_back(row_i[i]);
	}
	float std=standardDeviation(sample);
	stdVector.push_back(std);
  }
  Histogram  hist(&stdVector);
  return hist.getFeatures();
}