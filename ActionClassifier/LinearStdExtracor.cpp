#include "features.h"
#include "utils.h"

void addLinearStdExtractor(Dataset * dataset){
  LinearStdExtractor* extractor=new LinearStdExtractor();
  dataset->registerExtractor(extractor);
}

int LinearStdExtractor::numberOfFeatures(){
  return 5;
}

string LinearStdExtractor:: featureName(int i){
  string str="bin_" + intToString(i);
  return str;
}

FeatureVector LinearStdExtractor::getFeatures(DepthImage image){
  StdVector stdVector;  
  Mat * m_image=&image.image;

  for(int i=0;i<m_image->rows;i++){
    const uchar* row_i = m_image->ptr<uchar>(i);
	Sample sample;
	for(int j = 0; j < m_image->cols; j++){
	  sample.push_back((float) row_i[j]);
	}
	float std=standardDeviation(sample);
	stdVector.push_back(std);
  }
  Histogram  hist(&stdVector);
  return hist.getFeatures();
}
