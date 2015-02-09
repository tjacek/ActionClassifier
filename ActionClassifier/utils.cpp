#include "utils.h"

float standardDeviation(Sample data){
  float avg=0.0;
  vector<float>::iterator it ;
  for (it = data.begin(); it != data.end(); it++){
    avg+=*it;
  }
  float sd=0.0;
  for(it = data.begin(); it != data.end(); it++){
     float error=avg - *it;
	 error=error*error;
	 sd+=error;
  }
  return sqrt(sd);
}

Histogram::Histogram(StdVector* stdVector,int numberOfBins){
    this->numberOfBins=numberOfBins;
	this->bins=new int[numberOfBins];
	for(int i=0;i<numberOfBins;i++){
		bins[i]=0;
	}
	max = *min_element(stdVector->begin(), stdVector->end());
    min = *max_element(stdVector->begin(), stdVector->end());
	step= (max-min)/ ((float) numberOfBins);

	vector<float>::iterator it ;
    for (it = stdVector->begin(); it != stdVector->end(); it++){
      addNumber(*it);
    }
}

void Histogram::addNumber(float number){
   	for(int i=0;i<numberOfBins;i++){
	  float threshold=min+ (1.0+i)*step;
	  if(number<threshold){
		bins[i]+=1;
	  }
	}
}

FeatureVector Histogram::getFeatures(){
  FeatureVector features=new vector<float>();
  for(int i=0;i<numberOfBins;i++){
	features->push_back((float) bins[i]);
  }
  return features;
}
