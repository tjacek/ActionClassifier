#include "utils.h"

float standardDeviation(Sample data){
  double avg=0.0;
  vector<float>::iterator it ;
  double n= (data.size()-1);
  for (it = data.begin(); it != data.end(); it++){
      float x= *it;
	//  cout << x<<" ";
	  avg+=x;
  }
  avg /=n;
  	 //cout << avg <<" ";

  double sd=0.0;
  for(it = data.begin(); it != data.end(); it++){
     double error=avg - *it;
	 error=(error*error)/n;
	 sd+=error;
  }
  //cout << sd <<"\n";
  return sqrt((float) sd);
}

bool biasedCoin(){
  int n= rand() % 3;
  if(n==0)
	  return false;
  return true;
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
	//cout << number <<"\n";
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
