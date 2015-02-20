#include "utils.h"

void morfologicalEdge(Mat * m){
  int morph_elem = 0;
  int morph_size = 1;
  int const max_elem = 2;
  Mat element = getStructuringElement( morph_elem, cv::Size( 2*morph_size + 1, 2*morph_size+1 ), cv::Point( morph_size, morph_size ) );
  //cv::blur(*m,*m,cv::Size(21,21));
   cv::threshold(*m,*m,128,255,0);
 // Mat  dst;
  cv::morphologyEx( *m, *m, cv::MORPH_GRADIENT , element );
  const char * window_name= "Morphology";
  showImage(m,window_name);
 
}

void showImage(Mat * m,const char * name){
  cv::namedWindow( name, CV_WINDOW_AUTOSIZE );
  cv::imshow( name, *m);
  cv::waitKey(0);
}

void applyTransform(Images images,ImageTransform fun){
  vector<DepthImage>::iterator it;
  int i=0;
  for(it=images->begin(); it!=images->end(); ++it )
  {
     DepthImage image=*it;
	 fun(&image.image);
  }
}

float standardDeviation(Sample data){
  double avg=0.0;
  vector<float>::iterator it ;
  double n= (data.size()-1);
  for (it = data.begin(); it != data.end(); it++){
      float x= *it;
	  avg+=x;
  }
  avg /=n;

  double sd=0.0;
  for(it = data.begin(); it != data.end(); it++){
     double error=avg - *it;
	 error=(error*error)/n;
	 sd+=error;
  }
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
