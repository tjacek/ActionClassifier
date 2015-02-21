#include "utils.h"

void scaleImageX(int x,int y,int scaleFactor,Mat * mat){
 for(int i=0;i<scaleFactor+1;i++){
    int x_i=x-i;
	if(x_i<0) return;
	mat->at<uchar>(x_i,y)=255;
 }
}

void scaleImageY(int x,int y,int scaleFactor,Mat * mat){
 for(int i=0;i<scaleFactor;i++){
   int y_i=y-i;
   if(y_i<0) return;
   mat->at<uchar>(x,y_i)=255;
 }
}

vector<Mat> projection(Mat * img){
  int scaleFactor=4; 
  vector<Mat> results;
  int height =img->rows;
  int width = img->cols;
  Mat x_0= Mat::zeros(height,width,CV_LOAD_IMAGE_GRAYSCALE); 
  Mat y_0= Mat::zeros(height,width,CV_LOAD_IMAGE_GRAYSCALE); 
  Mat z_0= Mat::zeros(height,width,CV_LOAD_IMAGE_GRAYSCALE); 
  for(int i=0;i<height;i++){
	for(int j=0;j<width;j++){
		uchar z=img->at<uchar>(i,j);
		if(z!=255){
          int z_scaled=z* scaleFactor;
        if(z_scaled < height){
          x_0.at<uchar>(i,z_scaled)=255; 
		  x_0.at<uchar>(i,z_scaled-1)=255;
		  x_0.at<uchar>(i,z_scaled-2)=255;
		  x_0.at<uchar>(i,z_scaled-3)=255;

		}
		  //scaleImageX(z_scaled,j,scaleFactor,&x_0);
		if(z_scaled < width){
          y_0.at<uchar>(j,z_scaled)=255;
		  y_0.at<uchar>(j,z_scaled-1)=255;
		  y_0.at<uchar>(j,z_scaled-2)=255;
		  y_0.at<uchar>(j,z_scaled-3)=255;
		}
		  //scaleImageY(z_scaled,j,scaleFactor,&y_0);
		  z_0.at<uchar>(i,j)=255;
		}
    }
  }
  morfologicalEdge(&x_0);
  morfologicalEdge(&y_0);
  morfologicalEdge(&z_0);

  //showImage(&z_0,"x");
  results.push_back(x_0);
  results.push_back(y_0);
  results.push_back(z_0);

  return results;
}

void saveImages(vector<Mat> images){
  cv::imwrite( "Z.jpg", images.at(0));
  cv::imwrite( "X.jpg", images.at(1));
  cv::imwrite( "Y.jpg", images.at(2));

}

void morfologicalEdge(Mat * m){
  cv::blur(*m,*m,cv::Size(11,11));
  cv::threshold(*m,*m,128,255,0);
  int morph_elem = 0;
  int morph_size = 1;
  int const max_elem = 2;
  Mat element = cv::getStructuringElement( cv::MORPH_RECT,
                                       cv::Size( 2*morph_size + 1, 2*morph_size+1 ),
                                       cv::Point( morph_size, morph_size ) );

  morphologyEx( *m, *m, cv::MORPH_GRADIENT, element );
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
  FeatureVector features=new vector<double>();
  for(int i=0;i<numberOfBins;i++){
	features->push_back((float) bins[i]);
  }
  return features;
}
