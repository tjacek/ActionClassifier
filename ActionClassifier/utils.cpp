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
  Mat z_0= Mat::zeros(height,width,CV_LOAD_IMAGE_GRAYSCALE) ; 
  for(int i=0;i<height;i++){
	for(int j=0;j<width;j++){
		uchar z=img->at<uchar>(i,j);
		if(z!=255){
          int z_scaled=(z* scaleFactor) + scaleFactor;
        if(0<=z_scaled && z_scaled < width){
          x_0.at<uchar>(i,z_scaled)=255; 
		  x_0.at<uchar>(i,z_scaled-1)=255;
		  x_0.at<uchar>(i,z_scaled-2)=255;
		  x_0.at<uchar>(i,z_scaled-3)=255;

		}
		  //scaleImageX(z_scaled,j,scaleFactor,&x_0);
		if(0<=z_scaled && z_scaled < height){
          y_0.at<uchar>(z_scaled,j)=255;
		  y_0.at<uchar>(z_scaled-1,j)=255;
		  y_0.at<uchar>(z_scaled-2,j)=255;
		  y_0.at<uchar>(z_scaled-3,j)=255;
		}
		  //scaleImageY(z_scaled,j,scaleFactor,&y_0);
		  z_0.at<uchar>(i,j)=255;
		}
    }
  }
  morfologicalEdge(&x_0);
  morfologicalEdge(&y_0);
  morfologicalEdge(&z_0);

  connectedCommponents(&x_0);
  connectedCommponents(&y_0);
  connectedCommponents(&z_0);

  //showImage(&z_0,"x");
  results.push_back(x_0);
  results.push_back(y_0);
  results.push_back(z_0);

  return results;
}

void showProjections(Images images){
  for(int i=0;i<images->size(); i++){
	  vector<Mat> xyz= projection(&images->at(i).image);
	for(int j=0;j<xyz.size(); j++){
		Mat mat= xyz.at(j);
		showImage(&mat,"proj");
    }
  }
}

void showCounturs(Images images){
  for(int i=0;i<images->size(); i++){
	  vector<Mat> xyz= projection(&images->at(i).image);
	for(int j=0;j<xyz.size(); j++){
		Mat mat= xyz.at(j);
		vector<vector<cv::Point> > contours;
        vector<cv::Vec4i> hierarchy;
        cv::findContours(mat, contours, hierarchy,CV_RETR_LIST , CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );
		cv::Scalar color = cv::Scalar( rand() % 255, rand() % 255, rand() % 255 );
		Mat drawing = Mat::zeros( mat.size(), CV_8UC3 );
		int largestIndex=0;
		int countSize=0;
		for( int i = 0; i< contours.size(); i++ ) {
			int n=contours.at(i).size();
			if(n<countSize){
			  largestIndex=i;
			  countSize=n;
			}
		}
		drawContours( drawing, contours, largestIndex, color, 2, 8, hierarchy, 0, cv::Point(0,0) );
		showImage(&drawing,"contours");
    }
  }
}

void saveImages(vector<Mat> images){
  cv::imwrite( "Z.jpg", images.at(0));
  cv::imwrite( "X.jpg", images.at(1));
  cv::imwrite( "Y.jpg", images.at(2));

}

void morfologicalEdge(Mat * m){
  cv::blur(*m,*m,cv::Size(11,11));
  cv::threshold(*m,*m,128,255,0);
  //connectedCommponents(m);
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

void connectedCommponents(Mat * dimage){
  int ** relation=new int*[dimage->rows];
  for(int i=0;i<dimage->rows;i++){
    relation[i]=new int[dimage->cols];
  }
  init(  relation,dimage->rows,dimage->cols);
  int maxComponentSize=0;
  int maxComponent=1;
  int currentComponent=2;
  for(int i=1;i<dimage->rows-1;i++){
	for(int j=1;j<dimage->cols-1;j++){
        if(relation[i][j]==0){
			int sizeOfComponent=markComponent(i,j,currentComponent,relation,dimage,2000);
			if(maxComponentSize<sizeOfComponent ){
				maxComponentSize=sizeOfComponent;
				maxComponent=currentComponent;
			}
			currentComponent++;
		}
	}
  }
  clean(maxComponent,relation,dimage);
}

void init(int **  table,int height,int width){
  for(int i=0;i<height;i++){
	  table[i][0]=-1;
	  table[i][width-1]=-1;
  }
  for(int j=0;j<width;j++){
	  table[0][j]=-1;
	  table[height-1][j]=-1;
  }
  for(int i=1;i<height-1;i++){
    for(int j=1;j<width-1;j++){
	  table[i][j]=0;	 
    }
  }
}

bool checkBounds(int x,int y,int ** relation,Mat * dimage){
  if(x<=0){
	return false;
  }
  
  if(y<=0){
	return false;
  }

  if(dimage->rows <=x){
	return false;
  }

  if(dimage->cols <=y){
	return false;
  }

  return true;
}

int markComponent(int x,int y,int componentNumber,int ** relation,Mat * dimage,int iter){

  if(iter<0){
	return 0;
  }
  if(checkBounds( x, y, relation,dimage)){
   if(relation[x][y]!=0){
	   return 0;
   }
   uchar value=dimage->at<uchar>(x,y);
   int numberOfPixels=0;

   if(relation[x][y]==0 && value==0){
	   relation[x][y]=1;
	   return 0;
   }
   if(relation[x][y]==0 && value!=0){
     relation[x][y]=componentNumber;
	 numberOfPixels+=markComponent(x  ,y+1,componentNumber,relation, dimage,iter-1);
	 numberOfPixels+=markComponent(x+1,y  ,componentNumber,relation, dimage,iter-1);
     numberOfPixels+=markComponent(x+1,y+1,componentNumber,relation, dimage,iter-1);
	 numberOfPixels+=markComponent(x  ,y-1,componentNumber,relation, dimage,iter-1);
	 numberOfPixels+=markComponent(x-1,y  ,componentNumber,relation, dimage,iter-1);
	 numberOfPixels+=markComponent(x-1,y-1,componentNumber,relation, dimage,iter-1);
     numberOfPixels+=markComponent(x-1,y+1,componentNumber,relation, dimage,iter-1);
	 numberOfPixels+=markComponent(x+1,y-1,componentNumber,relation, dimage,iter-1);
	 numberOfPixels+=1;
	 return numberOfPixels;
     }
   }
   return 0 ;
}

void clean(int maxComponent,int ** relation,Mat * dimage){
  for(int i=0;i<dimage->rows;i++){
	for(int j=0;j<dimage->cols;j++){
	  if(relation[i][j]!=maxComponent){
		  dimage->at<uchar>(i,j)=0;
	  }
	}
  }
}
