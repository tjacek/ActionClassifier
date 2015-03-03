#include "transform.h"

Mat cleanEdge(Mat * m){
	//showImage(m,"OK");
	removeEdge( m);
	//showImage(m,"OK2");

  return medianaFilter(m);
}

void removeEdge(Mat * m){
  Mat newImage(m->rows,m->cols,CV_LOAD_IMAGE_GRAYSCALE);
 // cv::blur(*m,newImage,cv::Size(11,11));
  cv::threshold(*m,newImage,128,255,0);
  int morph_elem = 0;
  int morph_size = 6;
  int const max_elem = 2;
  Mat element = cv::getStructuringElement( cv::MORPH_RECT,
                                       cv::Size( 2*morph_size + 1, 2*morph_size+1 ),
                                       cv::Point( morph_size, morph_size ) );

  morphologyEx( newImage, newImage, cv::MORPH_GRADIENT, element );
  for(int i=0;i<m->rows;i++){
    for(int j=0;j<m->cols;j++){
      if(newImage.at<uchar>(i,j)!=0){
         m->at<uchar>(i,j)=255;
	  }
    }
  }

}


double mediana(vector<double> values){
  std::sort (values.begin(), values.end());
  int size=values.size();
  if((size % 2) ==0){
	       double a=values.at(size / 2);
	 double b=values.at( (size / 2)-1);
	 return (a+b)/2;
  }else{
    return values.at( ((size-1) / 2)+1);
  }
}

double getPointMediana(int i,int j,Mat* image){
  vector<double> points;
  points.push_back(image->at<uchar>(i,j));
  
  points.push_back(image->at<uchar>(i  ,j-1));
  points.push_back(image->at<uchar>(i+1,j-1));
  points.push_back(image->at<uchar>(i+1,j));
  points.push_back(image->at<uchar>(i+1,j+1));
  points.push_back(image->at<uchar>(i  ,j+1));
  points.push_back(image->at<uchar>(i-1,j+1));
  points.push_back(image->at<uchar>(i-1,j));
  points.push_back(image->at<uchar>(i-1,j-1));

  return mediana(points);
}

Mat medianaFilter(Mat * image){

  Mat newImage(image->rows,image->cols,CV_LOAD_IMAGE_GRAYSCALE);
  for(int i=1;i<image->rows-1;i++){
    for(int j=1;j<image->cols-1;j++){
      if(newImage.at<uchar>(i,j)!=255){
        uchar value= (uchar)getPointMediana( i, j,image);
	    newImage.at<uchar>(i,j)=value;
	  }else{
		newImage.at<uchar>(i,j)=255;
	  }
    }
  }
  for(int i=0;i<image->rows;i++){
	  newImage.at<uchar>(i,0)=255;
	  newImage.at<uchar>(i,image->cols-1)=255;

  }
  for(int j=0;j<image->cols;j++){
	  newImage.at<uchar>(0,j)=255;
	  newImage.at<uchar>(image->rows-1,j)=255;
  }
  return newImage;
}