#include "transform.h"

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
  return newImage;
}