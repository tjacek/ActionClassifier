#include "css.h"

 vector<Point>* extractCSSFeatures(Mat * image){
   vector<Point>* points=new vector<Point>();
   detectEdge(image);
   Curve* curve=getCurve(image);
  // curve->smooth(2.0);
   return points;
 }

static const double minThreshold = 30.0;
static const double maxThreshold = 50.0;

void detectEdge(Mat * mat){
  cv:blur(*mat,*mat,cv::Size(20,20));
  cv::threshold(*mat,*mat,128,255,0);
  cv::Canny(*mat, *mat, minThreshold, maxThreshold); 
}

Curve * getCurve(Mat * image){
  vector<vector<cv::Point> > contours;
  vector<cv::Vec4i> hierarchy;

  	 cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
	 cv::imshow( "Display window", *image );                  
     cv::waitKey(0);

   cv::findContours(*image, contours, hierarchy,CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
   draw( *image, &contours, &hierarchy);
  //vector<cv::Point> countur=contours.at(0);
  //Curve * curve = new Curve(countur);
  return NULL;//curve;
}

Curve::Curve(vector<cv::Point> countur){
  vector<double> x_vec;
  vector<double> y_vec;
  vector<Point>::iterator it;
  int i=0;
  for(it=countur.begin(); it!=countur.end(); ++it )
  {
     Point point=*it;
	 x_vec.push_back(point.x);
	 y_vec.push_back(point.y);
  }
  x=new Mat(x_vec);
  y=new Mat(y_vec);
}

void Curve::smooth(double sigma){
  smooth_x=new Mat();
  smooth_y=new Mat();
  cout << *x;
  gauss_conv(*x,*smooth_x, sigma);
  gauss_conv(*y,*smooth_y, sigma);
}

void gauss_conv(Mat in,Mat out,double sigma){
  int width=in.rows;
  Mat G;
  cv::transpose(cv::getGaussianKernel(width, sigma, CV_64FC1), G);
 // cout << G;
//  filter2D(in, out, out.depth(), G);
}

void draw(Mat image,vector<vector<cv::Point> > *contours,vector<cv::Vec4i> *hierarchy){
  Mat drawing = Mat::zeros( image.size(), CV_8UC3 );
  //for( int i = 0; i< contours.size(); i++ )
  //   {
       cv::Scalar color = cv::Scalar( rand() % 255, rand() % 255, rand() % 255 );
       drawContours( drawing, *contours, -1, color, 2, 8, *hierarchy, 0, Point() );
  //   }

  cv::namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
  imshow( "Contours", drawing );
  cv::waitKey(0);

}