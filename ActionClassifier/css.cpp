#include "css.h"

 vector<Point>* extractCSSFeatures(Mat * image){
   detectEdge(image);
   Curve* curve=getCurve(image);
   curve->smooth(2.0);
   curve->computeDervatives();
   return curve->crossingPoints();
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
  /*	 cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
	 cv::imshow( "Display window", *image );                  
     cv::waitKey(0);*/
  cv::findContours(*image, contours, hierarchy,CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
  //draw( *image, &contours, &hierarchy);
  vector<cv::Point> countur=contours.at(0);
  Curve * curve = new Curve(countur);
  return curve;
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
  //cout << *x;
  gauss_conv(*x,*smooth_x, sigma);
  G=gauss_conv(*y,*smooth_y, sigma);
}

void Curve::computeDervatives(){
  cv::Sobel(G, dG, -1, 1, 0, 3);
  cv::Sobel(G, ddG, -1, 2, 0, 3);
  cv::flip(dG, dG, 0);
  cv::flip(ddG, ddG, 0);
  /*int xa=dG.cols; 
  xa-= 0 -1;
  int ya= dG.rows;
  ya-= 1;*/
  Point anchor(-1,-1);

  filter2D(*x, dX,  -1, dG, anchor);
  filter2D(*y, dY,  -1, dG, anchor);
  filter2D(*x, ddX, -1, ddG, anchor);
  filter2D(*y, ddY, -1, ddG, anchor);
}

bool sign(double x){
  if(x<0){
    return true;
  }
  return false;
}

vector<Point>*  Curve::crossingPoints(){
  vector<Point>* crossingPoints= new vector<Point>();
  int size=this->x->rows;
  double last=curvatureAt(0);
  for(int i=1;i<size;i++){
    double current=curvatureAt(i);
	bool signChange=(sign(current)==sign(last) );
	if(current==0 || signChange){
        Point point;
		point.x=x->at<double>(i,0);
		point.y=y->at<double>(i,0);
		crossingPoints->push_back(point);
	}
	last=current;
  }
  return crossingPoints;
}

double Curve::curvatureAt(int i){
  double dx_i =dX.at<double>(i,0);
  double dy_i =dY.at<double>(i,0);
  double ddx_i=ddX.at<double>(i,0);
  double ddy_i=ddY.at<double>(i,0);
  return curvature( dx_i, dy_i, ddx_i, ddy_i);
}

double curvature(double dx,double dy,double ddx,double ddy){
  double div=pow ( dx*dx + dy*dy, 1.5);
  return (dx*ddy - ddx*dy)/div;
}

Mat gauss_conv(Mat in,Mat out,double sigma){
  int width=in.rows;
  Mat G;
  cv::transpose(cv::getGaussianKernel(width, sigma, CV_64FC1), G);
 // cout << G;
  filter2D(in, out, out.depth(), G);
  return G;
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