#pragma once

#include "ActionClassifier.h"
#include <math.h>

typedef vector<DepthImage>* Images;
typedef void (*ImageTransform)(Mat * m);
typedef vector<double> Sample;
typedef vector<float> StdVector;

extern void applyTransform(Images images,ImageTransform fun);
extern void morfologicalEdge(Mat * m);
extern void showProjections(Images images);
extern void showCounturs(Images images);
extern void showHistograms(Images images);
extern void saveImages(vector<Mat> images);
extern vector<Mat> projection(Mat * orginal);
extern void showImage(Mat * m,const char * name);

extern int markComponent(int x,int y,int componentNumber,int ** relation,Mat * dimage,int iter);
extern void clean(int maxComponent,int ** relation,Mat * dimage);
extern void init(int **  table,int height,int width);
extern void connectedCommponents(Mat * dimage);

extern double standardDeviation(Sample sample);
extern bool biasedCoin();

class Histogram{
  public:
    int numberOfBins;
	int * bins;  
    float max;
	float min;
	float step;

	Histogram(StdVector* stdVector,int numberOfBins=5 );
	void addNumber(float number);
	FeatureVector getFeatures();
};