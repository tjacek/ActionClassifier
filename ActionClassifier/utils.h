#include "ActionClassifier.h"
#include <math.h>

typedef vector<DepthImage>* Images;
typedef void (*ImageTransform)(Mat * m);
typedef vector<float> Sample;
typedef vector<float> StdVector;

extern void applyTransform(Images images,ImageTransform fun);
extern float standardDeviation(Sample sample);
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