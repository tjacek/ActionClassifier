#include "ActionClassifier.h"

typedef vector<float> Sample;
typedef vector<float> StdVector;

extern float standardDeviation(Sample sample);

class Histogram{
  public:
    int numberOfBins;
	int * bins;  
    float max;
	float min;
	float step;

	Histogram(StdVector* stdVector,int numberOfBins=10 );
	void addNumber(float number);
	FeatureVector getFeatures();
};