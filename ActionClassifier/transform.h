#include "ActionClassifier.h"
#include "utils.h"

extern Mat cleanEdge(Mat * m);
extern void removeEdge(Mat * m);
extern double mediana(vector<double> values);
extern double getPointMediana(int i,int j,Mat* image);
extern Mat medianaFilter(Mat * image);