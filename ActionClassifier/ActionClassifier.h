#pragma once

#include <opencv\cv.h>
#include <opencv\highgui.h>

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <windows.h>
#include <tchar.h> 
#include <strsafe.h>

//using namespace cv;
using namespace std;

typedef vector<float> * FeatureVector;
typedef cv::Mat * DepthImage;
typedef void  (*AddExtractorsFunc)(Dataset * data) ;

class Dataset{
  public:
    void addExample(DepthImage image);
  private:
    vector<FeatureExtractor*> extractors;
	vector<FeatureVector> examples;
};

class FeatureExtractor{
  public:
    virtual FeatureVector getFeatures(DepthImage image)=0;
};

class Classifier{
};