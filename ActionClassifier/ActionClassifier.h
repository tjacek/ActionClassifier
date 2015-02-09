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

static const int heightOfImage=320;
static const int widthOfImage=640;

typedef vector<float> * FeatureVector;
typedef cv::Mat * DepthImage;
typedef vector<string>* ImageList;

class FeatureExtractor{
  public:
    virtual FeatureVector getFeatures(DepthImage image)=0;
};

class Dataset{
  public:
    void addExample(DepthImage image);
	void registerExtractor(FeatureExtractor* extractor);
  private:
    vector<FeatureExtractor*> extractors;
	vector<FeatureVector> examples;
};

typedef void  (*AddExtractorsFunc)(Dataset * data);

extern Dataset * buildDataset(ImageList imageList, AddExtractorsFunc addExtractors);

class Classifier{
};