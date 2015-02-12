#pragma once

#include <opencv\cv.h>
#include <opencv\ml.h>
#include <opencv\highgui.h>

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <windows.h>
#include <tchar.h> 
#include <strsafe.h>

using namespace std;
using cv::Mat;

typedef vector<float>* FeatureVector;
typedef cv::Mat ImageDescriptor;
typedef cv::Mat * DepthImage;
typedef vector<string>* ImageList;
typedef cv::Mat * Labels;

class FeatureExtractor{
  public:
    virtual FeatureVector getFeatures(DepthImage image)=0;
};

class Dataset{
  public:
    vector<FeatureExtractor*> extractors;

    void addExample(DepthImage image);
	void registerExtractor(FeatureExtractor* extractor);
	Mat toMat();
	string toString();
  private:
	vector<ImageDescriptor> examples;
};

typedef void  (*AddExtractorsFunc)(Dataset * data);

class Classifier{
  public:
    virtual void learn(Labels labels,Dataset * dataset)=0;
    virtual float predict(DepthImage img)=0;
};

extern Dataset * buildDataset(ImageList imageList, AddExtractorsFunc addExtractors);
extern ImageDescriptor getImageDescriptor(DepthImage image,vector<FeatureExtractor*> extractors);
