#pragma once

#include <opencv\cv.h>
#include <opencv\ml.h>
#include <opencv\highgui.h>

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <math.h>
#include <random>

#include <windows.h>
#include <tchar.h> 
#include <strsafe.h>

using namespace std;
using cv::Mat;

typedef vector<double>* FeatureVector;
typedef cv::Mat ImageDescriptor;
typedef vector<string>* ImageList;
typedef cv::Mat * Labels;

class DepthImage{
  public:
    string name;
	Mat image;
    DepthImage(string imageName);
};

class FeatureExtractor{
  public:
	virtual int numberOfFeatures()=0;
	virtual string featureName(int i)=0;
    virtual FeatureVector getFeatures(DepthImage image)=0;
};

class Dataset{
  public:
    vector<FeatureExtractor*> extractors;
	Dataset();
    void addExample(DepthImage image);
	void registerExtractor(FeatureExtractor* extractor);
	int numberOfFeatures();
	Mat * toMat();
	void dimReduction(int k);
	string toString();
	string toArff(Labels labels);
  private:
	//vector<double> *examples;
	vector<vector<double>> * desc;
	string getAttributes();
	string getData(Labels labels);
};

typedef void  (*AddExtractorsFunc)(Dataset * data);

class Classifier{
  public:
    virtual void learn(Labels labels,Dataset * dataset)=0;
    virtual float predict(DepthImage img)=0;
};

extern Dataset * buildDataset(ImageList imageList, AddExtractorsFunc addExtractors);
extern vector<double> getImageDescriptor(DepthImage image,vector<FeatureExtractor*> extractors);
extern void addAllExtractors(Dataset * dataset);
extern void showImages(vector<DepthImage>* images);
extern vector<DepthImage> * readImages(ImageList images);