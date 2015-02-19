#include "fisherVector.h"

vector<float>* getFisherVector(Mat * image,int k){
  vector<Point>* cssPoints=extractCSSFeatures( image);
  //cout << "Size " << cssPoints->size() << endl;
  Mat samples(*cssPoints);
  samples=samples.reshape(1,2);
  cv::transpose(samples,samples);
  GMM gmm(k);
  gmm.learn(samples);
  return extractFisherVector(&gmm);
}

vector<float> *extractFisherVector(GMM * gmm){
  vector<float> * fisherVector=new vector<float>();
  Point point;
  for(int i=0;i<gmm->k;i++){
	  point=gmm->gamma_ex(i);
	  fisherVector->push_back(point.x);
	  fisherVector->push_back(point.y);
  }
  for(int i=0;i<gmm->k;i++){
	  point=gmm->gamma_var(i);
	  fisherVector->push_back(point.x);
	  fisherVector->push_back(point.y);
  }
  return fisherVector;
}

GMM::GMM(int k){ 
  this->k=k;
  TermCriteria termCrit(TermCriteria::COUNT+TermCriteria::EPS, EM::DEFAULT_MAX_ITERS, FLT_EPSILON);
  this->em_model= new EM(k,EM::COV_MAT_GENERIC ,  termCrit);
  means=Mat::zeros(k, 2, CV_64F);
}

void GMM::learn(Mat samples){
	x=samples;
    em_model->trainE(samples, means,noArray() , noArray(), noArray(), noArray(), probs);
    covs  = em_model->get<vector<Mat> >("covs");
	//cout << x;
}

double GMM::assigment(int cls,int x_j){
  double lambda=0.0;
  for(int t=0;t<probs.cols;t++){
	  lambda+=probs.at<double>(t,x_j);
  }
  return lambda;
}

Point GMM::gamma_ex(int cls){
  Point point(0,0);
  for(int t=0;t<probs.cols;t++){
	  double sigma_0 = covs[cls].at<double>(0,0);
	  double sigma_1 = covs[cls].at<double>(1,1);
	  double u_0= means.at<double>(cls,0);
	  double u_1= means.at<double>(cls,1);
	  //cout << x  <<"\n \n \n";
	  double x0_t= x.at<int>(t,0);
	  double x1_t= x.at<int>(t,1);
	  double lambda_i=assigment(cls,t);
	  point.x+= lambda_i * ( (x0_t -u_0) / sigma_0) ;
	  point.y+= lambda_i * ( (x1_t -u_1) / sigma_1) ;

  }	  
  return point;
}

Point GMM::gamma_var(int cls){
  Point point(0,0);
   for(int t=0;t<probs.cols;t++){
	   //this->em_model->
	  double sigma_0 = covs[cls].at< double>(0,0);
	  double sigma_1 = covs[cls].at< double>(1,1);
	  double u_0= means.at<double>(cls,0);
	  double u_1= means.at<double>(cls,1);
	  double x0_t= x.at<int>(t,0);
	  double x1_t= x.at<int>(t,1);
	  double lambda_i=assigment(cls,t);
	  point.x+= lambda_i * (((x0_t -u_0) / sigma_0) * ((x0_t -u_0) / sigma_0) -1.0) ;
	  point.y+= lambda_i * (((x1_t -u_1) / sigma_1) * ((x1_t -u_1) / sigma_1) -1.0);

  }	  
  return point;
}