#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <math.h>
#include <vector>
#include <windows.h>
#include <io.h>
#include <time.h>

using namespace cv;
using namespace std;


#define HORIZONTAL    1
#define VERTICAL    0

CvSVM svm;

const char strCharacters[] = { '0', '1', '2', '3', '4', '5', \
'6', '7', '8', '9' };
const int numCharacter = 10;

const int numNeurons = 40;
const int predictSize = 10;


void generateRandom(int n, int test_num, int min, int max, vector<int>*mark_samples)
{
	int range = max - min;
	int index = rand() % range + min;
	if (mark_samples->at(index) == 0)
	{
		mark_samples->at(index) = 1;
		n++;
	}

	if (n < test_num)
		generateRandom(n, test_num, min, max, mark_samples);

}


vector<string> getFiles(const string &folder,
	const bool all /* = true */) {
	vector<string> files;
	list<string> subfolders;
	subfolders.push_back(folder);

	while (!subfolders.empty()) {
		string current_folder(subfolders.back());

		if (*(current_folder.end() - 1) != '/') {
			current_folder.append("/*");
		}
		else {
			current_folder.append("*");
		}

		subfolders.pop_back();

		struct _finddata_t file_info;
		long file_handler = _findfirst(current_folder.c_str(), &file_info);

		while (file_handler != -1) {
			if (all &&
				(!strcmp(file_info.name, ".") || !strcmp(file_info.name, ".."))) {
				if (_findnext(file_handler, &file_info) != 0) break;
				continue;
			}

			if (file_info.attrib & _A_SUBDIR) {
				// it's a sub folder
				if (all) {
					// will search sub folder
					string folder(current_folder);
					folder.pop_back();
					folder.append(file_info.name);

					subfolders.push_back(folder.c_str());
				}
			}
			else {
				// it's a file
				string file_path;
				// current_folder.pop_back();
				file_path.assign(current_folder.c_str()).pop_back();
				file_path.append(file_info.name);

				files.push_back(file_path);
			}

			if (_findnext(file_handler, &file_info) != 0) break;
		}  // while
		_findclose(file_handler);
	}

	return files;
}

void AppendText(string filename, string text)
{
	fstream ftxt;
	ftxt.open(filename, ios::out | ios::app);
	if (ftxt.fail())
	{
		cout << "创建文件失败!" << endl;
		getchar();
	}
	ftxt << text << endl;
	ftxt.close();
}

// ！获取垂直和水平方向直方图
Mat ProjectedHistogram(Mat img, int t)
{
	int sz = (t) ? img.rows : img.cols;
	Mat mhist = Mat::zeros(1, sz, CV_32F);

	for (int j = 0; j<sz; j++) {
		Mat data = (t) ? img.row(j) : img.col(j);
		mhist.at<float>(j) = countNonZero(data);	//统计这一行或一列中，非零元素的个数，并保存到mhist中
	}

	//Normalize histogram
	double min, max;
	minMaxLoc(mhist, &min, &max);

	if (max>0)
		mhist.convertTo(mhist, -1, 1.0f / max, 0);//用mhist直方图中的最大值，归一化直方图

	return mhist;
}

Mat features(Mat in, int sizeData)
{
	
	Mat lowData;
	resize(in, lowData, Size(sizeData, sizeData));


	//Last 10 is the number of moments components
	int numCols =  lowData.cols*lowData.cols;
	//int numCols = vhist.cols + hhist.cols;
	Mat out = Mat::zeros(1, numCols, CV_32F);
	//Asign values to feature,ANN的样本特征为水平、垂直直方图和低分辨率图像所组成的矢量
	int j = 0;
	
	for (int x = 0; x<lowData.cols; x++)
	{
		for (int y = 0; y<lowData.rows; y++) {
			out.at<float>(j) = (float)lowData.at<unsigned char>(x, y);
			j++;
		}
	}
	//if(DEBUG)
	//	cout << out << "\n===========================================\n";
	return out;
}

Mat features2(Mat in, int sizeData)
{
	//Histogram features

	Mat vhist = ProjectedHistogram(in, VERTICAL);
	Mat hhist = ProjectedHistogram(in, HORIZONTAL);
	//Low data feature
	Mat lowData;
	resize(in, lowData, Size(sizeData, sizeData));


	//Last 10 is the number of moments components
	int numCols = vhist.cols + hhist.cols + lowData.cols*lowData.cols;
	//int numCols = vhist.cols + hhist.cols;
	Mat out = Mat::zeros(1, numCols, CV_32F);
	//Asign values to feature,ANN的样本特征为水平、垂直直方图和低分辨率图像所组成的矢量
	int j = 0;
	for (int i = 0; i<vhist.cols; i++)
	{
		out.at<float>(j) = vhist.at<float>(i);
		j++;
	}
	for (int i = 0; i<hhist.cols; i++)
	{
		out.at<float>(j) = hhist.at<float>(i);
		j++;
	}
	for (int x = 0; x<lowData.cols; x++)
	{
		for (int y = 0; y<lowData.rows; y++) {
			out.at<float>(j) = (float)lowData.at<unsigned char>(x, y);
			j++;
		}
	}
	//if(DEBUG)
	//	cout << out << "\n===========================================\n";
	return out;
}

void Svm_Train(Mat TrainData, Mat classes, int sizeData)
{
	CvTermCriteria criteria;
	criteria = cvTermCriteria(CV_TERMCRIT_EPS, 10000, FLT_EPSILON);
	CvSVMParams SVM_params;
	SVM_params = CvSVMParams(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 1, 0, 0, NULL, criteria);
	//SVM_params  = CvSVMParams(CvSVM::C_SVC, CvSVM::LINEAR, 10.0, 8.0, 1.0, 10.0, 0.5, 0.1, NULL, criteria);
	//svm.train(TrainData, classes, Mat(), Mat(), SVM_params);
	svm.train_auto(TrainData, classes, cv::Mat(),
		cv::Mat(),
		SVM_params,
		10,
		CvSVM::get_default_grid(CvSVM::C),
		CvSVM::get_default_grid(CvSVM::GAMMA),
		CvSVM::get_default_grid(CvSVM::P),
		CvSVM::get_default_grid(CvSVM::NU),
		CvSVM::get_default_grid(CvSVM::COEF),
		CvSVM::get_default_grid(CvSVM::DEGREE),
		true);
	string file_path;
	stringstream ss(stringstream::in | stringstream::out);
	ss << "train/" << sizeData;
	file_path = ss.str();

	FileStorage fsTo(file_path, cv::FileStorage::WRITE);
	svm.write(*fsTo, "svm");

}

int recog(Mat features)
{
	int result = -1;
	
	result = svm.predict(features);

	return result;
}

float SVM_test(Mat samples_set, Mat sample_labels)
{
	int correctNum = 0;
	float accurate = 0;
	for (int i = 0; i < samples_set.rows; i++)
	{
		int result = recog(samples_set.row(i));
		if (result == sample_labels.at<int>(i))
			correctNum++;
	}
	accurate = (float)correctNum / samples_set.rows;
	return accurate;
}

int saveTrainData()
{
	cout << "Begin saveTrainData" << endl;
	Mat classes;
	Mat trainingDataf5;
	Mat trainingDataf10;
	Mat trainingDataf15;
	Mat trainingDataf20;

	vector<int> trainingLabels;
	string path = "charSamples";

	for (int i = 0; i < numCharacter; i++)
	{
		cout << "Character: " << strCharacters[i] << "\n";
		stringstream ss(stringstream::in | stringstream::out);
		ss << path << "/" << strCharacters[i];

		auto files = getFiles(ss.str(), 1);

		int size = files.size();
		for (int j = 0; j < size; j++)
		{
			cout << files[j].c_str() << endl;
			Mat img = imread(files[j].c_str(), 0);
			Mat f5 =  features(img, 5);
			Mat f10 = features(img, 10);
			Mat f15 = features(img, 15);
			Mat f20 = features(img, 20);

			trainingDataf5.push_back(f5);
			trainingDataf10.push_back(f10);
			trainingDataf15.push_back(f15);
			trainingDataf20.push_back(f20);
			trainingLabels.push_back(i);			//每一幅字符图片所对应的字符类别索引下标
		}
	}



	trainingDataf5.convertTo(trainingDataf5, CV_32FC1);
	trainingDataf10.convertTo(trainingDataf10, CV_32FC1);
	trainingDataf15.convertTo(trainingDataf15, CV_32FC1);
	trainingDataf20.convertTo(trainingDataf20, CV_32FC1);
	Mat(trainingLabels).copyTo(classes);

	FileStorage fs("train/features_data.xml", FileStorage::WRITE);
	fs << "TrainingDataF5" << trainingDataf5;
	fs << "TrainingDataF10" << trainingDataf10;
	fs << "TrainingDataF15" << trainingDataf15;
	fs << "TrainingDataF20" << trainingDataf20;
	fs << "classes" << classes;
	fs.release();

	cout << "End saveTrainData" << endl;

	return 0;
}


void SVM_Cross_Train_and_Test(int Imagsize)
{

	String training;
	Mat TrainingData;
	Mat Classes;

	FileStorage fs;
	fs.open("train/features_data.xml", FileStorage::READ);


	cout << "Begin to SVM_Cross_Train_and_Test " << endl;

	char *txt = new char[50];
	sprintf(txt, "交叉训练，特征维度%d,", 40 + Imagsize * Imagsize);
	AppendText("output.txt", txt);
	cout << txt << endl;
	stringstream ss(stringstream::in | stringstream::out);
	ss << "TrainingDataF" << Imagsize;
	training = ss.str();

	fs[training] >> TrainingData;
	fs["classes"] >> Classes;
	fs.release();

	float result = 0.0;

	srand(time(NULL));

	vector<int> markSample(TrainingData.rows, 0);

	generateRandom(0, 50, 0, TrainingData.rows - 1, &markSample);

	Mat train_set, train_labels;
	Mat sample_set, sample_labels;

	for (int i = 0; i < TrainingData.rows; i++)
	{
		if (markSample[i] == 1)
		{
			sample_set.push_back(TrainingData.row(i));
			sample_labels.push_back(Classes.row(i));
		}
		else
		{
			train_set.push_back(TrainingData.row(i));
			train_labels.push_back(Classes.row(i));
		}
	}

	Svm_Train(train_set, train_labels);

	result = SVM_test(sample_set, sample_labels);


	sprintf(txt, "正确率%f\n", result);
	cout << txt << endl;
	AppendText("output.txt", txt);

	cout << "End the SVM_Cross_Train_and_Test" << endl;

	cout << endl;


}


void SVM_saveModel(int sizeData)
{
	FileStorage fs;
	fs.open("train/features_data.xml", FileStorage::READ);

	Mat TrainingData;
	Mat Classes;

	string training;
	if (1)
	{
		stringstream ss(stringstream::in | stringstream::out);
		ss << "TrainingDataF" << sizeData;
		training = ss.str();
	}

	fs[training] >> TrainingData;
	fs["classes"] >> Classes;


	cout << "Begin to saveModelChar predictSize:" << sizeData
		<< endl;


	Svm_Train(TrainingData, Classes);



	cout << "End the saveModelChar" << endl;


	string model_name = "train/svm.xml";


	FileStorage fsTo(model_name, cv::FileStorage::WRITE);
	svm.write(*fsTo, "svm");
}

void test_img(const string& filename)
{
	Mat src_img = imread(filename, 0);
	Mat feat = features(src_img, 10);
	int ret = svm.predict(feat);
	cout << ret << endl;
}
int main()
{
	
	int DigitSize[4] = { 5, 10, 15, 20};
	cout << "To be begin." << endl;

	saveTrainData();

	for (int i = 0; i < 4; i++)
	{
		SVM_Cross_Train_and_Test(DigitSize[i]);
	}

	cout << "To be end." << endl;
	
	//SVM_Cross_Train_and_Test(10);
	/*SVM_saveModel(10);*/
	/*svm.load("train/svm_auto_rbf.xml");
	test_img("....");*/


	return 0;
}