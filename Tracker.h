#pragma once

#include "Hungarian.h"
#include "KalmanTracker.h"

#include <set>

#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

//using namespace std;

typedef struct TrackingBox					//定义一个结构体，来存储帧序号，车id，车框，其中车框是rect类型的，默认车id为-1
{
	int frame;
	long long id = -1;
	int classId = -1;
	Rect_<float> box;
}TrackingBox;

// Computes IOU between two boundboxes   计算两个标定框之间的IOU值
double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt);

class Tracker
{
public:
	void Initial(vector<KalmanTracker> &trackers, vector<TrackingBox> &tempVec);

	void predict(vector<KalmanTracker>& trackers, vector<Rect_<float>>& predictedBoxes);

	void getMatchedPair(vector<KalmanTracker>& trackers, vector<TrackingBox>& tempVec, vector<Rect_<float>>& predictedBoxes);
  
  void getResult(vector<cv::Point>& matchedPairs, vector<KalmanTracker>& trackers, vector<TrackingBox>& tempVec);
	
	void Tracking(vector<TrackingBox>& tempVec);
	
	vector<cv::Point> MatchedPairResult();

	set<int> unmatchedDetectionsResult();

	vector<TrackingBox> getTrackedObject();
	
	vector<KalmanTracker> getBackTrackers();

private:

	vector<KalmanTracker> trackers;

	int max_age = 20;
	int min_hits = 3;
	double iouThreshold = 0.3;

	vector<Rect_<float>> predictedBoxes;

	vector<vector<double>> iouMatrix;
	vector<int> assignment;
	set<int> unmatchedDetections;
	set<int> unmatchedTrajectories;
	set<int> allItems;
	set<int> matchedItems;
	vector<cv::Point> matchedPairs;
  vector<TrackingBox> frameTrackingResult;
	vector<KalmanTracker> trackersResult;
	unsigned int trkNum = 0;
	unsigned int detNum = 0;


	int total_frames = 0;
	double total_time = 0.0;
};
