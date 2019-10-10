#include "Tracker.h"

double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt)		//计算两个框的交并比
{
	float in = (bb_test & bb_gt).area();
	float un = bb_test.area() + bb_gt.area() - in;

	if (un < DBL_EPSILON)
		return 0;
   
   return (double)(in / un);
}

void Tracker::Initial(vector<KalmanTracker> &trackers, vector<TrackingBox> &tempVec)
{
  for (unsigned int i = 0; i < tempVec.size(); i++)                         //第几帧，该帧有多少行
  {
    KalmanTracker trk = KalmanTracker(tempVec[i].box, tempVec[i].classid);  //该帧的第i行，其实这里就是用第1帧的所有行进行初始化
    trackers.push_back(trk);                                                //初始化后的结果放进一个vector中
  }
}

void Tracker::predict(vector<KalmanTracker>& trackers, vector<Rect_<float>>& predictedBoxes)
{
	//3.1. get predicted locations from existing trackers.		通过现有的追踪器来得到预测位置
	predictedBoxes.clear();
  for (auto it = trackers.begin(); it != trackers.end();)
	{
		Rect_<float> pBox = (*it).predict();
    if (pBox.x >= 0 && pBox.y >= 0)
		{
			predictedBoxes.push_back(pBox);
			it++;
		}
		else
		{
			it = trackers.erase(it);
		}
	}
}

void Tracker::getMatchedPair(vector<KalmanTracker>& trackers, vector<TrackingBox>& tempVec, vector<Rect_<float>& predictedBoxes)
{
  ///////////////////////////////////////
	// 3.2. associate detections to tracked object (both represented as bounding boxes)		将检测关联到追踪对象（均表现为标定框）
	// dets : detFrameData[fi]
	trkNum = predictedBoxes.size();		//预测框的尺寸，即有多少个框
	detNum = tempVec.size();			//一帧有多少行

	iouMatrix.clear();
  iouMatrix.resize(trkNum, vector<double>(detNum, 0));

	for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix		计算iou矩阵作为距离矩阵
	{
		for (unsigned int j = 0; j < detNum; j++)
		{
			// use 1-iou because the hungarian algorithm computes a minimum-cost assignment.	使用1-iou因为匈牙利算法计算最小成本分配
			iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], tempVec[j].box);					//每一个预测框和该帧中所有行计算
		}
	}

	// solve the assignment problem using hungarian algorithm.		使用匈牙利算法来解决分配问题
  // the resulting assignment is [track(prediction) : detection], with len=preNum		分配结果是[track(prediction) : detection]，长度等于preNum
	HungarianAlgorithm HungAlgo;
	assignment.clear();
  HungAlgo.Solve(iouMatrix, assignment);
  
  unmatchedTrajectories.clear();
	unmatchedDetections.clear();
	allItems.clear();
	matchedItems.clear();

	if (detNum > trkNum) //	there are unmatched detections		如果一帧的行数大于预测框的数量，即未匹配的检测
	{
    for (unsigned int n = 0; n < detNum; n++)
			allItems.insert(n);

		for (unsigned int i = 0; i < trkNum; ++i)
			matchedItems.insert(assignment[i]);

		set_difference(allItems.begin(), allItems.end(),
			matchedItems.begin(), matchedItems.end(),
			insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
	}
	else if (detNum < trkNum) // there are unmatched trajectory/predictions		一帧的行数小于预测框的数量，即有未匹配的追踪/预测
	{
		for (unsigned int i = 0; i < trkNum; ++i)
			if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm		未分配的标签将在分配算法中设置为-1
				unmatchedTrajectories.insert(i);
	}

	// filter out matched with low IOU		过滤掉低IOU的匹配
  matchedPairs.clear();
	for (unsigned int i = 0; i < trkNum; ++i)
	{
		if (assignment[i] == -1) //pass over invalid values
			continue;
		if (1 - iouMatrix[i][assignment[i]] < iouThreshold)
		{
      unmatchedTrajectories.insert(i);
			unmatchedDetections.insert(assignment[i]);
		}
		else
		{
			matchedPairs.push_back(cv::Point(i, assignment[i]));
		}
	}
}

void Tracker::getResult(vector<cv::Point>& matchedPairs, vector<KalmanTracker>& trackers, vector<TrackingBox>& tempVec)
{
	///////////////////////////////////////
	// 3.3. updating trackers

	// update matched trackers with assigned detections.
	// each prediction is corresponding to a tracker
	int detIdx, trkIdx;
	for (unsigned int i = 0; i < matchedPairs.size(); i++)
	{
		trkIdx = matchedPairs[i].x;
		detIdx = matchedPairs[i].y;
		trackers[trkIdx].update(tempVec[detIdx].box, tempVec[detIdx].classId);
	}

	// create and initialise new trackers for unmatched detections
	for (auto umd : unmatchedDetections)
	{
		KalmanTracker tracker = KalmanTracker(tempVec[umd].box, tempVec[umd].classId);
		trackers.push_back(tracker);
	}

	// get trackers' output
	frameTrackingResult.clear();
	for (auto it = trackers.begin(); it != trackers.end();)
	{
		if (((*it).m_time_since_update < 1) &&
			((*it).m_hit_streak >= min_hits))
		{
			TrackingBox res;
			res.box = (*it).get_state();
			res.id = (*it).m_id + 1;
			res.classId = (*it).classId;
			//res.frame = frame_count;
			frameTrackingResult.push_back(res);
			it++;
		}
		else
		{
			it++;
		}
			
		// remove dead tracklet
		if (it != trackers.end() && (*it).m_time_since_update > max_age)
		{
			it = trackers.erase(it);
		}			
	}

}


set<int> Tracker::unmatchedDetectionsResult()
{
	return unmatchedDetections;
}

vector<cv::Point> Tracker::MatchedPairResult()
{
	return matchedPairs;
}

vector<TrackingBox> Tracker::getTrackedObject()
{
	return frameTrackingResult;
}

vector<KalmanTracker> Tracker::getBackTrackers()
{
	return trackersResult;
}


void Tracker::Tracking(vector<TrackingBox>& tempVec)
{
	if (trackers.size() == 0) // the first frame met	第一帧
	{
		Initial(trackers, tempVec);
		
		//continue;
	}
	else
	{
		//得到预测框
		predict(trackers, predictedBoxes);
		//得到匹配的和未匹配的
		getMatchedPair(trackers, tempVec, predictedBoxes);
    matchedPairs = MatchedPairResult();
		unmatchedDetections = unmatchedDetectionsResult();
		////得到追踪结果
		getResult(matchedPairs, trackers, tempVec);
	}
	
}






