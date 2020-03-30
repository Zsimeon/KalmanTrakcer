#include <iostream>
#include <fstream>
#include <iomanip> // to format image names using setw() and setfill()
#include <io.h>    // to check file existence using POSIX function access(). On Linux include <unistd.h>.
#include <set>

#include "KalmanTracker.h"
#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "readtxt.h"
#include "Tracker.h"

// global variables for counting	用来计数的全局变量
#define CNUM 20

int main()
{
  //为了显示，随机初始化一些颜色
	RNG rng(0xFFFFFFFF);
	Scalar_<int> randColor[CNUM];
	for (int i = 0; i < CNUM; i++)
		rng.fill(randColor[i], RNG::UNIFORM, 0, 256);

	//// 3. update across frames		通过帧来更新

	// variables used in the for-loop		for循环中的变量
  vector<Rect_<float>> predictedBoxes;
	set<int> unmatchedDetections;

	unsigned int trkNum = 0;
	unsigned int detNum = 0;

	double cycle_time = 0.0;
	int64 start_time = 0;
  
 ////////////////////////////////////////
 std::string sequence = "F:/MOT/tail/tracker";    //读取txt文件，每个文件包含一帧所有的检测框
 std::string video_path = "F:/MOT/tail/dj.mp4";   //读取视频
 
 ///读取文件夹中的boundingbox信息
	std::vector<cv::String> txt_files;
	//std::vector<cv::String> txt_files;

  std::vector<std::vector<cv::Rect_<float> > > groundtruth_rect;		//用来存储所有框的信息，最外围是第几帧，内围是每一帧有多少行
	std::vector<std::vector<cv::Point3f>> carTypes;						//用来存储车辆类型及中心点，维度同上
	vector<TrackingBox> TrackingResult;									//最终的追踪结果
	vector<cv::Point> matchedPairs;

	cv::glob(sequence, txt_files);
  
  	for (size_t i = 0; i < txt_files.size(); ++i)
	{
		//std::cout << txt_files[i] << std::endl;					//检测是否读取全部txt文件，打印文件名
		groundtruth_rect.push_back(getgroundtruth(txt_files[i]));		//读取所有文件中的全部框信息
    carTypes.push_back(getCarType(txt_files[i]));					//读取每个框的车辆类型信息并计算中心点
	}

	//将读取的值存进TrackingBox的类当中，但是现在是所有帧，需要改写为一次给一帧数据
	vector<vector<TrackingBox>> detFrameData;
	vector<TrackingBox> tempVec;
  for (size_t i = 0; i < groundtruth_rect.size(); i++)
	{//第i帧
    TrackingBox tb;
		for (size_t j = 0; j < groundtruth_rect[i].size(); j++)
		{//第i帧的第j行
			tb.classId = carTypes[i][j].x;
			tb.box = Rect_<float>(Point_<float>(groundtruth_rect[i][j].x, groundtruth_rect[i][j].y), Point_<float>(groundtruth_rect[i][j].width, groundtruth_rect[i][j].height));
      tempVec.push_back(tb);		//把相同帧序号的tb放进一个数组里
		}
		detFrameData.push_back(tempVec);
		tempVec.clear();
   }
   
   //文件在上面已经读取完了，不需要再读取文件了，直接对已有的数组的数组里面的数据进行处理
   
   ///下面开始读取视频
	cv::VideoCapture cap(video_path);
	//if (!cap.isOpened())return -1;
	cv::Mat frame, outFrame;

	Tracker Tt;

	for (int fi = 0; fi < detFrameData.size(); fi++)
  {
    Tt.Tracking(detFrameData[fi]);
		TrackingResult = Tt.getTrackedObject();
    //////////////////////////////////////////////
		//追踪结束，下面是绘图显示
    cap >> frame;
		if (frame.empty())
			break;
    cv::resize(frame, outFrame, cv::Size(416, 416), (0, 0), (0, 0), cv::INTER_LINEAR);
    
    for (auto tb : TrackingResult)
		{
      std::stringstream st;
			st << tb.id;
			std::string str = st.str();
			cv::rectangle(outFrame, cv::Rect(tb.box.x, tb.box.y, tb.box.width, tb.box.height), randColor[tb.id % 20], 1, 1, 0);
			cv::putText(outFrame, str, cv::Point(tb.box.x, tb.box.y + tb.box.height), cv::FONT_HERSHEY_PLAIN, 2.0, randColor[tb.id % 20], 2);
			st.clear();
			str.clear();
			std::stringstream st2;
			st2 << tb.classId;
			std::string str2 = st2.str();
			cv::putText(outFrame, str2, cv::Point(tb.box.x, tb.box.y), cv::FONT_HERSHEY_PLAIN, 2.0, randColor[tb.id % 20], 2);
			st2.clear();
			str2.clear();
    }
    
    imshow("Video", outFrame);
		//TrackingResult.clear();
		//等待50ms，如果从键盘输入的是q、Q、或者是Esc键，则退出
		int key = cv::waitKey(10);
		if (key == 'q' || key == 'Q' || key == 27)
			break;
    
  }
  
  return 0;
  
}
