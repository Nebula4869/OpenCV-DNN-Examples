#include <stdio.h>
#include <fstream>
#include <vector>
#include <time.h>
#include <opencv2/opencv.hpp>

#define INWIDTH 416
#define INHEIGHT 416


void runYOLOv3(int cameraID, char* cfgFile, char* darknetModel, int frameWidth, int frameHeight, float scoreThreshold, float nmsThreshold) {
	printf("------Running YOLOv3!------\n");

	// Initialize network
	cv::dnn::Net net = cv::dnn::readNetFromDarknet(cfgFile, darknetModel);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);

	// Get output node
	std::vector<cv::String> outNames;
	outNames.push_back("yolo_82");
	outNames.push_back("yolo_94");
	outNames.push_back("yolo_106");

	// Load COCO names
	std::vector<std::string> classnames;
	std::ifstream f("coco.names");
	std::string name = "";
	while (std::getline(f, name)) {
		classnames.push_back(name);
	}

	// Initialize VideoCapture
	cv::VideoCapture cap(cameraID);
	cap.set(cv::CAP_PROP_FRAME_WIDTH, frameWidth);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, frameHeight);

	
	cv::Mat frame;
	std::vector<cv::Mat> outs;
	std::vector<int> classIDs;
	std::vector<float> scores;
	std::vector<cv::Rect> bboxes;
	std::vector<int> indices;
	while (cap.isOpened()) {
		clock_t start = clock();
		outs.clear();
		classIDs.clear();
		scores.clear();
		bboxes.clear();
		indices.clear();

		// Forward inference
		cap.read(frame);
		net.setInput(cv::dnn::blobFromImage(frame, 1 / 255.0, cv::Size(INWIDTH, INHEIGHT), cv::Scalar(0, 0, 0), true, false));
		net.forward(outs, outNames);

		// Decode result
		for (int i = 0; i < outs.size(); i++) {
			float* data = (float*)outs[i].data;
			double score;
			for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
				cv::Mat classConfs = outs[i].row(j).colRange(5, outs[i].cols);
				cv::Point classIDPoint;
				
				cv::minMaxLoc(classConfs, 0, &score, 0, &classIDPoint);

				if (score > scoreThreshold) {
					classIDs.push_back(classIDPoint.x);
					scores.push_back((float)score);
					bboxes.push_back(cv::Rect(
						(int)(data[0] * frameWidth - (int)(data[2] * frameWidth) / 2),
						(int)(data[1] * frameHeight - (int)(data[3] * frameHeight) / 2),
						(int)(data[2] * frameWidth),
						(int)(data[3] * frameHeight)));
				}
			}
		}

		// Non maximum suppression
		cv::dnn::NMSBoxes(bboxes, scores, scoreThreshold, nmsThreshold, indices);

		// Visualize result
		for (int i = 0; i < indices.size(); i++) {
			float conf = scores[indices[i]];
			int classID = classIDs[indices[i]];
			cv::Rect box = bboxes[indices[i]];
			float left = box.x;
			float top = box.y;
			float right = box.x + box.width;
			float bottom = box.y + box.height;

			cv::rectangle(frame, cv::Rect(left, top, (right - left), (bottom - top)), cv::Scalar(0, 255, 0), 2);
			cv::putText(
				frame,
				classnames[classID] + ":" + cv::format("%.2f", conf),
				cv::Point(left, top),
				cv::FONT_HERSHEY_SIMPLEX, (right - left) / 200, cv::Scalar(0, 255, 0), 2);
		}
		cv::putText(
			frame,
			"FPS: " + std::to_string(int(1000 / (clock() - start))),
			cv::Point(50, 50),
			cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
		imshow("", frame);
		if (cv::waitKey(1) == 27) break;
	}

	cap.release();
	frame.release();
}