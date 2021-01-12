#include <stdio.h>
#include <fstream>
#include <vector>
#include <time.h>
#include <opencv2/opencv.hpp>

#define INWIDTH 300
#define INHEIGHT 300


void runMobileNetSSD(int cameraID, char* model, char* config, int frameWidth, int frameHeight, float threshold) {
	printf("------Running MobileNet SSD!------\n");

	// Initialize network
	cv::dnn::Net net = cv::dnn::readNetFromTensorflow(model, config);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);

	// Load COCO names
	std::vector<std::string> classnames;
	std::ifstream f("coco91.names");
	std::string name = "";
	while (std::getline(f, name)) {
		classnames.push_back(name);
	}

	// Initialize VideoCapture
	cv::VideoCapture cap(cameraID);
	cap.set(cv::CAP_PROP_FRAME_WIDTH, frameWidth);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, frameHeight);

	cv::Mat frame;
	cv::Mat output;
	while (cap.isOpened()) {
		clock_t start = clock();

		// Forward inference
		cap.read(frame);
		net.setInput(cv::dnn::blobFromImage(frame, 1. / 255, cv::Size(INWIDTH, INHEIGHT), true, false));
		output = net.forward();
		cv::Mat detectionMat(output.size[2], output.size[3], CV_32F, output.ptr<float>());
		
		// Visualize result
		for (int i = 0; i < detectionMat.rows; i++) {
			float score = detectionMat.at<float>(i, 2);

			if (score > threshold) {
				int classID = (int)(detectionMat.at<float>(i, 1));
				float left = static_cast<float>(detectionMat.at<float>(i, 3) * frameWidth);
				float top = static_cast<float>(detectionMat.at<float>(i, 4) * frameHeight);
				float right = static_cast<float>(detectionMat.at<float>(i, 5) * frameWidth);
				float bottom = static_cast<float>(detectionMat.at<float>(i, 6) * frameHeight);

				cv::rectangle(frame, cv::Rect(left, top, (right - left), (bottom - top)), cv::Scalar(0, 255, 0), 2);
				cv::putText(
					frame,
					classnames[classID] + ":" + cv::format("%.2f", score),
					cv::Point(left, top),
					cv::FONT_HERSHEY_SIMPLEX, (right - left) / 200, cv::Scalar(0, 255, 0), 2);
			}
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
	output.release();
}