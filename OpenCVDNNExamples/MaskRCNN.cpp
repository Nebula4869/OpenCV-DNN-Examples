#include <stdio.h>
#include <fstream>
#include <vector>
#include <time.h>
#include <opencv2/opencv.hpp>

#define INWIDTH 800
#define INHEIGHT 800


void runMaskRCNN(char* imagePath, char* model, char* config, float threshold) {
	printf("------Running Mask RCNN!------\n");

	// Initialize network
	cv::dnn::Net net = cv::dnn::readNetFromTensorflow(model, config);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);

	// Get output node
	std::vector<cv::String> out_names;
	out_names.push_back("detection_out_final");
	out_names.push_back("detection_masks");

	// Load COCO names
	std::vector<std::string> classnames;
	std::ifstream f("coco.names");
	std::string name = "";
	while (std::getline(f, name)) {
		classnames.push_back(name);
	}

	// Forward inference
	cv::Mat image = cv::imread(imagePath);
	net.setInput(cv::dnn::blobFromImage(image, 1.0, cv::Size(INWIDTH, INHEIGHT), cv::Scalar(0, 0, 0), true, false));
	std::vector<cv::Mat> outs;
	clock_t start = clock();
	net.forward(outs, out_names);
	printf("Inference Time: %dms\n", clock() - start);
	cv::Mat detection = outs[0];
	cv::Mat masks = outs[1]; // Nx90x15x15
	cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
	
	// Visualize result
	for (int i = 0; i < detectionMat.rows; i++) {
		float score = detectionMat.at<float>(i, 2);
		if (score > threshold) {
			int classID = (int)(detectionMat.at<float>(i, 1));
			float left = static_cast<float>(detectionMat.at<float>(i, 3) * image.cols);
			float top = static_cast<float>(detectionMat.at<float>(i, 4) * image.rows);
			float right = static_cast<float>(detectionMat.at<float>(i, 5) * image.cols);
			float bottom = static_cast<float>(detectionMat.at<float>(i, 6) * image.rows);

			cv::rectangle(image, cv::Rect(left, top, (right - left), (bottom - top)), cv::Scalar(0, 255, 0), 2);

			cv::putText(image,
				classnames[classID] + ": " + cv::format("%.2f", score),
				cv::Point(left, top),
				cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));

			cv::Mat mask(masks.size[2], masks.size[3], CV_32F, masks.ptr<float>(i, classID));
			cv::Mat color_mask = cv::Mat::zeros(mask.size(), CV_8UC3);
			cv::Mat bin_mask = cv::Mat::zeros(mask.size(), CV_8UC1);
			for (int row = 0; row < color_mask.rows; row++) {
				for (int col = 0; col < color_mask.cols; col++) {
					float m = mask.at<float>(row, col);
					if (m >= 0.5) {
						color_mask.at<cv::Vec3b>(row, col) = cv::Vec3b(0, 0, 255);
						bin_mask.at<uchar>(row, col) = 255;
					}
				}
			}
			cv::Mat roi = image(cv::Rect(left, top, (right - left), (bottom - top)));
			cv::resize(color_mask, color_mask, roi.size());
			cv::resize(bin_mask, bin_mask, roi.size());
			cv::Mat result;
			cv::bitwise_and(roi, roi, result, bin_mask);
			cv::addWeighted(roi, 0.5, color_mask, 0.5, 0, roi);
		}
	}
	imshow("", image);
	cv::waitKey();
}