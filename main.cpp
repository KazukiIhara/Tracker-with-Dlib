#define NOMINMAX
#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> 
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <algorithm>

using namespace dlib;
using namespace std;


int main() {
	// カメラを開く。引数0番が初期設定のカメラ
	cv::VideoCapture cap(0);
	if (!cap.isOpened()) {
		// 見つからなかったらエラーコード
		cerr << "Unable to connect to camera" << endl;
		return 1;
	}

	// ウィンドウ
	image_window win;
	image_window win_left_eye; // 左目用のウィンドウ
	image_window win_right_eye; // 右目用のウィンドウ

	// 顔のランドマークモデルを読み込み
	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor pose_model;
	deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
	cv::Mat temp;

	// 目を検出するためのHaar Cascadeを読み込みます。
	cv::CascadeClassifier eye_cascade;
	if (!eye_cascade.load("haarcascade_eye.xml")) {
		std::cout << "Error loading haarcascade_eye.xml" << std::endl;
		return -1;
	}

	// メインループ
	while (!win.is_closed()) {
		// 1フレーム分の画像を受け取る箱
		temp.release();
		if (!cap.read(temp)) {
			break;
		}

		// 顔検出用の画像変数(BGR)
		cv_image<bgr_pixel> cimg(temp);

		// 顔を検出する 
		std::vector<rectangle> faces = detector(cimg);
		std::vector<full_object_detection> shapes;
		for (uint64_t i = 0; i < faces.size(); ++i)
			shapes.push_back(pose_model(cimg, faces[i]));

		win.clear_overlay();
		win.set_image(cimg);

		// 描画用の画像変数(RGB)
		dlib::array2d<dlib::rgb_pixel> img;
		// 画像をコピー
		assign_image(img, cimg);

		// 顔のランドマークの場所に円を描画
		for (uint64_t i = 0; i < shapes.size(); ++i) {
			for (uint64_t j = 0; j < shapes[i].num_parts(); ++j) {
				dlib::draw_solid_circle(img, shapes[i].part(j), 2, dlib::rgb_pixel(0, 255, 0));
			}


			// 目の部分を切り取る
			int margin = 5; // 追加するマージン（ピクセル単位）
			cv::Rect left_eye(std::max(0, static_cast<int>(shapes[i].part(42).x())), std::max(0, static_cast<int>(shapes[i].part(43).y()) - margin), std::min(temp.cols, static_cast<int>(shapes[i].part(45).x())) - std::max(0, static_cast<int>(shapes[i].part(42).x())), std::min(temp.rows, static_cast<int>(shapes[i].part(47).y()) + margin) - std::max(0, static_cast<int>(shapes[i].part(43).y()) - margin));
			cv::Rect right_eye(std::max(0, static_cast<int>(shapes[i].part(36).x())), std::max(0, static_cast<int>(shapes[i].part(38).y()) - margin), std::min(temp.cols, static_cast<int>(shapes[i].part(39).x())) - std::max(0, static_cast<int>(shapes[i].part(36).x())), std::min(temp.rows, static_cast<int>(shapes[i].part(40).y()) + margin) - std::max(0, static_cast<int>(shapes[i].part(38).y()) - margin));

			cv::Mat left_eye_img = temp(left_eye);
			cv::Mat right_eye_img = temp(right_eye);

			// アスペクト比を保持したまま画像をリサイズ
			int desired_size = 300; // 望むサイズ
			double aspect_ratio = static_cast<double>(left_eye_img.cols) / static_cast<double>(left_eye_img.rows);
			int new_width = desired_size;
			int new_height = static_cast<int>(desired_size / aspect_ratio);
			cv::resize(left_eye_img, left_eye_img, cv::Size(new_width, new_height));

			aspect_ratio = static_cast<double>(right_eye_img.cols) / static_cast<double>(right_eye_img.rows);
			new_width = desired_size;
			new_height = static_cast<int>(desired_size / aspect_ratio);
			cv::resize(right_eye_img, right_eye_img, cv::Size(new_width, new_height));

			// 平滑化
			cv::GaussianBlur(left_eye_img, left_eye_img, cv::Size(7, 7), 0);
			cv::GaussianBlur(right_eye_img, right_eye_img, cv::Size(7, 7), 0);



			// 二値化
			uint32_t thresholdValue = 8; // しきい値(小さくするほどより濃い黒のみを検出する)
			cv::Mat left_eye_img_binary, right_eye_img_binary;
			cv::threshold(left_eye_img, left_eye_img_binary, thresholdValue, 255, cv::THRESH_BINARY_INV);
			cv::threshold(right_eye_img, right_eye_img_binary, thresholdValue, 255, cv::THRESH_BINARY_INV);

			// グレースケールに変換
			cv::Mat left_eye_img_gray, right_eye_img_gray;
			cv::cvtColor(left_eye_img_binary, left_eye_img_gray, cv::COLOR_BGR2GRAY);
			cv::cvtColor(right_eye_img_binary, right_eye_img_gray, cv::COLOR_BGR2GRAY);

			// 輪郭の抽出
			std::vector<std::vector<cv::Point>> contours_left, contours_right;
			cv::findContours(left_eye_img_gray, contours_left, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
			cv::findContours(right_eye_img_gray, contours_right, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

			// 最大の輪郭を見つける
			double maxArea_left = 0, maxArea_right = 0;
			std::vector<cv::Point> maxContour_left, maxContour_right;

			for (size_t i = 0; i < contours_left.size(); i++) {
				double area = cv::contourArea(contours_left[i]);
				if (area > maxArea_left) {
					maxArea_left = area;
					maxContour_left = contours_left[i];
				}
			}

			for (size_t i = 0; i < contours_right.size(); i++) {
				double area = cv::contourArea(contours_right[i]);
				if (area > maxArea_right) {
					maxArea_right = area;
					maxContour_right = contours_right[i];
				}
			}

			// 最大の輪郭の中心を見つける
			cv::Moments moments_left = cv::moments(maxContour_left, false);
			cv::Moments moments_right = cv::moments(maxContour_right, false);

			cv::Point center_left(moments_left.m10 / moments_left.m00, moments_left.m01 / moments_left.m00);
			cv::Point center_right(moments_right.m10 / moments_right.m00, moments_right.m01 / moments_right.m00);

			// 中心に青い円を描画する
			cv::circle(left_eye_img, center_left, 10, cv::Scalar(255, 0, 0), -1);
			cv::circle(right_eye_img, center_right, 10, cv::Scalar(255, 0, 0), -1);

			cv::imshow("Left Eye", left_eye_img);
			cv::imshow("Right Eye", right_eye_img);
			cv::imshow("Left Eye Gray", left_eye_img_gray);
			cv::imshow("Right Eye Gray", right_eye_img_gray);
		}

		// ランドマーク付きの画像を出力
		win.set_image(img);
		cv::waitKey(1); // OpenCVのウィンドウがすぐに閉じないようにする
	}

	return 0;
}