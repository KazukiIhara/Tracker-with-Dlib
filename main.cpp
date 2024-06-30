#define NOMINMAX
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
			int margin = 15; // 追加するマージン（ピクセル単位）
			cv::Rect left_eye(std::max(0, static_cast<int>(shapes[i].part(42).x()) - margin), std::max(0, static_cast<int>(shapes[i].part(43).y()) - margin), std::min(temp.cols, static_cast<int>(shapes[i].part(45).x()) + margin) - std::max(0, static_cast<int>(shapes[i].part(42).x()) - margin), std::min(temp.rows, static_cast<int>(shapes[i].part(46).y()) + margin) - std::max(0, static_cast<int>(shapes[i].part(43).y()) - margin));
			cv::Rect right_eye(std::max(0, static_cast<int>(shapes[i].part(36).x()) - margin), std::max(0, static_cast<int>(shapes[i].part(37).y()) - margin), std::min(temp.cols, static_cast<int>(shapes[i].part(40).x()) + margin) - std::max(0, static_cast<int>(shapes[i].part(36).x()) - margin), std::min(temp.rows, static_cast<int>(shapes[i].part(39).y()) + margin) - std::max(0, static_cast<int>(shapes[i].part(37).y()) - margin));			cv::Mat left_eye_img = temp(left_eye);
			cv::Mat right_eye_img = temp(right_eye);

			// 画像をリサイズ
			cv::resize(left_eye_img, left_eye_img, cv::Size(300, 300));
			cv::resize(right_eye_img, right_eye_img, cv::Size(300, 300));

			cv::imshow("Left Eye", left_eye_img);
			cv::imshow("Right Eye", right_eye_img);

			cv::imshow("Left Eye", left_eye_img);
			cv::imshow("Right Eye", right_eye_img);
		}


		// ランドマーク付きの画像を出力
		win.set_image(img);
		cv::waitKey(1); // OpenCVのウィンドウがすぐに閉じないようにする
	}

	return 0;
}
