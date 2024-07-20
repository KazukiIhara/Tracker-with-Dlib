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
	// �J�������J���B����0�Ԃ������ݒ�̃J����
	cv::VideoCapture cap(0);
	if (!cap.isOpened()) {
		// ������Ȃ�������G���[�R�[�h
		cerr << "Unable to connect to camera" << endl;
		return 1;
	}

	// �E�B���h�E
	image_window win;
	image_window win_left_eye; // ���ڗp�̃E�B���h�E
	image_window win_right_eye; // �E�ڗp�̃E�B���h�E

	// ��̃����h�}�[�N���f����ǂݍ���
	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor pose_model;
	deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;
	cv::Mat temp;

	// Haar Cascade�̓ǂݍ���
	cv::CascadeClassifier eye_cascade;
	if (!eye_cascade.load("haarcascade_eye.xml")) {
		std::cout << "Error loading haarcascade_eye.xml" << std::endl;
		return -1;
	}

	// ���C�����[�v
	while (!win.is_closed()) {
		// 1�t���[�����̉摜���󂯎�锠
		temp.release();
		if (!cap.read(temp)) {
			break;
		}

		// �猟�o�p�̉摜�ϐ�(BGR)
		cv_image<bgr_pixel> cimg(temp);

		// ������o���� 
		std::vector<rectangle> faces = detector(cimg);
		std::vector<full_object_detection> shapes;
		for (uint64_t i = 0; i < faces.size(); ++i)
			shapes.push_back(pose_model(cimg, faces[i]));

		win.clear_overlay();
		win.set_image(cimg);

		// �`��p�̉摜�ϐ�(RGB)
		dlib::array2d<dlib::rgb_pixel> img;
		// �摜���R�s�[
		assign_image(img, cimg);

		// ��̃����h�}�[�N�̏ꏊ�ɉ~��`��
		for (uint64_t i = 0; i < shapes.size(); ++i) {
			for (uint64_t j = 0; j < shapes[i].num_parts(); ++j) {
				dlib::draw_solid_circle(img, shapes[i].part(j), 2, dlib::rgb_pixel(0, 255, 0));
			}


			// �ڂ̕�����؂���
			int margin = 15; // �ǉ�����}�[�W���i�s�N�Z���P�ʁj
			cv::Rect left_eye(std::max(0, static_cast<int>(shapes[i].part(42).x()) - margin), std::max(0, static_cast<int>(shapes[i].part(43).y()) - margin), std::min(temp.cols, static_cast<int>(shapes[i].part(45).x()) + margin) - std::max(0, static_cast<int>(shapes[i].part(42).x()) - margin), std::min(temp.rows, static_cast<int>(shapes[i].part(47).y()) + margin) - std::max(0, static_cast<int>(shapes[i].part(43).y()) - margin));
			cv::Rect right_eye(std::max(0, static_cast<int>(shapes[i].part(36).x()) - margin), std::max(0, static_cast<int>(shapes[i].part(37).y()) - margin), std::min(temp.cols, static_cast<int>(shapes[i].part(39).x()) + margin) - std::max(0, static_cast<int>(shapes[i].part(36).x()) - margin), std::min(temp.rows, static_cast<int>(shapes[i].part(41).y()) + margin) - std::max(0, static_cast<int>(shapes[i].part(37).y()) - margin));			cv::Mat left_eye_img = temp(left_eye);
			cv::Mat right_eye_img = temp(right_eye);

			// �摜�����T�C�Y
			cv::resize(left_eye_img, left_eye_img, cv::Size(300, 300));
			cv::resize(right_eye_img, right_eye_img, cv::Size(300, 300));

			// �O���[�X�P�[���ɕϊ�
			cv::Mat gray_left_eye, gray_right_eye;
			cv::cvtColor(left_eye_img, gray_left_eye, cv::COLOR_BGR2GRAY);
			cv::cvtColor(right_eye_img, gray_right_eye, cv::COLOR_BGR2GRAY);

			// �K�E�V�A���u���[��K�p
			cv::GaussianBlur(gray_left_eye, gray_left_eye, cv::Size(7, 7), 0);
			cv::GaussianBlur(gray_right_eye, gray_right_eye, cv::Size(7, 7), 0);

			// Otsu�̓�l��
			cv::Mat th_left_eye, th_right_eye;
			double thresh_left_eye = cv::threshold(gray_left_eye, th_left_eye, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
			double thresh_right_eye = cv::threshold(gray_right_eye, th_right_eye, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

			// �֊s�����o
			std::vector<std::vector<cv::Point>> contours_left_eye, contours_right_eye;
			cv::findContours(th_left_eye, contours_left_eye, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
			cv::findContours(th_right_eye, contours_right_eye, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

			// �ő�̗֊s��������
			std::vector<cv::Point> cnt_left_eye = contours_left_eye[0];
			for (const auto& c : contours_left_eye) {
				if (cnt_left_eye.size() < c.size()) {
					cnt_left_eye = c;
				}
			}
			std::vector<cv::Point> cnt_right_eye = contours_right_eye[0];
			for (const auto& c : contours_right_eye) {
				if (cnt_right_eye.size() < c.size()) {
					cnt_right_eye = c;
				}
			}

			// �ȉ~�t�B�b�e�B���O
			if (cnt_left_eye.size() > 5) {
				cv::RotatedRect ellipse = cv::fitEllipse(cnt_left_eye);
				cv::Point2f center = ellipse.center;
				// �d�S�ɉ~��`��
				cv::circle(left_eye_img, center, 5, cv::Scalar(0, 255, 255), -1);
			}

			if (cnt_right_eye.size() > 5) {
				cv::RotatedRect ellipse = cv::fitEllipse(cnt_right_eye);
				cv::Point2f center = ellipse.center;
				// �d�S�ɉ~��`��
				cv::circle(right_eye_img, center, 5, cv::Scalar(0, 255, 255), -1);
			}

			cv::imshow("Left Eye", left_eye_img);
			cv::imshow("Right Eye", right_eye_img);
		}

		// �����h�}�[�N�t���̉摜���o��
		win.set_image(img);
		cv::waitKey(1); // OpenCV�̃E�B���h�E�������ɕ��Ȃ��悤�ɂ���
	}

	return 0;
}