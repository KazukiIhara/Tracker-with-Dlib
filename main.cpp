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
			cv::Rect left_eye(std::max(0, static_cast<int>(shapes[i].part(42).x()) - margin), std::max(0, static_cast<int>(shapes[i].part(43).y()) - margin), std::min(temp.cols, static_cast<int>(shapes[i].part(45).x()) + margin) - std::max(0, static_cast<int>(shapes[i].part(42).x()) - margin), std::min(temp.rows, static_cast<int>(shapes[i].part(46).y()) + margin) - std::max(0, static_cast<int>(shapes[i].part(43).y()) - margin));
			cv::Rect right_eye(std::max(0, static_cast<int>(shapes[i].part(36).x()) - margin), std::max(0, static_cast<int>(shapes[i].part(37).y()) - margin), std::min(temp.cols, static_cast<int>(shapes[i].part(40).x()) + margin) - std::max(0, static_cast<int>(shapes[i].part(36).x()) - margin), std::min(temp.rows, static_cast<int>(shapes[i].part(39).y()) + margin) - std::max(0, static_cast<int>(shapes[i].part(37).y()) - margin));			cv::Mat left_eye_img = temp(left_eye);
			cv::Mat right_eye_img = temp(right_eye);

			// �摜�����T�C�Y
			cv::resize(left_eye_img, left_eye_img, cv::Size(300, 300));
			cv::resize(right_eye_img, right_eye_img, cv::Size(300, 300));

			cv::imshow("Left Eye", left_eye_img);
			cv::imshow("Right Eye", right_eye_img);

			cv::imshow("Left Eye", left_eye_img);
			cv::imshow("Right Eye", right_eye_img);
		}


		// �����h�}�[�N�t���̉摜���o��
		win.set_image(img);
		cv::waitKey(1); // OpenCV�̃E�B���h�E�������ɕ��Ȃ��悤�ɂ���
	}

	return 0;
}
