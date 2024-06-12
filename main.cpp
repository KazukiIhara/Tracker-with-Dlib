#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>

using namespace dlib;
using namespace std;

int main()
{
	// �J�������J���B����0�Ԃ������ݒ�̃J����
	cv::VideoCapture cap(0);
	if (!cap.isOpened())
	{
		// ������Ȃ�������G���[�R�[�h
		cerr << "Unable to connect to camera" << endl;
		return 1;
	}

	// �E�B���h�E
	image_window win;

	// ��̃����h�}�[�N���f����ǂݍ���
	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor pose_model;
	deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

	// ���C�����[�v
	while (!win.is_closed())
	{

		//
		// �X�V����
		//

		// 1�t���[�����̉摜���󂯎�锠
		cv::Mat temp;
		if (!cap.read(temp))
		{
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

		//
		// �`�揈��
		//

		// ��̃����h�}�[�N�̏ꏊ�ɉ~��`��
		for (uint64_t i = 0; i < shapes.size(); ++i)
		{
			for (uint64_t j = 0; j < shapes[i].num_parts(); ++j)
			{
				dlib::draw_solid_circle(img, shapes[i].part(j), 2, dlib::rgb_pixel(0, 255, 0));
			}
		}

		// �����h�}�[�N�t���̉摜���o��

		win.set_image(img);
	}

	return 0;
}

