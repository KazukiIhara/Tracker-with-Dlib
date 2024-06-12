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
	// カメラを開く。引数0番が初期設定のカメラ
	cv::VideoCapture cap(0);
	if (!cap.isOpened())
	{
		// 見つからなかったらエラーコード
		cerr << "Unable to connect to camera" << endl;
		return 1;
	}

	// ウィンドウ
	image_window win;

	// 顔のランドマークモデルを読み込み
	frontal_face_detector detector = get_frontal_face_detector();
	shape_predictor pose_model;
	deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

	// メインループ
	while (!win.is_closed())
	{

		//
		// 更新処理
		//

		// 1フレーム分の画像を受け取る箱
		cv::Mat temp;
		if (!cap.read(temp))
		{
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

		//
		// 描画処理
		//

		// 顔のランドマークの場所に円を描画
		for (uint64_t i = 0; i < shapes.size(); ++i)
		{
			for (uint64_t j = 0; j < shapes[i].num_parts(); ++j)
			{
				dlib::draw_solid_circle(img, shapes[i].part(j), 2, dlib::rgb_pixel(0, 255, 0));
			}
		}

		// ランドマーク付きの画像を出力

		win.set_image(img);
	}

	return 0;
}

