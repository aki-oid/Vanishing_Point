#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <windows.h>
#include <algorithm>

using namespace cv;
using namespace std;

// 2つの直線の交点を計算する関数
Point2f computeIntersect(Vec4i a, Vec4i b)
{
    // 直線1のパラメータ
    double x1 = a[0], y1 = a[1], x2 = a[2], y2 = a[3];
    // 直線2のパラメータ
    double x3 = b[0], y3 = b[1], x4 = b[2], y4 = b[3];

    // 分母の計算
    double det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);

    // 平行な直線は交点なし
    if (abs(det) < 1e-6) {
        return Point2f(-1, -1);
    }

    // 交点座標の計算
    double t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / det;
    double x = x1 + t * (x2 - x1);
    double y = y1 + t * (y2 - y1);

    return Point2f(x, y);
}

// 線分を画面端まで伸ばして描画する関数
void drawExtendedLine(Mat& image, const Vec4i& line_segment, const Scalar& color, int thickness) {
    double x1 = line_segment[0], y1 = line_segment[1];
    double x2 = line_segment[2], y2 = line_segment[3];
    int rows = image.rows, cols = image.cols;

    // 線分の向きを考慮して、延長点を計算
    Point pt1, pt2;
    if (abs(x2 - x1) < 1e-6) { // 垂直な線
        pt1 = Point(x1, 0);
        pt2 = Point(x1, rows);
    }
    else {
        double slope = (y2 - y1) / (x2 - x1);
        double intercept = y1 - slope * x1;

        // 左端 (x = 0)
        double y_at_left = intercept;
        // 右端 (x = cols)
        double y_at_right = slope * cols + intercept;
        // 上端 (y = 0)
        double x_at_top = -intercept / slope;
        // 下端 (y = rows)
        double x_at_bottom = (rows - intercept) / slope;

        vector<Point> boundary_points;
        if (y_at_left >= 0 && y_at_left <= rows) boundary_points.push_back(Point(0, y_at_left));
        if (y_at_right >= 0 && y_at_right <= rows) boundary_points.push_back(Point(cols, y_at_right));
        if (x_at_top >= 0 && x_at_top <= cols) boundary_points.push_back(Point(x_at_top, 0));
        if (x_at_bottom >= 0 && x_at_bottom <= cols) boundary_points.push_back(Point(x_at_bottom, rows));

        // 元の線分が境界点を含む場合があるため、重複を避ける
        if (boundary_points.size() < 2) {
            // 延長線が画面外にしかない場合
            if (x2 - x1 > 0) {
                pt1 = Point(0, intercept);
                pt2 = Point(cols, slope * cols + intercept);
            }
            else {
                pt1 = Point(cols, slope * cols + intercept);
                pt2 = Point(0, intercept);
            }
        }
        else {
            // 境界線との2つの交点を線分の両端とする
            pt1 = boundary_points[0];
            pt2 = boundary_points.size() > 1 ? boundary_points[1] : boundary_points[0];
        }
    }
    line(image, pt1, pt2, color, thickness, LINE_AA);
}

int main(int argc, char** argv)
{
    Mat src = imread("image/pic3.jpg", IMREAD_COLOR);
    if (src.empty()) {
        cerr << "Error: image not found." << endl;
        return -1;
    }

    Mat resized_img, drawn_img;
    double scale = 1.0;
    if (src.cols > 1000 || src.rows > 1000) {
        scale = min(1000.0 / src.cols, 1000.0 / src.rows);
    }
    resize(src, resized_img, Size(), scale, scale, INTER_AREA);
    resize(src, drawn_img, Size(), scale, scale, INTER_AREA);

    Mat gray;
    cvtColor(drawn_img, gray, COLOR_BGR2GRAY);

    Mat edges;
    Canny(gray, edges, 50, 150, 3);

    vector<Vec4i> lines;
    HoughLinesP(edges, lines, 1, CV_PI / 180, 75, 75, 7);//美術絵画用
    //HoughLinesP(edges, lines, 1, CV_PI / 180, 200, 100, 5);//風景写真用・・・ノイズが多いため値を調整

    // 検出された線分を画面端まで伸ばして緑色で描画
    for (const auto& l : lines) {
        drawExtendedLine(drawn_img, l, Scalar(0, 255, 0), 1);
    }

    // 交点とそれぞれの重みを格納するマップ
    vector<pair<Point2f, int>> intersection_weights;

    for (size_t i = 0; i < lines.size(); ++i) {
        for (size_t j = i + 1; j < lines.size(); ++j) {
            Vec4i l1 = lines[i];
            Vec4i l2 = lines[j];

            // ほぼ水平な線は無視する
            double angle1 = atan2(l1[3] - l1[1], l1[2] - l1[0]);
            double angle2 = atan2(l2[3] - l2[1], l2[2] - l2[0]);

            double a1 = abs(angle1);
            if (a1 > CV_PI / 2) a1 = CV_PI - a1;

            double a2 = abs(angle2);
            if (a2 > CV_PI / 2) a2 = CV_PI - a2;

            if (a1 < CV_PI / 180 * 10 || a2 < CV_PI / 180 * 10) {
                continue;
            }

            Point2f intersect = computeIntersect(l1, l2);
            if (intersect.x == -1) continue;

            if (intersect.x < -drawn_img.cols || intersect.x > drawn_img.cols * 2 ||
                intersect.y < -drawn_img.rows || intersect.y > drawn_img.rows * 2) {
                continue;
            }

            double w1 = 1.0 - abs(a1 - CV_PI / 4) / (CV_PI / 4);
            double w2 = 1.0 - abs(a2 - CV_PI / 4) / (CV_PI / 4);
            int score = (int)(w1 * w2 * 100);

            // 既存の交点との距離が近いかチェックし、重みを加算
            bool found_cluster = false;
            for (auto& pair : intersection_weights) {
                if (norm(pair.first - intersect) < 10.0) { // 許容範囲を50ピクセルに設定
                    pair.second += score;
                    found_cluster = true;
                    break;
                }
            }
            if (!found_cluster) {
                intersection_weights.push_back({ intersect, score });
            }
        }
    }

    if (intersection_weights.empty()) {
        cout << "No vanishing point found." << endl;
        imshow("Vanishing Point", drawn_img);
        waitKey(0);
        return 0;
    }

    // 最も重みの高い交点を見つける
    Point2f vanishingPoint;
    int maxWeight = 0;
    for (const auto& pair : intersection_weights) {
        if (pair.second > maxWeight) {
            maxWeight = pair.second;
            vanishingPoint = pair.first;
        }
    }

    // 消失点を画像に描画
    circle(resized_img, vanishingPoint, 10, Scalar(0, 0, 255), 3);
    circle(drawn_img, vanishingPoint, 10, Scalar(0, 0, 255), 3);

    imshow("Vanishing Point only", resized_img);
    imshow("Vanishing Point & Line", drawn_img);
    waitKey(0);

    return 0;
}