#include <cv.h>
#include <highgui.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include "Labeling.h"

using namespace cv;
using namespace std;

const char* WINDOW_TITLE = "puzzle";

int window = 0;
void showImage(Mat image) {
  char name[128];
  sprintf(name, "window %d", window);
  int height = image.size().height;
  double scale = 500.0 / height;
  printf("showImage: %s, height = %d, scale %.3f\n", name, height, scale);

  Mat output;
  resize(image, output, Size(), scale, scale);
  namedWindow(name, window);
  imshow(name, output);

  window++;
}

void drawLine(Mat &img, vector<Point> points, Scalar color, int thickness=1) {
  for (int i = 0; i < points.size() - 1; i++) {
    line(img, points[i], points[i+1], color, thickness);
  }
}

Point3f getLine(Point2f p1, Point2f p2) {
  Point3f w;

  if (p1.x == p2.x) {
    if (p1.y == p2.y) {
      w.x = 0;
      w.y = 0;
      w.z = 0;
    } else {
      w.x = 1;
      w.y = 0;
      w.z = - p1.x;
    }
  } else {
    double a = (p2.y - p1.y) / (p2.x - p1.x);
    double b = p1.y - a * p1.x;
    w.x = a;
    w.y = -1;
    w.z = b;
  }

  return w;
}

double getDistance(Point3f w, Point2f p) {
  return abs(w.x*p.x + w.y*p.y + w.z) / sqrt(w.x*w.x + w.y*w.y);
}

int product(Point a, Point b) {
  return a.x*b.y - a.y*b.x;
}

int side(Point p1, Point p2, Point p3) {
  const int n  = p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y);
  if      ( n > 0 ) return  1;
  else if ( n < 0 ) return -1;
  else              return  0;
}

bool inside(Point p, Point a, Point b, Point c) {
  const int pab = side( p, a, b );
  const int pbc = side( p, b, c );
  const int pca = side( p, c, a );

  if ( (0 <= pab) && (0 <= pbc) && (0 <= pca) )
    return true;
  if ( (0 >= pab) && (0 >= pbc) && (0 >= pca) )
    return true;

  return false;
}

double getAngle(Point p1, Point p2, Point p3) {
  Point v1 = Point(p1.x - p2.x, p1.y - p2.y);
  Point v2 = Point(p3.x - p2.x, p3.y - p2.y);

  double cos = v1.ddot(v2) / (norm(v1) * norm(v2));
  double angle = acos(cos);

  if (side(p1, p2, p3)) {
    angle *= -1;
  }
  
  return angle;
}

double compare(vector<double> f1, vector<double> f2) {
  if (f1.size() != f2.size()) {
    printf("feature vector size is different\n");
    exit(-1);
  }

  double diff = 0.0;
  for (int i = 0; i < f1.size(); i++) {
//     diff += log(1 - abs(f1[i] - f2[f2.size() - 1 - i]) / max(f1[i], f2[f2.size()-1-i]) );
    diff += abs(f1[i] - f2[f2.size() - 1 - i]);
  }

  return diff;
}

class Piece {
public:
  char file[128];
  Mat image;
  Mat blur;
  Mat binary;
  Mat label;
  Mat piece;
  Mat edge;
  vector<Point> contour;
  Point corners[4];
  vector<vector<Point> >edges;//  = { vector<Point>(), vector<Point>(), vector<Point>(), vector<Point>() };
  vector<vector<Point> > simpleEdges;
  bool convex[4];
  vector<vector<double> > features;
  int w;
  int h;

  void imageRead();
  void getPieceImage();
  void getContour();
  void getCorners();
  void getPieceEdge();
  void getSimpleEdge();
  void judgeConvexity();
  void getFeatures();
  Mat getColorEdge();
  
  Piece(const char* file);
};

Piece::Piece(const char* file) {
  strcpy(this->file, file);
  
  imageRead();
}

void Piece::imageRead() {
  image = imread(file, 0);
  int width = image.size().width;
  int height = image.size().height;
  int step = image.step;

  if (image.data == 0) {
    printf("Failed to read image file %s\n", WINDOW_TITLE);
    return;
  }
  printf("%s : %d, %d, %d, %d\n", file, width, height, step, (int)image.elemSize1());

  GaussianBlur(image, blur, Size(7, 7), 1.5, 1.5);
  threshold(blur, binary, 64, 255, THRESH_BINARY);

  getPieceImage();
  getContour();
  getCorners();
  getPieceEdge();
  getSimpleEdge();
  getFeatures();
  judgeConvexity();
}

void Piece::getPieceImage() {

  int width = image.size().width;
  int height = image.size().height;
  int step = image.step;

  Labeling<uchar, uchar> labeling;
  label = Mat(height, width, CV_8UC1);
  printf("%s : %d, %d, %d, %d\n", "label", label.size().width, label.size().height, (int)label.step1(), (int)label.elemSize1());
  labeling.Exec(binary.data, label.data, width, height, true, 10);

  int minX = width;
  int minY = height;
  int maxX = 0;
  int maxY = 0;

  for(int y=0; y<height; ++y) {
    for(int x=0; x<width; ++x) {
      int a = step*y + x;
      if (label.data[a] == 1) {
	label.data[a] = 255;

	if (x < minX)
	  minX = x;
	if (x > maxX)
	  maxX = x;
	if (y < minY)
	  minY = y;
	if (y > maxY)
	  maxY = y;

      } else {
	label.data[a] = 0;

      }
    }
  }

  minX -= 10;
  minY -= 10;
  maxX += 10;
  maxY += 10;

  if (minX < 0)
    minX = 0;
  if (minY < 0)
    minY = 0;
  if (maxX > width)
    maxX = width;
  if (maxY > height)
    maxY = height;

  printf("getPieceImage: (%d, %d), (%d, %d)\n", minX, minY, maxX, maxY);

  w = maxX - minX;
  h = maxY - minY;
  getRectSubPix(label, Size(w, h), Point2f(minX + w/2, minY + h/2), piece);
}

void Piece::getContour() {
  Mat src;
  piece.copyTo(src);
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  findContours(src, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
  contour = contours[0];
}

void Piece::getCorners() {
  edge = Mat::zeros(piece.rows, piece.cols, CV_8UC1);

  vector<double> angle;
  int step = edge.step;
  for (int i = 0 ; i < contour.size(); ++i) {
    int s = (i - 20 + contour.size()) % contour.size();
    int t = (i + 20 + contour.size()) % contour.size();

    angle.push_back(1 - abs(getAngle(contour[s], contour[i], contour[t])) / M_PI);

    edge.data[contour[i].y*step + contour[i].x] = 255 * angle[i]; // 255 * i / contour.size();
  }

  vector<Point> features;
  goodFeaturesToTrack(edge, features, 4, 0.01, 100);

  Mat corner;
  cvtColor(piece, corner, CV_GRAY2RGB);
  vector<Point>::iterator it_feature = features.begin();
  for(; it_feature!=features.end(); ++it_feature) {
    printf("corner : (%d, %d)\n", it_feature->x, it_feature->y);
    circle(corner, Point(it_feature->x, it_feature->y), 1, Scalar(0,200,255), -1);
    circle(corner, Point(it_feature->x, it_feature->y), 8, Scalar(0,200,255));
  }

  for (int i = 0 ; i < features.size(); i++) {
    int cornerX = features[i].x;
    int cornerY = features[i].y;

    double min = 999999;
    int minX, minY;

    for (int j = 0 ; j < contour.size(); j++) {
      double l = sqrt((contour[j].x - cornerX)*(contour[j].x - cornerX) + (contour[j].y - cornerY)*(contour[j].y - cornerY));
      if (l < min) {
	min = l;
	minX = contour[j].x;
	minY = contour[j].y;
      }
    }

    corners[i] = Point(minX, minY);
    printf("corners : (%d, %d)\n", corners[i].x, corners[i].y);
  }
}

void Piece::getPieceEdge() {
  edges = vector<vector<Point> >(4);
  vector<Point> tmp;
  
  int edgeIndex = -1;
  for (int i = 0 ; i < contour.size(); i++) {
    if ((contour[i].x == corners[0].x && contour[i].y == corners[0].y) || 
	(contour[i].x == corners[1].x && contour[i].y == corners[1].y) ||
	(contour[i].x == corners[2].x && contour[i].y == corners[2].y) ||
	(contour[i].x == corners[3].x && contour[i].y == corners[3].y) ) {
      edgeIndex++;
      edgeIndex %= 4;
      printf("corner [%d]: (%d, %d)\n", edgeIndex, contour[i].x, contour[i].y);
    }

    if (edgeIndex == -1) {
      tmp.push_back(contour[i]);
    } else {
      edges[edgeIndex].push_back(contour[i]);
    }
  }

  edges[3].insert(edges[3].end(), tmp.begin(), tmp.end());
}

int findFarestPoint(Point3f w, vector<Point> points, int begin, int end) {
  double max = -1;
  int maxIndex = -1;
  for (int i = begin; i < end; i++) {
    double d = getDistance(w, points[i]);
    if (d > max) {
      max = d;
      maxIndex = i;
    }
  }

  return maxIndex;
}

void Piece::getSimpleEdge() {

  simpleEdges = vector<vector<Point> >(4);

  for (int n = 0; n < 4; n++) {
    vector<int> queue;
    queue.push_back(0);
    queue.push_back(edges[n].size());
    
    for (int k = 0; k < 4; k++) {

      vector<int> new_queue;
      for(int i = 0; i < queue.size() - 1; i++) {
	int start = queue[i];
	int end = queue[i+1];

	new_queue.push_back(start);

	int argMax = findFarestPoint(getLine(edges[n][start], edges[n][end-1]), edges[n], start+1, end-1);
	
	new_queue.push_back(argMax);
      }
      new_queue.push_back(edges[n].size()-1);

      queue = new_queue;
    }

    for (int i = 0; i < queue.size(); i++) {
      simpleEdges[n].push_back(edges[n][queue[i]]);
    }

    /*
    int argMax0 = findFarestPoint(getLine(edges[n][0], edges[n][edges[n].size()-1]),
				  edges[n], 0, edges[n].size());
  
    int argMax11 = findFarestPoint(getLine(edges[n][0], edges[n][argMax0]),
				   edges[n], 0, argMax0);
    int argMax12 = findFarestPoint(getLine(edges[n][argMax0], edges[n][edges[n].size()-1]),
				   edges[n], argMax0, edges[n].size());

    int argMax21 = findFarestPoint(getLine(edges[n][0], edges[n][argMax11]),
				   edges[n], 0, argMax11);
    int argMax22 = findFarestPoint(getLine(edges[n][argMax11], edges[n][argMax0]),
				   edges[n], argMax11, argMax0);
    int argMax23 = findFarestPoint(getLine(edges[n][argMax0], edges[n][argMax12]),
				   edges[n], argMax0, argMax12);
    int argMax24 = findFarestPoint(getLine(edges[n][argMax12], edges[n][edges[n].size()-1]),
				   edges[n], argMax12, edges[n].size());

    int argMax31 = findFarestPoint(getLine(edges[n][argMax11], edges[n][argMax22]),
				   edges[n], argMax11, argMax22);
    int argMax32 = findFarestPoint(getLine(edges[n][argMax22], edges[n][argMax0]),
				   edges[n], argMax22, argMax0);
    int argMax33 = findFarestPoint(getLine(edges[n][argMax0], edges[n][argMax23]),
				   edges[n], argMax0, argMax23);
    int argMax34 = findFarestPoint(getLine(edges[n][argMax23], edges[n][argMax12]),
				   edges[n], argMax23, argMax12);
    
    simpleEdges[n].push_back(edges[n][0]);
    simpleEdges[n].push_back(edges[n][argMax21]);
    simpleEdges[n].push_back(edges[n][argMax11]);
    simpleEdges[n].push_back(edges[n][argMax31]);
    simpleEdges[n].push_back(edges[n][argMax22]);
    simpleEdges[n].push_back(edges[n][argMax32]);
    simpleEdges[n].push_back(edges[n][argMax0]);
    simpleEdges[n].push_back(edges[n][argMax33]);
    simpleEdges[n].push_back(edges[n][argMax23]);
    simpleEdges[n].push_back(edges[n][argMax34]);
    simpleEdges[n].push_back(edges[n][argMax12]);
    simpleEdges[n].push_back(edges[n][argMax24]);
    simpleEdges[n].push_back(edges[n][edges[n].size()-1]);
    */
  }
}

void Piece::judgeConvexity() {
  int center = simpleEdges[0].size() / 2 + 1;
  for (int n = 0; n < 4; n++) {
    if (inside(simpleEdges[n][center], corners[0], corners[1], corners[2]) ||
	inside(simpleEdges[n][center], corners[2], corners[3], corners[0])) {
      convex[n] = true;
    } else {
      convex[n] = false;
    }

//     printf("edge [%d] is %d\n", n, convex[n]);
  }
}
void Piece::getFeatures() {

  features = vector<vector<double> >(4);

  for (int n = 0; n < 4; n++) {

    double lenSum = 0;
    vector<double> origLengthes;
    
    for (int i = 0; i < simpleEdges[n].size() - 1; i++) {
      Point v = Point(simpleEdges[n][i+1].x - simpleEdges[n][i].x, simpleEdges[n][i+1].y - simpleEdges[n][i].y);
      double length = norm(v);
      lenSum += length;
      origLengthes.push_back(length);
    }

    double lenNormWeight = simpleEdges[n].size() / lenSum;

    for (int i = 0; i < simpleEdges[n].size() - 1; i++) {
      double length = lenNormWeight * origLengthes[i];
      printf("%.2f ", length);
//       features[n].push_back(length);
    }
    printf("\n");

    for (int i = 1; i < simpleEdges[n].size() - 1; i++) {
      double theta = M_PI - getAngle(simpleEdges[n][i-1], simpleEdges[n][i], simpleEdges[n][i+1]);
      printf("%.2f ", theta);
      features[n].push_back(theta);
    }
    printf("\n");

  }
}

Mat Piece::getColorEdge() {
  Mat result = Mat::zeros(piece.rows, piece.cols, CV_8UC3);
  drawLine(result, edges[0], Scalar(255, 0, 0), 2);
  drawLine(result, edges[1], Scalar(0, 255, 0), 2);
  drawLine(result, edges[2], Scalar(0, 0, 255), 2);
  drawLine(result, edges[3], Scalar(255, 0, 255), 2);

  for (int n = 0; n < 4; n++) {
    drawLine(result, simpleEdges[n], Scalar(255, 255, 255));
    for (int k = 0; k < simpleEdges[n].size(); k++) {
      circle(result, simpleEdges[n][k], 5, Scalar(0, 255, 255));
    }
  }

  return result;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("usage: %s image_file\n", argv[0]);
    return -1;
  }

  const char* file1 = argv[1];
  const char* file2 = argv[2];

  Piece piece1(file1);
  Piece piece2(file2);

  for (int m = 0; m < 4; m++) {
    for (int n = 0; n < 4; n++) {
      if (piece1.convex[m] != piece2.convex[n]) {
	printf("piece1 [%d] : piece2 [%d] : %.3f\n", m, n, compare(piece1.features[m], piece2.features[n]));
      }
    }
  }
    
  showImage(piece1.getColorEdge());
  showImage(piece2.getColorEdge());

  waitKey(0);
  return 0;
}
