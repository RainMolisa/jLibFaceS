#pragma once

//#include<vector>
#ifndef noStream
#include<fstream>
#endif
namespace bx {
	struct point {
	public:
		float x;
		float y;
	public:
		point(float x, float y) :x(x), y(y) {};
	};
	class faceBox {
	public:
		float data[15];
	public:
		faceBox() {};
		faceBox(point p1,point p2);
	public:
		void setScore(float score) { data[14] = score; };
		float getScore() { return data[14]; };
		inline float x1() const { return data[0]; }
		inline float y1() const { return data[1]; }
		inline float x2() const { return data[2]; }
		inline float y2() const { return data[3]; }
		point operator[](int i);
		void scale(float w,float h);
		bool islegal();
#ifndef noStream
		friend std::ostream& operator<<(std::ostream& out, faceBox& fb);
		friend std::istream& operator>>(std::istream& out, faceBox& fb);
#endif
	};
	bool faceBox::islegal() {
		for (int i = 0; i < 14; i++) {
			if (data[i] < 0)
				return false;
			else
				if (data[i] > 1)
					return false;
		}
		return true;
	}
	void faceBox::scale(float w, float h) {
		data[0] = data[0] * w;
		data[1] = data[1] * h;
		data[2] = data[2] * w;
		data[3] = data[3] * h;
		data[4] = data[4] * w;
		data[5] = data[5] * h;
		data[6] = data[6] * w;
		data[7] = data[7] * h;
		data[8] = data[8] * w;
		data[9] = data[9] * h;
		data[10] = data[10] * w;
		data[11] = data[11] * h;
		data[12] = data[12] * w;
		data[13] = data[13] * h;
	}
	faceBox::faceBox(point p1, point p2) {
		data[0] = p1.x;
		data[1] = p1.y;
		data[2] = p2.x;
		data[3] = p2.y;
	}
	float JaccardOverlap(const faceBox& a,const faceBox& b) {
		float minx = a.x1() < b.x1() ? a.x1() : b.x1();
		float miny = a.y1() < b.y1() ? a.y1() : b.y1();
		float maxx = a.x2() > b.x2() ? a.x2() : b.x2();
		float maxy = a.y2() > b.y2() ? a.y2() : b.y2();
		float unio = (maxy - miny) * (maxx - minx);
		float xx1 = a.x1() > b.x1() ? a.x1() : b.x1();
		float xx2 = a.x2() < b.x2() ? a.x2() : b.x2();
		float yy1 = a.y1() > b.y1() ? a.y1() : b.y1();
		float yy2 = a.y2() < b.y2() ? a.y2() : b.y2();
		float intrW = xx2 - xx1;
		float intrH = yy2 - yy1;
		if (intrW > 0 && intrH > 0) {
			return intrW * intrH / unio;
		}else
			return 0.0;
	}
	void nms(std::vector<faceBox>& dets, float thres) {
		std::vector<faceBox> dets2;
		while (dets.size() > 0) {
			const faceBox bb1 = dets.front();
			bool keep = true;
			for (int k = 0; k < dets2.size(); k++) {
				if (keep) {
					const faceBox bb2 = dets2[k];
					float overlap = JaccardOverlap(bb1, bb2);
					keep = (overlap <= thres);
				}else
					break;
			}
			if (keep) {
				dets2.push_back(dets.front());
			}
			dets.erase(dets.begin());
		}
		for (int i = 0; i < dets2.size(); i++) {
			dets.push_back(dets2[i]);
		}
	}
	point faceBox::operator[](int i) {
		return point(data[2 * i + 0], data[2 * i + 1]);
	}
#ifndef noStream
	std::istream& operator>>(std::istream& out, faceBox& fb) {
		for (int i = 0; i < 15; i++) {
			out >> fb.data[i];
		}
		return out;
	}
	std::ostream& operator<<(std::ostream& out, faceBox& fb) {
		for (int i = 0; i < 15; i++) {
			out << fb.data[i] << " ";
		}
		return out;
	}
#endif
}


