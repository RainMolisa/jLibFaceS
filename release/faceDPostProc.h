#pragma once

#include<math.h>
#include<algorithm>
#include"NCHW.h"
#include"faceBox.h"
#include"fDPPItf.h"
namespace fdpp {
	typedef std::pair<float, int> scOrder;
	void softMax2(float& v1, float& v2);
	inline float cutOff01(float v);
	bool sortScBboxDesc(const scOrder& pair1, const scOrder& pair2);
	inline float clamp01(float v);

	std::vector<bx::faceBox> faceDPostProc(NCHW::nchw<float>* ts) {
		float v0 = _cfg.varience[0];
		float v1 = _cfg.varience[1];
		float w = _cfg.w;
		float h = _cfg.h;
		std::vector<bx::faceBox> fbs;
		for (int i = 0; i < layNum; i++) {
			for (int y = 0; y < ts[i].h; y++) {
				for (int x = 0; x < ts[i].w; x++) {
					for (int j = 0; j < _minSizes.lys[i]; j++) {
						float minSz = _minSizes(i, j);
						float s_kx = minSz / float(w);
						float s_ky = minSz / float(h);
						float cx = (float(x) + 0.5) * steps[i] / float(w);
						float cy = (float(y) + 0.5) * steps[i] / float(h);
						s_kx = clamp01(s_kx);
						s_ky = clamp01(s_ky);
						cx = clamp01(cx);
						cy = clamp01(cy);
						//
						float cf0 = ts[i](0, j * 17 + 14, y, x);
						float cf1 = ts[i](0, j * 17 + 15, y, x);
						float iou = ts[i](0, j * 17 + 16, y, x);
						iou = cutOff01(iou);
						softMax2(cf0, cf1);
						float score = sqrt(iou * cf1);
						if (score > _cfg.confThres) {
							bx::faceBox fb;
							fb.setScore(score);
							fb.data[0] = (ts[i](0, j * 17 + 0, y, x) * s_kx * v0 + cx);
							fb.data[1] = (ts[i](0, j * 17 + 1, y, x) * s_ky * v0 + cy);
							fb.data[2] = s_kx * exp(ts[i](0, j * 17 + 2, y, x) * v1);
							fb.data[3] = s_ky * exp(ts[i](0, j * 17 + 3, y, x) * v1);
							fb.data[0] = fb.data[0] - fb.data[2] / 2;
							fb.data[1] = fb.data[1] - fb.data[3] / 2;
							fb.data[2] = fb.data[2] + fb.data[0];
							fb.data[3] = fb.data[3] + fb.data[1];
							fb.data[4] = (ts[i](0, j * 17 + 4, y, x) * s_kx * v0 + cx);
							fb.data[5] = (ts[i](0, j * 17 + 5, y, x) * s_ky * v0 + cy);
							fb.data[6] = (ts[i](0, j * 17 + 6, y, x) * s_kx * v0 + cx);
							fb.data[7] = (ts[i](0, j * 17 + 7, y, x) * s_ky * v0 + cy);
							fb.data[8] = (ts[i](0, j * 17 + 8, y, x) * s_kx * v0 + cx);
							fb.data[9] = (ts[i](0, j * 17 + 9, y, x) * s_ky * v0 + cy);
							fb.data[10] = (ts[i](0, j * 17 + 10, y, x) * s_kx * v0 + cx);
							fb.data[11] = (ts[i](0, j * 17 + 11, y, x) * s_ky * v0 + cy);
							fb.data[12] = (ts[i](0, j * 17 + 12, y, x) * s_kx * v0 + cx);
							fb.data[13] = (ts[i](0, j * 17 + 13, y, x) * s_ky * v0 + cy);
							if(fb.islegal())
								fbs.push_back(fb);
						}
					}
				}
			}
		}
		std::vector<scOrder> order;
		order.resize(fbs.size());
		for (int i = 0; i < order.size(); i++) {
			order[i].first = fbs[i].getScore();
			order[i].second = i;
		}
		std::stable_sort(order.begin(), order.end(), sortScBboxDesc);
		while (order.size() > _cfg.topK) {
			order.pop_back();
		}
		std::vector<bx::faceBox> fbs2;
		for (int i = 0; i < order.size(); i++) {
			fbs2.push_back(fbs[order[i].second]);
		}
		bx::nms(fbs2, _cfg.nmsThres);
		return fbs2;
	}

	float cutOff01(float v) {
		float r = v < 0 ? 0 : v;
		r = r > 1 ? 1 : r;
		return r;
	}
	void softMax2(float& v1, float& v2) {
		float a1 = exp(v1);
		float a2 = exp(v2);
		v1 = a1 / (a1 + a2);
		v2 = a2 / (a1 + a2);
	}
	bool sortScBboxDesc(const scOrder& pair1, const scOrder& pair2) {
		return pair1.first > pair2.first;
	}
	float clamp01(float v) {
		float ret = v > 0 ? v : 0;
		ret = ret < 1 ? ret : 1;
		return ret;
	}
}