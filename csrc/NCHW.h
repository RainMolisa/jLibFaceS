#pragma once

//#define noStream
#include<string>
#ifndef noStream
#include<fstream>
#include<iostream>
#endif
#include<iosfwd>
#include<vector>
namespace NCHW {
	template<typename T>
	class nchw {
	public:
		T* data;
		int n;
		int c;
		int h;
		int w;
	public:
		nchw() :n(0), c(0), h(0), w(0) { data = NULL; };
		nchw(int n,int c,int h,int w) {
			data = 0;
			create(n, c, h, w);
		}
#ifndef noStream
		nchw(std::string fPath) {
			read(fPath);
		}
#endif
		nchw(std::vector<T> view,int W) {
			int N = view.size();
			n = 1;
			c = 1;
			h = N / W;
			w = W;
			data = new T[n * c * h * w];
			for (int i = 0; i < N; i++) {
				data[i] = view[i];
			}
		}
		template<typename S>
		nchw(int n, int c, int h, int w, S* p, int frac) {
			create<S>(n, c, h, w, p, frac);
		}
		~nchw(){
			setNULL();
		}
		template<typename S>
		void create(int n, int c, int h, int w, S* p, int frac) {
			T fc = 1 << frac;
			this->create(n, c, h, w);
			int bt = sizeof(S) * 8;
			if (bt <= 8)
				bt = 32;
			int W = w + (bt - w % bt);
			for (int i = 0; i < n * c * h; i++) {
				for (int j = 0; j < w; j++) {
					data[i * w + j] = p[i * W + j] / fc;
				}
			}
		}
#ifndef noStream
		template<typename S>
		void create(int n, int c, int h, int w, int frac, std::string file) {
			int bt = sizeof(S) * 8;
			if (bt <= 8)
				bt = 32;
			int W = w + (bt - w % bt);
			std::fstream fstm(file, std::ios::in);
			int N = n * c * h * W;
			S* buf = new S[N];
			for (int i = 0; i < N; i++) {
				int val;
				fstm>>val;
				buf[i] = val;
			}
			this->create(n, c, h, w, buf, frac);
			delete[] buf;
			fstm.close();
		}
#endif // !noStream
		void create(int n, int c, int h, int w) {
			this->n = n;
			this->c = c;
			this->h = h;
			this->w = w;
			data = new T[n*c*h*w];
		}
		void zeros(int n, int c, int h, int w) {
			this->create(n, c, h, w);
			for (int i = 0; i < n * c * h * w; i++)
				data[i] = 0;
		}
		void ones(int n, int c, int h, int w) {
			this->create(n, c, h, w);
			for (int i = 0; i < n * c * h * w; i++)
				data[i] = 1;
		}
		void setNULL() {
			if (data) {
				delete[] data;
				data = NULL;
			}
			n = c = h = w = 0;
		}
		inline bool isEmpty(){
			return (n <= 0 || c <= 0 || h <= 0 || w<=0 || data == NULL);
		}
		inline T getElement(int i0, int i1, int i2, int i3) {
			if (data) {
				T p = data[i0*n*c*w*h+i1*w*h+i2*w+i3];
				return p;
			}
			return (T)(0);
		}
		inline T& operator()(int i0, int i1, int i2, int i3) {
			return data[i0 * n * c * w * h + i1 * w * h + i2 * w + i3];
		}
		inline T& operator()(int i2, int i3) {
			return data[i2 * w + i3];
		}
		inline T& operator() (int i) {
			return data[i];
		}
		nchw<T> getW(int h) {
			nchw<T> ret(1, 1, 1, w);
			for (int i = 0; i < w; i++) {
				ret(i) = this->getElement(0, 0, h, i);
			}
			return ret;
		}
		void getW(int h,T* p) {
			for (int i = 0; i < w; i++) {
				p[i] = this->getElement(0, 0, h, i);
			}
		}
		void setW(int h, T* p) {
			for (int i = 0; i < w; i++) {
				(*this)(h, i) = p[i];
			}
		}
		void permute0231(nchw<T> &ret) {
			//nchw<T> ret(n,h,w,c);
			ret.setNULL();
			ret.create(n, h, w, c);
			for (int ni = 0; ni < n; ni++) {
				for (int ci = 0; ci < c; ci++) {
					for (int hi = 0; hi < h; hi++) {
						for (int wi = 0; wi < w; wi++) {
							ret(ni, hi, wi, ci) = getElement(ni, ci, hi, wi);
						}
					}
				}
			}
			//return ret;
		}
		std::vector<T> view1() {
			std::vector<T> ret;
			int N = n * h * w * c;
			ret.resize(N);
			for (int i = 0; i < N; i++)
				ret[i] = data[i];
			return ret;
		}
		void viewW(int W,nchw<T> & ret) {
			int N = n * c * h * w;
			int H = N / W;
			//nchw<T> ret(1, 1, H, W);
			ret.setNULL();
			ret.create(1, 1, H, W);
			for (int i = 0; i < N; i++)
				ret.data[i] = data[i];
			//return ret;
		}
		void getWRange(int st,int ed,nchw<T> &ret) {
			//nchw<T> ret(n, c, h, ed - st);
			ret.setNULL();
			ret.create(n, c, h, ed - st);
			for (int ni = 0; ni < n; ni++) {
				for (int ci = 0; ci < c; ci++) {
					for (int hi = 0; hi < h; hi++) {
						for (int wi = st; wi < ed; wi++) {
							ret(ni, ci, hi, wi - st) = this->getElement(ni, ci, hi, wi);
						}
					}
				}
			}
			//return ret;
		}
#ifndef noStream
		void create(std::string fPath) {
			read(fPath);
		}
		void read(std::string fPath) {
			std::fstream fStm(fPath,std::ios::in);
			fStm >> n;
			fStm >> c;
			fStm >> h;
			fStm >> w;
			data = new T[n * c * h * w];
			for (int i = 0; i < n * c * h * w; i++) {
				fStm >> data[i];
			}
			fStm.close();
		}
		void write(std::string fPath) {
			std::fstream fStm(fPath, std::ios::out);
			fStm << n << std::endl;
			fStm << c << std::endl;
			fStm << h << std::endl;
			fStm << w << std::endl;
			for (int i = 0; i < n * c * h * w; i++) {
				fStm << data[i] << std::endl;
			}
			fStm.close();
		}
		void writeWH(std::string fPath) {
			std::fstream fStm(fPath, std::ios::out);
			for (int hi = 0; hi < h; hi++) {
				for (int wi = 0; wi < w; wi++) {
					fStm << (*this)(hi,wi)<< " ";
				}
				fStm << std::endl;
			}
			fStm.close();
		}
#endif
		void copyTo(nchw<T>& v) {
			v.setNULL();
			v.create(n, c, h, w);
			for (int i = 0; i < n * c * h * w; i++) {
				v.data[i] = data[i];
			}
		}
		//void operator=(nchw&& v) {
		//	//v.setNULL();
		//	//return *this;
		//}
		void multi(nchw<T>& v) {
			int N = n * c * h * w;
			for (int i = 0; i < N; i++) {
				data[i] = data[i] * v.data[i];
			}
		}
		void add(nchw<T>& v) {
			int N = n * c * h * w;
			for (int i = 0; i < N; i++) {
				data[i] = data[i] + v.data[i];
			}
		}
#ifndef noStream
		friend std::ostream& operator<<(std::ostream& out, nchw& db) {
			for (int ni = 0; ni < db.n; ni++) {
				out << "[";
				for (int ci = 0; ci < db.c; ci++) {
					out << "[";
					for (int hi = 0; hi < db.h; hi++) {
						out << "[";
						for (int wi = 0; wi < db.w; wi++) {
							out << db(ni, ci, hi, wi) << " ";
						}
						out << "]" << std::endl<<"  ";
					}
					out << "]"<<std::endl<<" ";
				}
				out << "]" << std::endl;
			}
			return out;
		}
#endif
	};
	//
	template<typename A,typename B>
	void whMul1(nchw<A> &a,B* b) {
		for (int i = 0; i < a.h; i++) {
			for (int j = 0; j < a.w; j++) {
				a(i, j) = a(i, j) * b[j];
			}
		}
	}
};


