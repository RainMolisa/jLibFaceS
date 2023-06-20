#include<iostream>
#include<fstream>
#include"faceDPostProc.h"
int main(int argc, char** argv) {
	std::vector<bx::faceBox> res = fdpp::faceDPostProc();
	for (int i = 0; i < res.size(); i++) {
		res[i].scale(fdpp::_cfg.w, fdpp::_cfg.h);
	}
	for (int i = 0; i < res.size(); i++) {
		std::cout << res[i] << std::endl;
	}
	std::fstream fs("res.txt", std::ios::out);
	for (int i = 0; i < res.size(); i++) {
		fs << res[i] << std::endl;
	}
	fs.close();
	return 0;
}