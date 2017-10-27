#include <cstdlib>
#include <iostream>
#include <string>
#include <boost/program_options.hpp>
#include "LSTM.h"

using namespace std;
using namespace boost::program_options;

int main(int argc,char ** argv)
{
	string initNet, predictNet;
	int length;
	options_description desc;
	desc.add_options()
		("help,h","打印当前使用方法")
		("init,i",value<string>(&initNet)->default_value("deploy_models/lstm_init_net.pbtxt"),"初始化模型")
		("predict,p",value<string>(&predictNet)->default_value("deploy_models/lstm_predict_net.pbtxt"),"预测模型")
		("length,l",value<int>(&length)->default_value(10),"采样序列的长度");
	variables_map vm;
	store(parse_command_line(argc,argv,desc),vm);
	notify(vm);
	
	if(1 == vm.count("help")) {
		cout<<desc;
		return EXIT_SUCCESS;
	}
	
	LSTM lstm(initNet, predictNet);
	string result = lstm.sample(length);
	cout<<result<<endl;
	
	return EXIT_SUCCESS;
}
