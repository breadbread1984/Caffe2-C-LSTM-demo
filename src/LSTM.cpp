#include <algorithm>
#include <fstream>
#include <boost/archive/text_iarchive.hpp>
#include <caffe2/core/context_gpu.h>
#include "LSTM.h"

using namespace std;
using namespace boost::archive;

LSTM::LSTM(string initNet,string predictNet)
:workspace(nullptr)
{
#ifndef NDEBUG
	cout<<"init: "<<initNet<<endl<<"preict: "<<predictNet<<endl;
#endif
	DeviceOption option;
	option.set_device_type(CUDA);
	new CUDAContext(option);
	//载入模型
	NetDef init_net_def, predict_net_def;
	CAFFE_ENFORCE(ReadProtoFromFile(initNet, &init_net_def));
	CAFFE_ENFORCE(ReadProtoFromFile(predictNet, &predict_net_def));
	init_net_def.mutable_device_option()->set_device_type(CUDA);
	predict_net_def.mutable_device_option()->set_device_type(CUDA);
	//网络初始化
	workspace.RunNetOnce(init_net_def);
	//创建判别器
	predict_net = CreateNet(predict_net_def,&workspace);
	//载入字符编码
	std::ifstream in("index.dat");
	if(false == in.is_open()) throw runtime_error("字符编码文件不存在！");
	text_iarchive ia(in);
	ia >> pl;
}

LSTM::~LSTM()
{
}

string LSTM::sample(int length)
{
	string retVal;
	for(int i = 0 ; i < length ; i++) {
		//运行神经网
		predict_net->Run();
		TensorCPU tensor = TensorCPU(workspace.GetBlob("softmax")->Get<TensorCUDA>());
		retVal += postProcess(tensor);
	}
	return retVal;
}

char LSTM::postProcess(TensorCPU output)
{
	const float * probs = output.data<float>();
	vector<TIndex> dims = output.dims();
#ifndef NDEBUG
	assert(3 == dims.size());
	assert(1 == dims[0]);	//seq_length
	assert(1 == dims[1]);	//batch
	assert(pl.size() == dims[2]);	//D
#endif
	vector<float> data(pl.size());
	copy(probs,probs+pl.size(),data.begin());
	vector<float>::iterator which = max_element(data.begin(),data.end());
	return pl.get<0>().find(which-data.begin())->c;
}
