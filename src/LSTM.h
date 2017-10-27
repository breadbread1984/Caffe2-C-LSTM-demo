#ifndef LSTM_H
#define LSTM_H

#include <iostream>
#include <string>
#include <caffe2/core/init.h>
#include <caffe2/core/predictor.h>
#include <caffe2/utils/proto_utils.h>
#include "PairList.h"

using namespace std;
using namespace caffe2;

class LSTM {
public:
	LSTM(string initNet = "deploy_models/lstm_init_net.pbtxt",string predictNet = "deploy_models/lstm_predict_net.pbtxt");
	virtual ~LSTM();
	string sample(int length);
protected:
	char postProcess(TensorCPU output);
	Workspace workspace;
	unique_ptr<NetBase> predict_net;
	PairList pl;
};

#endif
