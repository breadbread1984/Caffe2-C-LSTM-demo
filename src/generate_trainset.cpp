#include <cstdlib>
#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <set>
#include <vector>
#include <boost/program_options.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/filesystem.hpp>
#include <caffe2/core/common.h>
#include <caffe2/core/db.h>
#include <caffe2/core/init.h>
#include <caffe2/proto/caffe2.pb.h>
#include "PairList.h"

using namespace std;
using namespace boost;
using namespace boost::program_options;
using namespace boost::filesystem;
using namespace caffe2;

int main(int argc,char ** argv)
{
	string textpath,outputpath;
	int batch_size,seq_length;
	options_description desc;
	desc.add_options()
		("help,h","打印当前使用方法")
		("input,i",value<string>(&textpath),"作为训练集的文本")
		("output,o",value<string>(&outputpath),"输出lmdb的路径")
		("batch,b",value<int>(&batch_size)->default_value(1),"并发训练的样本个数")
		("seq_length,s",value<int>(&seq_length)->default_value(25),"样本的长度");
	variables_map vm;
	store(parse_command_line(argc,argv,desc),vm);
	notify(vm);
	
	if(1 == argc || 1 == vm.count("help") || 1 != vm.count("input") || 1 != vm.count("output")) {
		cout<<desc;
		return EXIT_SUCCESS;
	}
	
	remove_all(outputpath);
	
	//读入文本，计算文本长度
	std::ifstream in(textpath);
	if(false == in.is_open()) {
		cout<<"failed to open the text file"<<endl;
		return EXIT_FAILURE;
	}
	stringstream buffer;
	buffer << in.rdbuf();
	string text = buffer.str();
	long N = text.size();
	//计算按照batch划分text，记录每个划分起始位置和长度
	long text_block_size = N / batch_size;
	vector<boost::tuple<int,int,int> > parts;
	for(int i = 0 ; i < N && parts.size() < batch_size ; i += text_block_size)
		parts.push_back(
			boost::make_tuple(
				i,		//起始位置
				(parts.size() != batch_size - 1)?text_block_size:(text_block_size + N % batch_size),		//长度
				0		//offset是样本从哪里开始获取
			)
		);
	assert(parts.size() == batch_size);
	//将训练文本里面出现的所有文字进行编码
	set<char> vs(text.begin(),text.end());
	int D = vs.size();
	PairList pl;
	int index = 0;
	for(set<char>::iterator it = vs.begin() ; it != vs.end() ; it++)
		pl.insert(Pair(index++,*it));
	std::ofstream out("index.dat");
	text_oarchive oa(out);
	oa<<pl;
	//填充训练集
	unique_ptr<db::DB> rnndb(db::CreateDB("lmdb",outputpath,db::NEW));
	unique_ptr<db::Transaction> transaction(rnndb->NewTransaction());
	TensorProtos protos;
	//设置训练数据维度
	TensorProto * data = protos.add_protos();
	data->set_data_type(TensorProto::FLOAT);
	data->add_dims(seq_length);
	data->add_dims(D);
	//设置监督值维度
	TensorProto * label = protos.add_protos();
	label->set_data_type(TensorProto::INT32);
	label->add_dims(seq_length);
	//因为lstm输入要求seq_length x batchsize x dimension
	//但是写入lmdb的格式是seq_length x dimension
	//所以在模型中读出来是batchsize x seq_length x dimension
	//需要用caffe2的transpose操作符转换下数据的存储顺序
	string value;
	int count = 0;
	bool flag;
	do {
		for(int b = 0 ; b < batch_size ; b++) {
			//每次重新创建的原因是为了清空数据
			vector<float> input(seq_length * D);
			vector<int> output(seq_length);
			data->clear_float_data();
			label->clear_int32_data();
			assert(0 == data->float_data_size());
			assert(0 == label->int32_data_size());
			for(int s = 0 ; s < seq_length ; s++) {
				int pos = get<0>(parts[b]) + get<2>(parts[b]);
				input[s * D + pl.get<1>().find(text[pos])->i] = 1;
				output[s] = pl.get<1>().find(text[(pos + 1) % N])->i;
				//offset向后移动一个位置
				get<2>(parts[b]) = (get<2>(parts[b]) + 1) % get<1>(parts[b]);
			}
			for(int i = 0 ; i < seq_length * D ; i++) data->add_float_data(input[i]);
			for(int i = 0 ; i < seq_length ; i++) label->add_int32_data(output[i]);
			protos.SerializeToString(&value);
			stringstream sstr;
			sstr<<setw(8)<<setfill('0')<<count;
			transaction->Put(sstr.str(),value);
			if(++count % 1000 == 0) {
				transaction->Commit();
			}
		}//end for
		flag = true;
		for(int b = 0 ; b < batch_size ; b++)
			if(0 == get<2>(parts[b])) {
				//如果offset已经回到开头，再提取样本就与之前的重复，
				//所以停止当前batch提取工作
				flag = false;
				break;
			}
	} while(flag);
	if(count) transaction->Commit();
	
	cout<<"字符"<<pl.size()<<"个"<<endl;
	cout<<"产生"<<count<<"个样本"<<endl;
	
	return EXIT_SUCCESS;
}
