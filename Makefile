CAFFE2_PREFIX=/home/xieyi/opt/caffe2
CXXFLAGS = `pkg-config --cflags eigen3` -Isrc -I${CAFFE2_PREFIX}/include -std=c++14 -g2
LIBS=`pkg-config --libs eigen3` -lboost_program_options -lboost_filesystem -lboost_system -lboost_serialization -L${CAFFE2_PREFIX}/lib -lglog -lprotobuf
OBJS=$(patsubst %.cpp,%.o,$(wildcard src/*.cpp))

all: generate_trainset predictor

dataset: generate_trainset
	./generate_trainset -i res/shakespeare.txt -o ${@} -b 10

.PHONY: train test

train:
	run_plan --plan train_plan.pbtxt

test:
	run_plan --plan test_plan.pbtxt

generate_trainset: src/generate_trainset.o
	$(CXX) $^ $(LIBS) -lcaffe2 -o ${@}

predictor: src/LSTM.o src/main.o
	$(CXX) $^ $(LIBS) -lcaffe2 -lcaffe2_gpu -lcurand -lcudart -o ${@}
	
clean:
	$(RM) predictor generate_trainset index.dat $(OBJS)
	$(RM) -r dataset LSTM_params
