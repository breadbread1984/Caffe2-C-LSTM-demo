#ifndef PAIRLIST_H
#define PAIRLIST_H

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/identity.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/composite_key.hpp>
#include <boost/archive/text_oarchive.hpp>

using namespace boost::multi_index;
using namespace boost::archive;

struct Pair {
	int i;
	char c;
	Pair() {}
	Pair(int ii,char cc):i(ii),c(cc){}
};

typedef multi_index_container<
	Pair,
	indexed_by<
		ordered_unique<
			member<Pair,int,&Pair::i>
		>,
		ordered_unique<
			member<Pair,char,&Pair::c>
		>
	>
> PairList;

namespace boost {
	namespace serialization {
		template<class Archive> void serialize(Archive & ar, Pair & p, const unsigned int version) {
			ar & p.i & p.c;
		}
	}
}

#endif
