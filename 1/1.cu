#include <iostream>
#include <math.h>
#include <algorithm>
#include <map>
#include <random>
#include <time.h>
#include <cuda_runtime.h>

using namespace std;

__host__ __device__
unsigned hash_func(unsigned key, int hash_num, unsigned tablesize){
	int c2=0x27d4eb2d;
	switch (hash_num){
		case 0:
  			key = (key+0x7ed55d16) + (key<<12);
  			key = (key^0xc761c23c) ^ (key>>19);
  			key = (key+0x165667b1) + (key<<5);
  			key = (key+0xd3a2646c) ^ (key<<9);
  			key = (key+0xfd7046c5) + (key<<3);
  			key = (key^0xb55a4f09) ^ (key>>16);
  			return key%tablesize;
		case 1:
  			key = (key^61)^(key>>16);
  			key = key+(key<<3);
  			key = key^(key>>4);
  			key = key*c2;
  			key = key^(key>>15);
  			return key%tablesize;
		case 2:
			return ((66*key+32)%537539573)%tablesize;
		default:
			//printf("wrong hash_num\n");
			return 0;
	}
}

__device__
void secondtableInsertion(unsigned key, unsigned* secondtable){
	unsigned secondtablesize = pow(2,18);
	unsigned location = ((33*key+87)%116743349)&(secondtablesize-1);
	for(unsigned i = 0; i < 200; ++i) {
		key = atomicExch(&secondtable[location],key);
		if(key!=NULL){
			location++;
			if(location == secondtablesize-1){
				location = 0;
			}
			continue;
		}
		return;
	}
	printf("Failed.\n");
	return;
}

__global__ void lookupHash(unsigned* keys, unsigned* table,unsigned keysize,unsigned tablesize, unsigned* secondtable,int hash_num){
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index>keysize){
		return;
	}
	unsigned key = keys[index];
	
	unsigned location[3];
	for(unsigned j = 0; j < hash_num; ++j) {
		location[j] = hash_func(key,j,tablesize);
	}
	for(unsigned i = 0; i < hash_num; ++i) {
        if(atomicCAS(&table[location[i]], key, key) == key){
            return;
        }
	}
	unsigned secondtablesize = pow(2,18);
	unsigned location1 = ((33*key+87)%116743349)&(secondtablesize-1);
	unsigned key1;
	
	for(unsigned i = 0; i < 200; ++i) {
		key1 = atomicCAS(&table[location1],key,key);
        if(key1 == key || key1 == NULL){
            return;
        }
		location1++;
		if(location1 == secondtablesize-1){
			location1 = 0;
		}
	}

	return;
}

__global__ void cuckooHash(unsigned* cuda_tables, unsigned* cuda_keys, unsigned keysize, int M,int hash_num, unsigned tablesize, unsigned* secondtable){
	int index = blockDim.x*blockIdx.x + threadIdx.x;
	if(index >= keysize) {
		return;
	}
	unsigned key = cuda_keys[index];
	unsigned location[3];
	location[0] = hash_func(key,0,tablesize);
	for(unsigned i = 0; i <= hash_num*M; ++i) {
		if(i==hash_num*M) {
			secondtableInsertion(key,secondtable);
		}
		key = atomicExch(&cuda_tables[location[i%hash_num]],key);
		if(key==NULL) {
			return;
		}
		for(unsigned j = 0; j < hash_num; ++j) {
			location[j] = hash_func(key,j,tablesize);
		}
	}
}

int main() {
	for(unsigned t = 0; t < 5; ++t) {
		for(unsigned s = 10; s < 25; ++s) {
			int hash_num = 3;
			unsigned tablesize = pow(2,25);
			unsigned secondtablesize = pow(2,18);
			unsigned *tables = (unsigned *)malloc(tablesize*sizeof(unsigned));
			unsigned *secondtable = (unsigned *)malloc(secondtablesize*sizeof(unsigned));
			unsigned keysize = pow(2,s);
			unsigned *keys = (unsigned *)malloc(keysize*sizeof(unsigned));
			for(unsigned i = 0; i < tablesize; ++i) {
				tables[i] = 0;
			}
			for(unsigned i = 0; i < secondtablesize; ++i) {
				secondtable[i] = 0;
			}
			std::map<unsigned ,bool> randommap;
			std::random_device rd;
			std::mt19937 gen(rd());
			std::uniform_int_distribution<unsigned> dis(1, pow(2,32)-1);
	    	for(unsigned i = 0; i < keysize; ++i) {
	    		unsigned rand = dis(gen);
	    		while(randommap.find(rand) != randommap.end()) {
	    			rand = dis(gen);
	    		}
	    		randommap[rand] = true;
	    		keys[i] = rand; 
	    	}
	    	unsigned* cuda_keys;
	    	unsigned* cuda_tables;
	    	unsigned* cuda_secondtable;
	    	int blockSize;
	    	int minGridSize;
	    	int gridSize;
		
	    	cudaDeviceReset();
	    	cudaMalloc(&cuda_tables, tablesize*sizeof(unsigned));
	    	cudaMalloc(&cuda_keys, keysize*sizeof(unsigned));
	    	cudaMalloc(&cuda_secondtable, secondtablesize*sizeof(unsigned));
	    	cudaMemcpy(cuda_tables, tables, tablesize, cudaMemcpyHostToDevice);
	    	cudaMemcpy(cuda_keys, keys, keysize, cudaMemcpyHostToDevice);
	    	cudaMemcpy(cuda_secondtable, secondtable, secondtablesize, cudaMemcpyHostToDevice);
	    	cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, cuckooHash, 0, 1000000);
	    	gridSize = (keysize + blockSize - 1) / blockSize; 
	    	int M = (int)4*ceil(log2((double)keysize));
		
			cudaEvent_t start, stop;
			cudaEventCreate(&start);    
			cudaEventCreate(&stop);  
			cudaEventRecord(start, 0);
	 	
			cuckooHash<<<gridSize,blockSize>>>(cuda_tables, cuda_keys, keysize, M,hash_num,tablesize,cuda_secondtable);
		
			cudaEventRecord(stop, 0); 
			cudaEventSynchronize(stop);
			float kernelTime;
			cudaEventElapsedTime(&kernelTime, start, stop);
			cudaEventDestroy(start);
			cudaEventDestroy(stop);
			printf("s = %d,time: %.2f ms\n",s,kernelTime);
		}
	}
	return 0;
}