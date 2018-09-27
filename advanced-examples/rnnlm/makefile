CC = x86_64-linux-g++-4.5
#CFLAGS = -lm -O2 -g -fomit-frame-pointer
CFLAGS = -lm -O2 -funroll-loops -fprefetch-loop-arrays

ifeq ($(USE_BLAS),1)
BLAS_LIBS = -lcblas -latlas
OPT_DEF = -D USE_BLAS
endif

all: rnnlmlib.o rnnlm

rnnlmlib.o : rnnlmlib.cpp
	$(CC) $(CFLAGS) $(BLAS_LIBS) $(OPT_DEF) -c rnnlmlib.cpp

rnnlm : rnnlm.cpp
	$(CC) $(CFLAGS) $(BLAS_LIBS) $(OPT_DEF) rnnlm.cpp rnnlmlib.o -o rnnlm

clean:
	rm -rf *.o rnnlm
