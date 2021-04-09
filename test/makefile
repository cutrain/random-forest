SRC_PREFIX=../src
SRCOBJ=Forest.o Tree.o quantreg_simplex.o CenQRF-functions.o
SRCOBJ:=$(patsubst %.o,$(SRC_PREFIX)/%.o,$(SRCOBJ))
TESTOBJ=test.o speedtest.o 
OBJ=$(TESTOBJ) $(SRCOBJ)

INCL=-I/usr/share/R/include -I/usr/local/include -I/usr/local/lib/R/site-library/RInside/include -I/usr/local/lib/R/site-library/Rcpp/include -I/usr/local/lib/R/site-library/RcppArmadillo/include 
LIB=-L/usr/lib/R/lib -L/usr/local/lib -L/usr/local/lib/R/site-library/RInside/lib -lprofiler -fopenmp -llapack -lblas -lgfortran -lm -lquadmath -lRInside -lR -lc
DEF=-DDEBUG

all: test

test: $(OBJ)
	g++ -std=gnu++11 -g -o $@ $^ $(LIB)

%.o:%.cpp
	g++ -std=gnu++11 $(DEF) $(INCL) -fopenmp -fpic -g -O2 -fdebug-prefix-map=/build/r-base-jbaK_j/r-base-3.6.3=. -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 -o $@ -c $^


.PHONY:clean clean_test
clean_test:
	rm -rf *.o test

clean:
	rn -rf $(OBJ) test
