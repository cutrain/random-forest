all: test

test: test.o quantreg_simplex.o
	g++ -std=gnu++11 -L/usr/lib/R/lib -L/usr/local/lib -L/usr/local/lib/R/site-library/RInside/lib -g -o test test.o quantreg_simplex.o -lprofiler -fopenmp -llapack -lblas -lgfortran -lm -lquadmath -lRInside -lR -lc

test.o: test.cpp
	g++ -std=gnu++11 -I/usr/share/R/include -DDEBUG -fopenmp -I/usr/local/include -I/usr/local/lib/R/site-library/RInside/include -I/usr/local/lib/R/site-library/Rcpp/include -I/usr/local/lib/R/site-library/RcppArmadillo/include -fpic -g -O2 -fdebug-prefix-map=/build/r-base-jbaK_j/r-base-3.6.3=. -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 -o test.o -c test.cpp

quantreg_simplex.o: quantreg_simplex.cpp
	g++ -std=gnu++11 -I/usr/share/R/include -DDEBUG -fopenmp -I/usr/local/include -I/usr/local/lib/R/site-library/RInside/include -I/usr/local/lib/R/site-library/Rcpp/include -I/usr/local/lib/R/site-library/RcppArmadillo/include -fpic -g -O2 -fdebug-prefix-map=/build/r-base-jbaK_j/r-base-3.6.3=. -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 -o quantreg_simplex.o -c quantreg_simplex.cpp
