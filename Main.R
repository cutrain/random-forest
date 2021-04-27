library(rbenchmark)
library(quantreg)
library(MASS)

dyn.load('lib/rf.so')
CenQRForest_C <- function(matZ0, matX0, matY0, delta0, tau, weight_rf0, rfsrc_time_surv0, time_interest, quantile_level, numTree, minSplit1, maxNode, mtry) {
    .Call("_rf_CenQRForest_C", matZ0, matX0, matY0, delta0, tau, weight_rf0, rfsrc_time_surv0, time_interest, quantile_level, numTree, minSplit1, maxNode, mtry)
}
qr_tau_para_diff_cpp <- function(x, y, weights, taurange, tau_min = 1e-10, tol = 1e-14, maxit = 100000L, max_num_tau = 1000L, use_residual = TRUE) {
    .Call("_rf_qr_tau_para_diff_cpp", x, y, weights, taurange, tau_min, tol, maxit, max_num_tau, use_residual)
}
QPRForest_C <- function(matZ0, matX0, matY0, matXnew, taurange, quantile_level, max_num_tau, numTree, minSplit1, maxNode, mtry) {
    .Call("_rf_QPRForest_C", matZ0, matX0, matY0, matXnew, taurange, quantile_level, max_num_tau, numTree, minSplit1, maxNode, mtry)
}
print("start")

load("Example/exampledataQPRF.RData")

Z = as.matrix(train[,control.z])
Y = as.matrix(train[,namey])
X = as.matrix(train[,namex])

N_test = round(200^(1/2))^2
X_test = matrix(c(rep(seq(0.05,0.95,length.out = round(200^(1/2))),
                      each = round(200^(1/2))),
                  rep(seq(0.05,0.95,length.out = round(200^(1/2))),
                      times = round(200^(1/2))),
                  runif((d-2)*N_test,0,1)),ncol = d)

set.seed(taskid)
QPR_forest = .Call('_rf_QPRForest_C', PACKAGE = 'rf',
                      Z,X,Y,X_test,taurange = c(0,1),
                      quantile_level = as.matrix(seq(0,1,length.out = 50)[2:49]),
                      max_num_tau = 2*10^3,numTree = 500,
                      minSplit1 = 8,maxNode = 500,
                      mtry = min(ceiling(sqrt(length(namex)) + 20),
                                 length(namex)-1))
quit()

# tset CenQRForest
load("Example/tempdata.RData")

data_weight_MM = train
set.seed(234)
source("R/Forest.R")
CQRforest(matY = as.matrix(data_weight_MM[,namey]),
          delta = as.matrix(data_weight_MM[,status]),
          matZ = as.matrix(cbind(1,data_weight_MM[,control.z])),
          matX = as.matrix(data_weight_MM[,namex]),
          control = list(numTree = 1))

quit()

# test simplex
set.seed(123)
N = 1000
d = 2
X = matrix(runif(N*d,0,1),N,d)  # design matrix
Y = cbind(1,X)%*%c(1,2,3)+rnorm(N) # response variable
weights = runif(N,0,1) #weights with the same length as Y

benchmark("cpp" = {
   qr_tau_cpp = qr_tau_para_diff_cpp(X,Y,weights = rep(1,length(Y)),
                                     taurange = c(0,1),maxit = 10^5,
                                     max_num_tau = 2*10^3)
 },
# "R" = {
#   qr_tau_R = qr_tau_para_diff(X,Y,taurange = c(0,1),maxit = 10^5,
#                               max_num_tau = 2*10^3)
# },
 "fortran" = {
   qr_fortran = rq(Y~X,tau = -1)
 },replications = 10)
