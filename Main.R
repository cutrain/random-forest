library(rbenchmark)
library(quantreg)
library(MASS)
rm(list = ls())
# set working directory as current active document
#setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

set.seed(123)
N = 1000
d = 2
X = matrix(runif(N*d,0,1),N,d)  # design matrix
Y = cbind(1,X)%*%c(1,2,3)+rnorm(N) # response variable
weights = runif(N,0,1) #weights with the same length as Y


dyn.load('lib/qr_simplex.so')
CenQRForest_C <- function(matZ0, matX0, matY0, delta0, tau, weight_rf0, rfsrc_time_surv0, time_interest, quantile_level, numTree, minSplit1, maxNode, mtry) {
    .Call("_qr_simplex_CenQRForest_C", matZ0, matX0, matY0, delta0, tau, weight_rf0, rfsrc_time_surv0, time_interest, quantile_level, numTree, minSplit1, maxNode, mtry)
}
qr_tau_para_diff_cpp <- function(x, y, weights, taurange, tau_min = 1e-10, tol = 1e-14, maxit = 100000L, max_num_tau = 1000L, use_residual = TRUE) {
    .Call("_qr_simplex_qr_tau_para_diff_cpp", x, y, weights, taurange, tau_min, tol, maxit, max_num_tau, use_residual)
}

print("start")
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
quit()
source("qr_tau_para_diff.R")
source("NewRanks.R")



#######################################################
##########Quantile Regression without Weights##########
#######################################################

###quantile regression with R package written by fortran###
qr_fortran = rq(Y~X,tau = -1)
qr_fortran_transfer = transfer_v(qr_fortran, diff_dsol = FALSE) # diff_dsol = F tells the function this is obtained from quantreg
# with in-build ranks function
rank0 = ranks(qr_fortran, score = "wilcoxon")
# with user-written ranks function
rank1 = NewRanks(qr_fortran_transfer, score = "wilcoxon")


###quantile regression written by R###
qr_tau_R = qr_tau_para_diff(X,Y,taurange = c(0,1),maxit = 10^5,
                            max_num_tau = 2*10^3)
qr_R_transfer = transfer_v(qr_tau_R, diff_dsol = TRUE) # This is what you need diff_dsol = T tells the function this is obtained from Tianchen's code
rank2 = NewRanks(qr_R_transfer, score = "wilcoxon")


###quantile regression written by cpp###
qr_tau_cpp = qr_tau_para_diff_cpp(X,Y,weights = rep(1,length(Y)),
                                  taurange = c(0,1),maxit = 10^5,
                                  max_num_tau = 2*10^3,
                                  tau_min = 1e-10)

null.fit3 = list()
null.fit3$taus = qr_tau_cpp$tau
null.fit3$J = length(qr_tau_cpp$tau)
null.fit3$diff_dsol = qr_tau_cpp$diff_dual_sol
rank3 = NewRanks(null.fit3, score = "wilcoxon")


###compare ranks based on different coding
plot(rank0$ranks,rank1$ranks)
plot(rank0$ranks,rank2$ranks)
plot(rank0$ranks,rank3$ranks)
plot(rank2$ranks,rank3$ranks)





#######################################################
##########Quantile Regression without Weights##########
#######################################################

###quantile regression with R package written by fortran###
qr_fortran_weight = rq(Y~X, tau = -1,
                       weights = weights/mean(weights))
qr_fortran_transfer = transfer_v(qr_fortran_weight, diff_dsol = FALSE) # diff_dsol = F tells the function this is obtained from quantreg
# with in-build ranks function
rank0 = ranks(qr_fortran_weight, score = "wilcoxon")
# with user-written ranks function
rank1 = NewRanks(qr_fortran_transfer, score = "wilcoxon")



qr_tau_R_weight = qr_tau_para_diff_weight(X,Y,weights = weights,
                                          taurange = c(0,1),
                                          maxit = 10^5,
                                          max_num_tau = 2*10^3,
                                          tau_min = 1e-10)
qr_R_weight_transfer = transfer_v(qr_tau_R_weight,
                                  diff_dsol = TRUE) # This is what you need diff_dsol = T tells the function this is obtained from Tianchen's code
rank2 = NewRanks(qr_R_weight_transfer, score = "wilcoxon")

qr_tau_cpp = qr_tau_para_diff_cpp(X,Y,weights = weights/mean(weights),
                                  taurange = c(0,1),maxit = 10^5,
                                  max_num_tau = 2*10^3,
                                  tau_min = 1e-10)
null.fit3 = list()
null.fit3$taus = qr_tau_cpp$tau[-1]
null.fit3$J = length(qr_tau_cpp$tau)-1
null.fit3$diff_dsol = qr_tau_cpp$diff_dual_sol[,-1]
rank3 = NewRanks(null.fit3, score = "wilcoxon")

plot(rank0$ranks,rank1$ranks)
plot(rank0$ranks,rank2$ranks)
plot(rank0$ranks,rank3$ranks)
plot(rank2$ranks,rank3$ranks)



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

 },replications = 100)


