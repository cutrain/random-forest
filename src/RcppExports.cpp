// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// CenQRForest_C
SEXP CenQRForest_C(const arma::mat& matZ0, const arma::mat& matX0, const arma::vec& matY0, const arma::vec& delta0, const double& tau, const arma::vec& weight_rf0, const arma::mat& rfsrc_time_surv0, const arma::vec& time_interest, const arma::vec& quantile_level, int numTree, int minSplit1, int maxNode, int mtry);
RcppExport SEXP _CenQRF_CenQRForest_C(SEXP matZ0SEXP, SEXP matX0SEXP, SEXP matY0SEXP, SEXP delta0SEXP, SEXP tauSEXP, SEXP weight_rf0SEXP, SEXP rfsrc_time_surv0SEXP, SEXP time_interestSEXP, SEXP quantile_levelSEXP, SEXP numTreeSEXP, SEXP minSplit1SEXP, SEXP maxNodeSEXP, SEXP mtrySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type matZ0(matZ0SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type matX0(matX0SEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type matY0(matY0SEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type delta0(delta0SEXP);
    Rcpp::traits::input_parameter< const double& >::type tau(tauSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type weight_rf0(weight_rf0SEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type rfsrc_time_surv0(rfsrc_time_surv0SEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type time_interest(time_interestSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type quantile_level(quantile_levelSEXP);
    Rcpp::traits::input_parameter< int >::type numTree(numTreeSEXP);
    Rcpp::traits::input_parameter< int >::type minSplit1(minSplit1SEXP);
    Rcpp::traits::input_parameter< int >::type maxNode(maxNodeSEXP);
    Rcpp::traits::input_parameter< int >::type mtry(mtrySEXP);
    rcpp_result_gen = Rcpp::wrap(CenQRForest_C(matZ0, matX0, matY0, delta0, tau, weight_rf0, rfsrc_time_surv0, time_interest, quantile_level, numTree, minSplit1, maxNode, mtry));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_CenQRF_CenQRForest_C", (DL_FUNC) &_CenQRF_CenQRForest_C, 13},
    {NULL, NULL, 0}
};

RcppExport void R_init_CenQRF(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
