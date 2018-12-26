/*********************************************************************
 Created by thiagols on 27.11.18
 Copyright (C) 2018 Thiago Lima Silva<thiagolims@gmail.com>
 Modified 2018-2019 Mathias Bellout <mathias.bellout@ntnu.no>

 This file is part of the FieldOpt project.

 FieldOpt is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published
 by the Free Software Foundation, either version 3 of the License,
 or (at your option) any later version.

 FieldOpt is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with FieldOpt. If not, see <http://www.gnu.org/licenses/>.
*********************************************************************/

#include "TrustRegionModel.h"
#include <Settings/optimizer.h>

#include "Utilities/printer.hpp"
#include "Utilities/stringhelpers.hpp"
#include <Utilities/verbosity.h>
#include <Eigen/Dense>

#include <numeric>
#include <functional>

// ___________________________________________________________________
using namespace Eigen;

namespace Optimization {
namespace Optimizers {

bool compare(
        const int &lhs,
        const int &rhs,
        const double *distances) {
    return distances[lhs] <= distances[rhs];
}

TrustRegionModel::TrustRegionModel() {
}

TrustRegionModel::TrustRegionModel(
        const Matrix<double,Dynamic,Dynamic>& initial_points,
        const RowVectorXd& initial_fvalues,
        VectorXd& lb,
        VectorXd& ub,
        Settings::Optimizer *settings) {


    lb_ = lb;
    ub_ = ub;

    settings_ = settings;
    radius_ =  settings_->parameters().tr_initial_radius;
    tr_center_ = 0;
    dim_ = initial_points.rows();
    cache_max_ = 3*pow(dim_,2);

    points_abs_.setZero(initial_points.rows(), initial_points.cols());
    points_abs_ << initial_points;

    fvalues_.setZero(initial_fvalues.size());
    fvalues_ << initial_fvalues;

    pivot_values_.resize(0);
    cached_fvalues_.resize(0);

    index_vector_.resize(0);
    distances_.resize(0);

    cached_points_.resize(0,0);
    points_shifted_.resize(0,0);

    rebuildModel();
    moveToBestPoint();
    computePolynomialModels();

    if (points_abs_.cols() < 2) {
        ensureImprovement();
    }
}

int TrustRegionModel::getDimension() {
    return dim_;
}

double TrustRegionModel::getRadius() {
    return radius_;
}

Matrix<double,Dynamic,Dynamic> TrustRegionModel::getPoints() {
    return points_abs_;
}

Matrix<double,Dynamic,Dynamic> TrustRegionModel::getPointsShifted() {
    return points_shifted_;
}

RowVectorXd TrustRegionModel::getFunctionValues() {
    return fvalues_;
}

std::vector<Polynomial> TrustRegionModel::getPivotPolynomials() {
    return pivot_polynomials_;
}

void TrustRegionModel::moveToBestPoint() {
    auto best_i = findBestPoint();
    if (best_i != tr_center_) {
        tr_center_ = best_i;
   }

// Here should rebuild polynomials!!!
}

void TrustRegionModel::criticalityStep() {
    //TODO: implement this method
}

double TrustRegionModel::checkInterpolation() {
    //TODO: implement this method
}

bool TrustRegionModel::rebuildModel() {
    double pivot_threshold = settings_->parameters().tr_pivot_threshold * std::fmin(1, radius_);

    //!<All points we know>
    all_points_.resize(points_abs_.rows(), points_abs_.cols() + cached_points_.cols());
    if (cached_points_.size() == 0) {
        all_points_ = points_abs_;
    } else {
        all_points_ << points_abs_;
        all_points_ << cached_points_;
    }

    //!<All function values we know>
    all_fvalues_.resize(fvalues_.size() + cached_fvalues_.size());

    if (cached_fvalues_.size() == 0) {
        all_fvalues_ = fvalues_;
    } else {
        all_fvalues_ << fvalues_;
        all_fvalues_ << cached_fvalues_;
    }

    int dim = all_points_.rows();
    int n_points = all_points_.cols();
    if (tr_center_ != 0) {
        //!<Center will be first>
        all_points_.col(0).swap(all_points_.col(tr_center_));
        auto center_fvalue = all_points_(tr_center_);
        all_fvalues_(tr_center_) = all_fvalues_(0);
        all_fvalues_(0) = center_fvalue;
    }
    //!<Calculate distances from tr center>
    points_shifted_.resize(dim, n_points);
    distances_.resize(n_points);
    points_shifted_.setZero(dim, n_points);
    distances_.setZero(n_points);

    //!<Shift all points to TR center>
    for (int i = 1; i < n_points; i++) {
        //!<Compute distances>
        points_shifted_.col(i) << all_points_.col(i) - all_points_.col(0);
        //<!distances in infinity or 2-norm>
        distances_(i) = points_shifted_.col(i).lpNorm<Infinity>();
    }

    //!<Reorder points based on their distances to the tr center>
    index_vector_.setLinSpaced(distances_.size(), 0, distances_.size() - 1);
    std::sort(index_vector_.data(), index_vector_.data() + index_vector_.size(),
              std::bind(compare, std::placeholders::_1, std::placeholders::_2, distances_.data()));
    sortVectorByIndex(distances_, index_vector_);
    sortVectorByIndex(all_fvalues_, index_vector_);

    sortMatrixByIndex(points_shifted_, index_vector_);
    sortMatrixByIndex(all_points_, index_vector_);

    nfpBasis(dim);//!<build nfp polynomial basis>

    int polynomials_num = pivot_polynomials_.size();
    pivot_values_.resize(polynomials_num);
    pivot_values_.setZero(polynomials_num);

    //!<Constant term>
    int last_pt_included = 0;
    int poly_i = 1;
    pivot_values_(0) = 1;

    //!<Gaussian elimination (using previous points)>
    for (int iter = 1; iter < polynomials_num; iter++) {

        pivot_polynomials_[poly_i] = orthogonalizeToOtherPolynomials(poly_i, last_pt_included);

        double max_layer;
        double farthest_point = distances_(distances_.size() - 1);
        double distance_farthest_point = (double) (farthest_point / radius_);

        int block_beginning;
        int block_end;

        if (poly_i <= dim) {
            block_beginning = 1;
            block_end = dim;
            //!<linear block -- we allow more points (*2)

            max_layer = std::min(2 * settings_->parameters().tr_radius_factor, distance_farthest_point);
            if (iter > dim) {
                //!< We already tested all linear terms.
                //!< We do not have points to build a FL model.
                //!< How did this happen??? see Comment [1]>
                break;

            }
        } else { //!<Quadratic block -- being more carefull>
            max_layer = min(settings_->parameters().tr_radius_factor, distance_farthest_point);
            block_beginning = dim + 1;
            block_end = polynomials_num - 1;
        }

        max_layer = std::fmax(1, max_layer);

        VectorXd all_layers;
        all_layers.setLinSpaced(ceil(max_layer), 1, max_layer);

        double max_absval = 0;
        double pt_max = 0;
        for (int i = 0; i < all_layers.size(); i++) {
            auto layer = all_layers(i);
            double dist_max = layer * radius_;
            for (int n = last_pt_included + 1; n < n_points; n++) {
                if (distances_(n) > dist_max) {
                    break; //!<for n>
                }

                auto val = evaluatePolynomial(pivot_polynomials_[poly_i], points_shifted_.col(n));
                val = val / dist_max; //!<minor adjustment>
                if (abs(max_absval) < abs(val)) {
                    max_absval = val;
                    pt_max = n;
                }
                if (abs(max_absval) > pivot_threshold) {
                    break; //!<for(layer)>
                }
            }
        }

        if (abs(max_absval) > pivot_threshold) {
            //!<Points accepted>
            int pt_next = last_pt_included + 1;
            if (pt_next != pt_max) {
                points_shifted_.col(pt_next).swap(points_shifted_.col(pt_max));
                all_points_.col(pt_next).swap(all_points_.col(pt_max));
                std::swap(all_fvalues_[pt_next], all_fvalues_[pt_max]);
                std::swap(distances_[pt_next], distances_[pt_next]);
            }

            pivot_values_(pt_next) = max_absval;

            //!<Normalize polynomial value>
            pivot_polynomials_[poly_i] = normalizePolynomial(poly_i, pt_next);

            //!<Re-orthogonalize (just to make sure it still assumes 0 in previous points).
            //!< Unnecessary in infinite precision>
            pivot_polynomials_[poly_i] = orthogonalizeToOtherPolynomials(poly_i, last_pt_included);

            //!<Orthogonalize polynomials on present block (deferring subsequent ones)>
            orthogonalizeBlock(points_shifted_.col(poly_i), poly_i, block_beginning, poly_i);

            last_pt_included = pt_next;
            poly_i++;
        } else {
            //!<These points don't render good pivot value for this>
            //!<specific polynomial. Exchange some polynomials>
            //!<and try to advance moving this polynomial>
            //!<to the end of the block>

            shiftPolynomialToEndBlock(poly_i, block_end);

            //!<Comment [1]: If we are on the linear block,>
            //!<this means we won't be able to build a Fully Linear model>
        }

        tr_center_ = 0;
        points_abs_ = all_points_.leftCols(last_pt_included + 1);
        points_shifted_ = points_shifted_.leftCols(last_pt_included + 1);
        fvalues_ = all_fvalues_.head(last_pt_included + 1);

        auto cache_size = std::fmin(n_points - last_pt_included - 1, 3 * pow(dim, 2));
        modeling_polynomials_.clear();

        //!<Points not included>
        if (cache_size > 0) {
            cached_points_ = all_points_.middleCols(last_pt_included, cache_size);
            cached_fvalues_ = fvalues_.segment(last_pt_included, cache_size);
        } else {
            cached_points_.resize(0, 0);
            cached_fvalues_.resize(0);
        }

    }
    //!<Clean auxiliary objects>
    all_points_.resize(0, 0);
    all_fvalues_.resize(0);

    return last_pt_included < n_points; //!<model has changed>
}

bool TrustRegionModel::improveModelNfp() {
    //TODO: implement this method
    return true;
}

int TrustRegionModel::ensureImprovement() {
    bool model_complete = isComplete();
    bool model_fl = isLambdaPoised();
    bool model_old = isOld();
    int exit_flag = 4;
    bool success = false;

    if (!model_complete && (!model_old || !model_fl)) {
        //!<Calculate a new point to add>
        success = improveModelNfp(); //!<improve model>
        if (success) {
            exit_flag = 1;
        }
    } else if ((model_complete) && (!model_old)){
        //!<Replace some point with a new one that improves geometry>
        success = chooseAndReplacePoint(); //!<replace point>
        if (success) {
            exit_flag = 2;
        }
    }
    if (!success) {
        bool model_changed = rebuildModel();
        if (!model_changed) {
            if (!model_complete) {
               //!<Improve model>
                success = improveModelNfp();
            } else {
               //!<Replace point>
                success = chooseAndReplacePoint();
            }
        } else {
            success = true;
        }
        if (model_old) {
            exit_flag = 3;
        } else {
            exit_flag = 4;
        }
    }
    return exit_flag;
}

bool TrustRegionModel::isLambdaPoised() {
    int dim = points_abs_.rows();
    int points_num = points_abs_.cols();
    double pivot_threshold = settings_->parameters().tr_pivot_threshold;
    bool result = false;

    if (settings_->parameters().tr_basis.compare("dummy")) {
        result = true;
    } else if(points_num >= dim+1) {
        //!<fully linear, already>
        result = true;
        //!<but lets double check>
//        if (pivot_values_.lpNorm<Infinity>() > settings_->parameters().tr_pivot_threshold) {
//            Printer::ext_warn("Low pivot values.", "Optimization", "TrustRegionOptimization");
//        }
    } else {
        result = false;
    }
    return result;
}

void TrustRegionModel::changeTrCenter(
        VectorXd new_point,
        RowVectorXd fvalues) {
    //TODO: implement this method
}

std::map<VectorXd, VectorXd> TrustRegionModel::solveTrSubproblem() {
    //TODO: implement this method
}

void TrustRegionModel::computePolynomialModels() {
    int dim = points_abs_.rows();
    int points_num = points_abs_.cols();
    int functions_num = 1; //!<Code currently supports only 1 function>
    int linear_terms = dim+1;
    int full_q_terms = (dim+1)*(dim+2)/2;
    std::vector<Polynomial> polynomials(functions_num);

    if ((linear_terms < points_num) && (points_num < full_q_terms)) {
        //!<Compute quadratic model>
        polynomials = computeQuadraticMNPolynomials();
    }

    //!<Ideally we should check if the model is a badly conditioned system
    if ((points_num <= linear_terms) || (points_num == full_q_terms)) {
        //!<Compute model with incomplete (complete) basis>
        auto l_alpha = nfpFiniteDifferences(points_num);
        for (int k=functions_num-1; k>=0; k--) {
            polynomials[k] = combinePolynomials(points_num, l_alpha);
            polynomials[k] = shiftPolynomial(polynomials[k]);
        }
    }
    modeling_polynomials_ = polynomials;
}


void TrustRegionModel::shiftPolynomialToEndBlock(
        int point_index,
        int block_end) {

    std::swap(pivot_polynomials_[point_index], pivot_polynomials_[block_end]);
    for (int pi=block_end-1; pi>point_index; pi--) {
        std::swap(pivot_polynomials_[pi], pivot_polynomials_[point_index]);
    }
}


void TrustRegionModel::sortVectorByIndex(
        VectorXd &vec,
        const VectorXd &ind) {
    VectorXd vec_ord(vec.size());
    for (int i=0; i < vec.size(); i++) {
        int index = int(ind(i));
        vec_ord(i) = vec(index);
    }

    for(int i=0; i<vec.size(); i++) {
        vec(i) = vec_ord(i);
    }
    vec_ord.resize(0);
}

void TrustRegionModel::sortVectorByIndex(
            RowVectorXd &vec,
            const VectorXd &ind) {
        VectorXd vec_ord(vec.size());
        for (int i=0; i < vec.size(); i++) {
            int index = int(ind(i));
            vec_ord(i) = vec(index);
        }

        for(int i=0; i<vec.size(); i++) {
            vec(i) = vec_ord(i);
        }
        vec_ord.resize(0);
}

void TrustRegionModel::sortMatrixByIndex(
        Matrix<double,Dynamic,Dynamic> &points,
        const VectorXd &ind) {

    Matrix<double,Dynamic,Dynamic> points_ord(points.rows(), points.cols());
    for (int i=0; i<points.cols(); i++) {
        int index = int(ind(i));
        points_ord.col(i) << points.col(index);
    }

    for (int i=0; i<points.cols(); i++) {
        points.col(i) << points_ord.col(i);
    }
    points_ord.resize(0,0);
}

void TrustRegionModel::nfpBasis(int dim) {
    //!<Number of terms>
    int poly_num = (dim+1)*(dim+2)/2;
    int linear_size = dim+1;

    //!<Calculating basis of polynomials>
    pivot_polynomials_.resize(poly_num);
    pivot_polynomials_[poly_num-1].dimension = dim;
    pivot_polynomials_[poly_num-1].coefficients.resize(poly_num);
    pivot_polynomials_[poly_num-1].coefficients.setZero(poly_num);

    for (int i=0; i<linear_size;i++) {
        pivot_polynomials_[i].dimension = dim;
        pivot_polynomials_[i].coefficients.resize(poly_num);
        pivot_polynomials_[i].coefficients.setZero(poly_num);
        pivot_polynomials_[i].coefficients(i) = 1;
    }

    //!<Quadratic entries>
    int c0 = 0;
    int m = 0;
    int n = 0;
    VectorXd g0(dim);
    g0.setZero(dim);

    for (int poly_i=linear_size; poly_i<poly_num; poly_i++) {
        MatrixXd H(dim,dim);
        H.setZero(dim,dim);
        if (m == n) {
            H(m,n) = 2;
        } else {
            H(m,n) = 1;
            H(n,m) = 1;
        }

        pivot_polynomials_[poly_i] = matricesToPolynomial(c0, g0, H);
        if (n < dim-1) {
            n++;
        } else {
            m++;
            n = m;
        }
    }
}

Polynomial TrustRegionModel::matricesToPolynomial(
        double c0,
        const VectorXd &g0,
        const MatrixXd &H) {

    int dim = g0.size();
    int n_terms = (dim+1)*(dim+2)/2;
    VectorXd coefficients(n_terms);
    coefficients.setZero(n_terms);

    //!<Zero order>
    coefficients(0) = c0;

    //!<First order>
    int ind_coefficients = dim;
    coefficients.segment(1,ind_coefficients) = g0;

    //!<Second order>
    for (int k=0; k<dim; k++) {
        for (int m=0; m <= k; m++) {
            ind_coefficients = ind_coefficients + 1;
            coefficients(ind_coefficients) = H(k, m);

            if (H(m, k) != H(k, m)) {
                Printer::ext_info("H not symmetrical: ","Optimization", "Trust Region");
            }
        }
    }

    Polynomial p;
    p.dimension = dim;
    p.coefficients = coefficients;
    return p;
}

std::tuple<double, VectorXd, MatrixXd> TrustRegionModel::coefficientsToMatrices(
        int dimension,
        VectorXd coefficients) {
    if (coefficients.size() == 0) {
        coefficients.resize(dimension);
        coefficients.setZero(dimension);
    }

    int n_terms = (dimension+1)*(dimension+2)/2;
    if (coefficients.size() != n_terms) {
        Printer::ext_warn("Wrong dimension of coefficients.", "Optimization", "TrustRegionModel");
        throw std::runtime_error("Failed to convert polynomial coefficients to matrices.");
    }

    //!<Constant term>
    double c = coefficients(0);

    //!<Order one term>
    int idx_coefficients = dimension + 1;
    auto g = coefficients.segment(1,idx_coefficients-1);

    //!<Second order term>
    MatrixXd H(dimension, dimension);
    H.setZero(dimension,dimension);

    for (int k=0; k<dimension; k++) {
        for (int m=0; m < k; m++) {
            idx_coefficients++;
            H(k,m) = coefficients(idx_coefficients);
            H(m,k) = H(k,m);
        }
    }

    return std::make_tuple(c, g, H);
}


Polynomial TrustRegionModel::normalizePolynomial(
        int poly_i,
        int pt_next) {

    auto polynomial = pivot_polynomials_[poly_i];
    auto point = points_shifted_.col(pt_next);
    auto val = evaluatePolynomial(polynomial, point);
    for (int i=0; i<3; i++) {
        polynomial = multiplyPolynomial(polynomial, (double) 1/val);
        val = evaluatePolynomial(polynomial, point);
        if ((val - 1) == 0) {
            break;
        }
    }
    return polynomial;
}

Polynomial TrustRegionModel::orthogonalizeToOtherPolynomials(
        int poly_i,
        int last_pt) {
    auto polynomial = pivot_polynomials_[poly_i];
    for (int n=0; n<=last_pt; n++) {
        if (n != poly_i) {
            polynomial = zeroAtPoint(polynomial, pivot_polynomials_[n], points_shifted_.col(n));
        }
    }
    return polynomial;
}

void TrustRegionModel::orthogonalizeBlock(
        VectorXd point,
        int np,
        int block_beginning,
        int block_end) {
    for (int p = block_beginning; p<=block_end; p++) {
        if (p != np) {
            pivot_polynomials_[p] = zeroAtPoint(pivot_polynomials_[p], pivot_polynomials_[np], point);
        }
    }
}


Polynomial TrustRegionModel::zeroAtPoint(
        Polynomial p1,
        Polynomial p2,
        VectorXd x) {
    Polynomial p;
    p = p1;

    auto px = evaluatePolynomial(p, x);
    auto p2x = evaluatePolynomial(p2, x);
    int iter = 1;
    while (px != 0) {
        p = addPolynomial(p, multiplyPolynomial(p2, -px/p2x));
        px = evaluatePolynomial(p, x);

        if (iter >= 2)
            break;
        iter++;
    }
    return p;
}


double TrustRegionModel::evaluatePolynomial(
        Polynomial p1,
        VectorXd x) {
    int c;
    double terms[3];
    VectorXd g(p1.dimension);
    MatrixXd H(p1.dimension, p1.dimension);
    std::tie(c, g, H) = coefficientsToMatrices(p1.dimension, p1.coefficients);

    terms[0] = c;
    terms[1] = g.transpose()*x;
    terms[2] = (x.transpose()*H*x);
    terms[2] *= 0.5;

    return terms[0] + terms[1] + terms[2];

}

Polynomial TrustRegionModel::addPolynomial(
        Polynomial p1,
        Polynomial p2) {
    if (p1.dimension != p2.dimension) {
        Printer::ext_warn("Summation of polynomials with different dimensions.", "Optimization", "TrustRegionModel");
        throw std::runtime_error("Failed to compute add polynomials. They have different dimensions.");
    }

    Polynomial p;
    p.dimension = p1.dimension;
    p.coefficients.resize(p1.coefficients.size());
    p.coefficients = p1.coefficients + p2.coefficients;

    if (p.coefficients.size() == 0) {
        Printer::ext_warn("Empty resulting polynomial.", "Optimization", "TrustRegionModel");
        throw std::runtime_error("Failed to compute add polynomials. The resulting polynomial is empty.");
    }

    return p;
}

Polynomial TrustRegionModel::multiplyPolynomial(
        Polynomial p1,
        double factor) {
    Polynomial p = p1;
    p.coefficients = p1.coefficients.array()*factor;
    return p;
}

int TrustRegionModel::findBestPoint() {
    int dim = points_abs_.rows();
    int n_points = points_abs_.cols();
    int best_i = 0;
    auto min_f = std::numeric_limits<double>::infinity();


    for (int k=0; k<n_points; k++) {
        VectorXd point = points_abs_.col(k);
        if (((point - lb_).minCoeff() >= 0) &&  ((ub_ - point).minCoeff() > 0)) {
            auto val = fvalues_(k);
            if (val < min_f) {
                min_f = val;
                best_i = k;
            }
        }
    }
    return best_i;
}

std::vector<Polynomial> TrustRegionModel::computeQuadraticMNPolynomials() {
    int dim = points_abs_.rows();
    int points_num = points_abs_.cols();
    int functions_num = fvalues_.rows();
    std::vector<Polynomial> polynomials(functions_num);

    MatrixXd points_shifted = MatrixXd::Zero(dim, points_num-1);
    RowVectorXd fvalues_diff = RowVectorXd::Zero(functions_num, points_num-1);

    int m2 = 0;
    for (int m = tr_center_; (m<points_num) && (m != tr_center_); m++) {
        points_shifted.col(m2) = points_abs_.col(m) - points_abs_.col(tr_center_);
        fvalues_diff(m2) = fvalues_(m) - fvalues_(tr_center_);
        m2++;
    }

    MatrixXd M = points_shifted.transpose()*points_shifted;
    M += 0.5*(MatrixXd)(M.array().square());

    //!<Solve symmetric system> //TODO: choose best algorithm for solving linear system of equations automatically.
    PartialPivLU<Ref<MatrixXd> > lu(M);
    lu.compute(M); //!<Update LU matrix>
    auto mult_mn = lu.solve(fvalues_diff.transpose());

    //TODO: raise a warning if the system is badly conditioned using the resulting conditioning number (tol=1e4*eps(1))

    for (int n=0; n<functions_num; n++) {
        VectorXd g = VectorXd::Zero(dim);
        MatrixXd H = MatrixXd::Zero(dim,dim);
        for (int m=0; m<points_num-1; m++) {
            g += mult_mn(m)*points_shifted.col(m);
            H += mult_mn(m)*(points_shifted.col(m)*points_shifted.col(m).transpose());
        }
        auto c = fvalues_(tr_center_);
        polynomials[n] = matricesToPolynomial(c, g, H);
    }
    return polynomials;
}

RowVectorXd TrustRegionModel::nfpFiniteDifferences(int points_num) {
    //!<Change so we can interpolate more functions at the same time>
    int dim = points_shifted_.cols();
    RowVectorXd l_alpha = fvalues_;
    std::vector<Polynomial> polynomials = std::vector<Polynomial>(pivot_polynomials_.begin(), pivot_polynomials_.begin() + points_num);

    //!<Remove constant polynomial>
    for (int m=1; m<points_num; m++) {
        auto val = evaluatePolynomial(polynomials[0], points_shifted_.col(m));
        l_alpha(m) = l_alpha(m) - l_alpha(0)*val;
    }

    //!<Remove terms corresponding to degree 1 polynomials>
    for (int m=dim+1; m <points_num; m++) {
        for (int n=1; n<dim+1; n++) {
            auto val = evaluatePolynomial(polynomials[n], points_shifted_.col(m));
            l_alpha(m) = l_alpha(m) - l_alpha(n)*val;
        }
    }
    polynomials.clear();
    return l_alpha;
}

Polynomial TrustRegionModel::combinePolynomials(
        int points_num,
        RowVectorXd coefficients) {
    auto polynomials = std::vector<Polynomial>(pivot_polynomials_.begin(), pivot_polynomials_.begin() + points_num);

    int terms = polynomials.size();
    if ((terms == 0) || (coefficients.size() != terms)) {
        Printer::ext_warn("Polynomial and coefficients have different sizes.", "Optimization", "TrustRegionModel");
        throw std::runtime_error(
                "Failed to combine polynomials. Polynomial and coefficients have different dimensions.");
    }

    auto p = multiplyPolynomial(polynomials[0], coefficients(0));
    for (int k = 1; k < terms; k++) {
        p = addPolynomial(p, multiplyPolynomial(polynomials[k], coefficients[k]));
    }
    return p;
}

Polynomial TrustRegionModel::shiftPolynomial(Polynomial polynomial) {
    double c;
    double terms[3];
    VectorXd s = points_shifted_.col(tr_center_);
    VectorXd g(polynomial.dimension);
    MatrixXd H(polynomial.dimension, polynomial.dimension);

    std::tie(c, g, H) = coefficientsToMatrices(polynomial.dimension, polynomial.coefficients);


    double c_mod = c + (double)(g.transpose()*s) + 0.5*(double)(s.transpose()*H*s);

    VectorXd g_mod(g.size());
    g_mod = g + H*s;

    return matricesToPolynomial(c_mod,g_mod, H);
}

bool TrustRegionModel::isComplete() {
    int dim = points_abs_.rows();
    int points_num = points_abs_.cols();
    int max_terms = ((dim+1)*(dim+2))/2;
    if (points_num > max_terms) {
        Printer::ext_warn("Too many points in the Trust Region model.", "Optimization", "TrustRegionModel");
    }
    return (points_num >= max_terms );;
}

bool TrustRegionModel::isOld() {
    VectorXd distance(points_abs_.rows());
    distance = points_abs_.col(0) - points_abs_.col(tr_center_);
    return (distance.lpNorm<Infinity>() > settings_->parameters().tr_radius_factor);
}

bool TrustRegionModel::chooseAndReplacePoint() {
    //TODO: implement this method
}

}
}