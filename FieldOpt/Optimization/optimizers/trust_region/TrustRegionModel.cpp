/******************************************************************************
   Created by thiagols on 27/11/18.
   Copyright (C) 2018 Thiago Lima Silva <thiagolims@gmail.com>

   This file is part of the FieldOpt project.

   FieldOpt is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   FieldOpt is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with FieldOpt.  If not, see <http://www.gnu.org/licenses/>.
******************************************************************************/
#include "TrustRegionModel.h"
#include <numeric>
#include <functional>
#include <Settings/optimizer.h>
#include "Utilities/printer.hpp"
#include "Utilities/stringhelpers.hpp"
#include <Utilities/verbosity.h>

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
        const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& initial_points,
        const Eigen::RowVectorXd& initial_fvalues,
        Settings::Optimizer *settings) {

    settings_ = settings;
    points_abs_.setZero(initial_points.rows(), initial_points.cols());
    points_abs_ << initial_points;

    fvalues_.setZero(initial_fvalues.size());
    fvalues_ << initial_fvalues;

    radius_ =  settings_->parameters().tr_initial_radius;
    tr_center_ = 0;
    dim_ = initial_points.rows();
    cache_max_ = 3*pow(dim_,2);


    cached_fvalues_.resize(0);
    index_vector_.resize(0);
    distances_.resize(0);
    cached_points_.resize(0,0);
    points_shifted_.resize(0,0);

    rebuildModel();
    moveToBestPoint();
    computePolynomialModels();

    if (sizeof(points_abs_) < 2) {
        ensureImprovement();
    }
}

int TrustRegionModel::getDimension() {
    return dim_;
}

double TrustRegionModel::getRadius() {
    return radius_;
}

Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> TrustRegionModel::getPoints() {
    return points_abs_;
}

Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> TrustRegionModel::getPointsShifted() {
    return points_shifted_;
}

Eigen::RowVectorXd TrustRegionModel::getFunctionValues() {
    return fvalues_;
}

std::vector<Polynomial> TrustRegionModel::getPivotPolynomials() {
    return pivot_polynomials_;
}

void TrustRegionModel::moveToBestPoint() {
    //TODO: implement this method
}

void TrustRegionModel::criticalityStep() {
    //TODO: implement this method
}

double TrustRegionModel::checkInterpolation() {
    //TODO: implement this method
}

void TrustRegionModel::rebuildModel() {
//    //!<dbg>
//    Eigen::IOFormat frmt(3, 0, " ", "\n", "             [", "]");
//    cout << "[          ] all_points_" << endl;
//    cout << all_points_.format(frmt) << endl;

    auto pivot_threshold = settings_->parameters().tr_pivot_threshold*std::fmin(1, radius_);
    all_points_.resize(points_abs_.rows(), points_abs_.cols() + cached_points_.cols()); //!<All points we know>
    if (cached_points_.size() == 0) {
        all_points_ = points_abs_;
    } else {
        all_points_ << points_abs_;
        all_points_ << cached_points_;
    }

    all_fvalues_.resize(fvalues_.size() + cached_fvalues_.size()); //!<All function values we know>

    if (cached_fvalues_.size() == 0) {
        all_fvalues_ = fvalues_;
    } else {
        all_fvalues_ << fvalues_;
        all_fvalues_ << cached_fvalues_;
    }

    int dim = all_points_.rows();
    int n_points = all_points_.cols();
    if (tr_center_ != 0) {
        all_points_.col(0).swap(all_points_.col(tr_center_)); //!<center will be first>
        auto center_fvalue = all_points_(tr_center_);
        all_fvalues_(tr_center_) = all_fvalues_(0);
        all_fvalues_(0) = center_fvalue;
    }
    //!<calculate distances from tr center>
    points_shifted_.resize(dim, n_points);
    distances_.resize(n_points);
    points_shifted_.Zero(dim, n_points);
    distances_.Zero(n_points);
    for (int i=1; i<n_points; i++) { //!<Shift all points to TR center>
        points_shifted_.col(i) << all_points_.col(i) - all_points_.col(0); //!<Compute distances>
        distances_(i) = points_shifted_.col(i).lpNorm<Eigen::Infinity>(); //<!distances in infinity or 2-norm>
    }

    //!<Reorder points based on their distances to the tr center>
    index_vector_.setLinSpaced(distances_.size(),0,distances_.size()-1);
    std::sort(index_vector_.data(), index_vector_.data() + index_vector_.size(), std::bind(compare, std::placeholders::_1,  std::placeholders::_2, distances_.data()));
    sortVectorByIndex(distances_, index_vector_);
    sortVectorByIndex(all_fvalues_, index_vector_);

    sortMatrixByIndex(points_shifted_, index_vector_);
    sortMatrixByIndex(all_points_, index_vector_);

    nfpBasis(dim);//!<build nfp polynomial basis>

    int polynomials_num = pivot_polynomials_.size();
    Eigen::RowVectorXd pivot_values(polynomials_num);
    pivot_values.setZero(polynomials_num);

    //!<constant term>
    int last_pt_included = 1;
    int poly_i = 1;
    pivot_values(0) = 1;

    //!<Gaussian elimination (using previous points)>
    for (int iter=1; iter<polynomials_num; iter++) {
        pivot_polynomials_[poly_i] = orthogonalizeToOtherPolynomials(poly_i, last_pt_included);

        double max_layer;
        double farthest_point = distances_(distances_.size()-1);
        double distance_farthest_point = (double) (farthest_point/radius_);
        if (poly_i <= dim) {
            int block_beginning = 1;
            int block_end = dim;

            //!<linear block -- we allow more points (*2)

            max_layer = std::min(2*settings_->parameters().tr_radius_factor, distance_farthest_point);
            if (iter > dim) {
                //!< We already tested all linear terms.
                //!< We do not have points to build a FL model.
                //!< How did this happen??? see Comment [1]>
                break;
            } else {  //!<Quadratic block -- being more carefull>
                max_layer = min(settings_->parameters().tr_radius_factor, distance_farthest_point);
                block_beginning = dim+1;
                block_end = polynomials_num-1;
            }

            max_layer = std::fmax(1, max_layer); //TODO: check this max function
            Eigen::VectorXd all_layers;

            all_layers.setLinSpaced(ceil(max_layer), 1, max_layer);
            double max_absval = 0;
            double pt_max = 0;
            for (int i=0; i<all_layers.size();i++) {
                auto layer = all_layers(i);
                double dist_max = layer*radius_;
                for (int n=last_pt_included+1; n<n_points; n++) {
                    if (distances_(n) > dist_max) {
                        break; //!<for n>
                    }
                    auto val = evaluatePolynomial(pivot_polynomials_[poly_i], points_shifted_.col(n));
                    val = val/dist_max; //!<minor adjustment>
                    if (abs(max_absval) < abs(val)) {
                        max_absval = val;
                        pt_max = n;
                    }
                    if (abs(max_absval) > pivot_threshold) {
                        break; //!<for(layer)>
                    }
                }
            }
            //TODO: finish method gaussian elimination.

        }



    }

}

void TrustRegionModel::improveModelNfp() {
    //TODO: implement this method
}

void TrustRegionModel::ensureImprovement() {
    //TODO: implement this method
}

bool TrustRegionModel::isLambdaPoised() {
    //TODO: implement this method
}

void TrustRegionModel::changeTrCenter(
        Eigen::VectorXd new_point,
        Eigen::RowVectorXd fvalues) {
    //TODO: implement this method
}

std::map<Eigen::VectorXd, Eigen::VectorXd> TrustRegionModel::solveTrSubproblem() {
    //TODO: implement this method
}

void TrustRegionModel::computePolynomialModels() {
    //TODO: implement this method
}

void TrustRegionModel::sortVectorByIndex(
        Eigen::VectorXd &vec,
        const Eigen::VectorXd &ind) {
    Eigen::VectorXd vec_ord(vec.size());
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
            Eigen::RowVectorXd &vec,
            const Eigen::VectorXd &ind) {
        Eigen::VectorXd vec_ord(vec.size());
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
        Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> &points,
        const Eigen::VectorXd &ind) {

    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> points_ord(points.rows(), points.cols());
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
    //!<number of terms>
    int poly_num = (dim+1)*(dim+2)/2;
    int linear_size = dim+1;

    //!<calculating basis of polynomials>
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

    //!<quadratic entries>
    int c0 = 0;
    int m = 0;
    int n = 0;
    Eigen::VectorXd g0(dim);
    g0.setZero(dim);

    for (int poly_i=linear_size; poly_i<poly_num; poly_i++) {
        Eigen::MatrixXd H(dim,dim);
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
        int c0,
        const Eigen::VectorXd &g0,
        const Eigen::MatrixXd &H) {

    int dim = g0.size();
    int n_terms = (dim+1)*(dim+2)/2;
    Eigen::VectorXd coefficients(n_terms);
    coefficients.setZero(n_terms);

    //!<zero order>
    coefficients(0) = c0;

    //!<first order>
    int ind_coefficients = dim;
    coefficients.segment(0,ind_coefficients) = g0;

    //!<second order>
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
    p.dimension = 2;
    p.coefficients = coefficients;
    return p;
}

std::tuple<int, Eigen::VectorXd, Eigen::MatrixXd> TrustRegionModel::coefficientsToMatrices(
        int dimension,
        Eigen::VectorXd coefficients) {
    if (coefficients.size() == 0) {
        coefficients.resize(dimension);
        coefficients.setZero(dimension);
    }

    int n_terms = (dimension+1)*(dimension+2)/2;
    if (coefficients.size() != n_terms) {
        Printer::ext_warn("Wrong dimension of coefficients.", "Optimization", "TrustRegionModel");
        throw std::runtime_error("Failed to convert polynomial coefficients to matrices.");
    }

    //!<constant term>
    auto c = coefficients(0);

    //!<order one term>
    int idx_coefficients = dimension + 1;
    auto g = coefficients.segment(1,idx_coefficients-1);

    //!<second order term>
    Eigen::MatrixXd H(dimension, dimension);
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

Polynomial TrustRegionModel::orthogonalizeToOtherPolynomials(
        int poly_i,
        int last_pt) {
    auto polynomial = pivot_polynomials_[poly_i];
    for (int n=0; n<last_pt; n++) {
        if (n != poly_i) {
            polynomial = zeroAtPoint(polynomial, pivot_polynomials_[n], all_points_.col(n));
        }
    }
    return polynomial;
}

Polynomial TrustRegionModel::zeroAtPoint(
        Polynomial p1,
        Polynomial p2,
        Eigen::VectorXd x) {
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
        Eigen::VectorXd x) {
    int c;
    double terms[3];
    Eigen::VectorXd g(p1.dimension);
    Eigen::MatrixXd H(p1.dimension, p1.dimension);
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

}
}