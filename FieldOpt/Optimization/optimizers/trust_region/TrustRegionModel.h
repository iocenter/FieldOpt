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

#ifndef FIELDOPT_TRUSTREGIONMODEL_H
#define FIELDOPT_TRUSTREGIONMODEL_H

#include <Settings/optimizer.h>
#include <Eigen/Core>
#include <vector>
#include <tuple>

struct Polynomial {
    int dimension = 0;
    Eigen::VectorXd coefficients;
};

namespace Optimization {
namespace Optimizers {

/*!
 * @brief This class is an implementation of a Trust Region model.
 *
 * It delimits a region within a radius, with a set of points
 * and the corresponding function values. A polynomial model
 * is built to represent the function within the delimited region.
 */
class TrustRegionModel {
 public:
  /*!
   * @brief Initialize the trust region model.
   */
    TrustRegionModel();
    TrustRegionModel(
            const Eigen::Matrix<double,
            Eigen::Dynamic,Eigen::Dynamic>& initial_points,
            const Eigen::RowVectorXd& initial_fvalues ,
            Settings::Optimizer *settings);
    int getDimension();
    double getRadius();
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> getPoints();
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> getPointsShifted();
    Eigen::RowVectorXd getFunctionValues();
    std::vector<Polynomial> getPivotPolynomials();

    void moveToBestPoint();
    void criticalityStep();
    double checkInterpolation();
    void rebuildModel();
    void improveModelNfp();
    void ensureImprovement();
    bool isLambdaPoised();
    void changeTrCenter(Eigen::VectorXd new_point, Eigen::RowVectorXd fvalues);
    std::map<Eigen::VectorXd, Eigen::VectorXd> solveTrSubproblem();
    void computePolynomialModels();


 private:
    Settings::Optimizer *settings_;
    std::vector<Polynomial> pivot_polynomials_;
    std::vector<Polynomial> modeling_polynomials_;
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> all_points_;
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> points_abs_;
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> points_shifted_;
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> cached_points_;
    Eigen::RowVectorXd all_fvalues_;
    Eigen::RowVectorXd fvalues_;
    Eigen::RowVectorXd cached_fvalues_;
    Eigen::RowVectorXd pivot_values_;
    Eigen::VectorXd index_vector_;
    Eigen::VectorXd distances_;

    double radius_;
    int tr_center_; //!<index of trust region center point in points_abs>
    int cache_max_;
    int dim_;

   /*!
   * @brief shift the point
   * @param pointer to the matrix containing the points to be reordered.
   * @param point_index point index and beginning of the block.
   * @param end of block index.
   */
    void shiftPolynomialToEndBlock(
           int point_index,
           int block_end);

    /*!
   * @brief sort row vector for a given index ordering.
   * @param &vec pointer to the vector.
   * @param &ind pointer to the vector of ordered index.
   */
    void sortVectorByIndex(
            Eigen::RowVectorXd &vec,
            const Eigen::VectorXd &ind);


    /*!
   * @brief sort column vector for a given index ordering.
   * @param &vec pointer to the vector.
   * @param &ind pointer to the vector of ordered index.
   */
    void sortVectorByIndex(
            Eigen::VectorXd &vec,
            const Eigen::VectorXd &ind);

    /*!
   * @brief sort matrix of column vectors (points) for a given index ordering.
   * @param &points pointer to the matrix of points.
   * @param &ind pointer to the vector of ordered index.
   */
    void sortMatrixByIndex(
            Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> &points,
            const Eigen::VectorXd &ind);


     /*!
    * @brief Build a Newton Fundamental Polynomial basis for the corresponding dimenion.
    * @param dim polynomial basis dimension.
    */
    void nfpBasis(int dim);

    /*!
   * @brief converts matrices c, g, and H into polynomial coefficients.
   * @param c0 constant term.
   * @param g0 linear terms (vector).
   * @param H quadratic term (matrix)
   * @return Polynomial containing the corresponding dimension and coefficients.
   */
    Polynomial matricesToPolynomial(
            int c0,
            const Eigen::VectorXd &g0,
            const Eigen::MatrixXd &H);

    /*!
     * @brief converts polynomial coefficients to matrices c, g, H
     * @param dimension dimension of polynomial.
     * @param coefficients coefficients of polynomial.
     * @return tuple<int, Eigen::VectorXd, Eigen::MatrixXd> matrices c, g, and H  respectively.
     */
    std::tuple<int, Eigen::VectorXd, Eigen::MatrixXd> coefficientsToMatrices(
            int dimension,
            Eigen::VectorXd coefficients);



    /*!
     * @brief normalize polynomial with respect to a point
     * @param poly_i index of polynomial to be normalized.
     * @param pt_next index point in points_shifted participating in the normalization.
     * @return resulting polynomial.
     */
    Polynomial normalizePolynomial(
            int poly_i,
            int pt_next);

    /*!
     * @brief orthogonalize polynomial pivot_polynomials_(poly_i) with respect to others in pivot_polynomials_.
     * @param poly_i index of polynomial to be orthogonalized.
     * @param last_pt index of last polynomial participating in the orthogonalization.
     * @return resulting polynomial.
     */
    Polynomial orthogonalizeToOtherPolynomials(
            int poly_i,
            int last_pt);

    /*!
   * @brief orthogonalize a block from pivot_polynomials_.
   * @param point point used to orthogonalize polynomials.
   * @param poly_i polynomial index in pivot_polynomials_.
   * @param poly_i end of the block to be orthogonalized.
   * @param block_beginning beginning of the block to be orthogonalized.
   */
    void orthogonalizeBlock(
            Eigen::VectorXd point,
            int poly_i,
            int block_beginning,
            int block_end);


    /*!
     * @brief subtract polynomials p1 and p2 so that the result is zero at x.
     * @param p1 first polynomial.
     * @param p2 second polynomial.
     * @return resulting polynomial.
     */
    Polynomial zeroAtPoint(
            Polynomial p1,
            Polynomial p2,
            Eigen::VectorXd x);


    /*!
     * @brief evaluate polynomial p1 in given point x.
     * @param p1 the polynomial.
     * @param x the point.
     * @return a scalar with the evaluation result.
     */
    double evaluatePolynomial(
            Polynomial p1,
            Eigen::VectorXd x);

    /*!
     * @brief Add two polynomials.
     * @param p1 first polynomial
     * @param p2 second polynomial
     * @return resulting polynomial from the summation of p1 and p2.
     */
    Polynomial addPolynomial(
            Polynomial p1,
            Polynomial p2);

    /*!
     * @brief Multiply a polynomial by a constant factor.
     * @param p1 the polynomial.
     * @param factor the constant factor.
     * @return resulting polynomial p = p1.*factor.
     */
    Polynomial multiplyPolynomial(
            Polynomial p1,
            double factor);
};

}
}


#endif //FIELDOPT_TRUSTREGIONMODEL_H
