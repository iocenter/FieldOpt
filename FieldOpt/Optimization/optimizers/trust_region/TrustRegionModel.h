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

#ifndef FIELDOPT_TRUSTREGIONMODEL_H
#define FIELDOPT_TRUSTREGIONMODEL_H

#include <Settings/optimizer.h>
#include <Eigen/Core>
#include <vector>
#include <tuple>

using namespace Eigen;

struct Polynomial {
    int dimension = 0;
    VectorXd coefficients;
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
            const Matrix<double,Dynamic,Dynamic>& initial_points,
            const RowVectorXd& initial_fvalues,
            VectorXd& lb,
            VectorXd& ub,
            Settings::Optimizer *settings);

    int TrustRegionModel::getDimension() { return dim_; }
    double TrustRegionModel::getRadius() { return radius_; }

    Matrix<double,Dynamic,Dynamic> TrustRegionModel::getPoints() { return points_abs_; }
    Matrix<double,Dynamic,Dynamic> TrustRegionModel::getPointsShifted() { return points_shifted_; }

    RowVectorXd TrustRegionModel::getFunctionValues() { return fvalues_; }
    std::vector<Polynomial> TrustRegionModel::getPivotPolynomials() { return pivot_polynomials_; }

    std::vector<Polynomial> getModelingPolynomials() { return modeling_polynomials_ ;}
    RowVectorXd getPivotValues() { return pivot_values_;}
    
    /*!
   * @brief changes TR center pointer to best point
   * considering lower and upper bounds on variables.
   */
    void moveToBestPoint();

    void criticalityStep();

    double checkInterpolation();

    /*!
    * @brief Rebuild the polynomial model
    * @return True if the model has changed, and false otherwise.
    */
    bool rebuildModel();

    /*!
    * @brief Improve the Newton Fundamental Polynomial model
    * @return True if the model was improved, and false otherwise.
    */
    bool improveModelNfp();

    /*!
    * @brief Ensure improvement of model.
     *@return exitflag: 1 = new point calculated,
     *                  2 = replaced point an existing one that improve geometry,
     *                  3 = model was old and had to be rebuilt,
     *                  4 = failed to ensure improvement.
    * considering lower and upper bounds on variables.
    */
    int ensureImprovement();

    /*!
    * @brief tests whether a model is lambda-poised for the given options.
    * Important: it assumes there is a finite radius_max defined.    *
    * pivot_threshold defines how well-poised we demand a model to be.
    * @return true if the model is lambda poised, and false otherwise.
    */
    bool isLambdaPoised();

    void changeTrCenter(VectorXd new_point, RowVectorXd fvalues);

    std::map<VectorXd, VectorXd> solveTrSubproblem();

    void computePolynomialModels();


 private:
    Settings::Optimizer *settings_;
    std::vector<Polynomial> pivot_polynomials_;
    std::vector<Polynomial> modeling_polynomials_;
    Matrix<double,Dynamic,Dynamic> all_points_;
    Matrix<double,Dynamic,Dynamic> points_abs_;
    Matrix<double,Dynamic,Dynamic> points_shifted_;
    Matrix<double,Dynamic,Dynamic> cached_points_;
    RowVectorXd all_fvalues_;
    RowVectorXd fvalues_;
    RowVectorXd cached_fvalues_;
    RowVectorXd pivot_values_;
    VectorXd index_vector_;
    VectorXd distances_;
    VectorXd lb_;
    VectorXd ub_;

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
            RowVectorXd &vec,
            const VectorXd &ind);


    /*!
   * @brief sort column vector for a given index ordering.
   * @param &vec pointer to the vector.
   * @param &ind pointer to the vector of ordered index.
   */
    void sortVectorByIndex(
            VectorXd &vec,
            const VectorXd &ind);

    /*!
  * @brief sort matrix of column vectors (points)
  * for a given index ordering.
   * @param &points pointer to the matrix of points.
   * @param &ind pointer to the vector of ordered index.
   */
    void sortMatrixByIndex(
            Matrix<double,Dynamic,Dynamic> &points,
            const VectorXd &ind);


     /*!
  * @brief Build a Newton Fundamental Polynomial
   * basis for the corresponding dimenion.
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
            double c0,
            const VectorXd &g0,
            const MatrixXd &H);

    /*!
     * @brief converts polynomial coefficients to matrices c, g, H
     * @param dimension dimension of polynomial.
     * @param coefficients coefficients of polynomial.
     * @return tuple<int, VectorXd, MatrixXd> matrices c, g, and H  respectively.
     */
    std::tuple<double, VectorXd, MatrixXd> coefficientsToMatrices(
            int dimension,
            VectorXd coefficients);


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
     * Orthogonalize polynomials on present block (deffering subsequent ones)
   * @param point point used to orthogonalize polynomials.
   * @param poly_i polynomial index in pivot_polynomials_.
   * @param poly_i end of the block to be orthogonalized.
   * @param block_beginning beginning of the block to be orthogonalized.
   */
    void orthogonalizeBlock(
            VectorXd point,
            int np,
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
            VectorXd x);

    /*!
     * @brief evaluate polynomial p1 in given point x.
     * @param p1 the polynomial.
     * @param x the point.
     * @return a scalar with the evaluation result.
     */
    double evaluatePolynomial(
            Polynomial p1,
            VectorXd x);

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



    /*!
     * @brief Find the best point of the model, i.e., the one for which the function has minimum value.
     * @return best point index.
     */
    int findBestPoint();


    /*!
     * @brief Compute the modelling polynomials
     * @return vector with the resulting polynomials.
     */
    std::vector<Polynomial> computeQuadraticMNPolynomials();


    /*!
    * @brief Evaluate NFP with finite differences.
    * @param Number of points in the model.
    * @return row vector with the size of number of points containing the resulting values
    */
    RowVectorXd nfpFiniteDifferences(int points_num);

    /*!
    * @brief Combine polynomials.
    * @param number of points in the model.
    * @param coefficient values (alpha).
    * @return resulting polynomial.
    */
    Polynomial combinePolynomials(
            int points_num,
            RowVectorXd coefficients);

    /*!
    * @brief Shift polynomial with respect to the TR center.
    * @param polynomial to be shifted.
    * @return resulting polynomial.
    */
    Polynomial shiftPolynomial(Polynomial polynomial);

    /*!
    * @brief Check if the model is complete, i.e., number of points is at least (dimension+1)*(dimension+2)/2
    * @return true if model is complete, and false otherwise.
    */
    bool isComplete();

    /*!
    * @brief Check if model is old, i.e., if the max distance from points to tr center is greater than the radius.
    * @return true if the model is old, and false otherwise.
    */
    bool isOld();

    /*!
    * @brief Choose worst point and replace it by a new one.
    * @return true if the worst point was successfully replaced, and false otherwise.
    */
    bool chooseAndReplacePoint();
};

}
}


#endif //FIELDOPT_TRUSTREGIONMODEL_H
