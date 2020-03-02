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

#include <Optimization/case.h>
#include <Settings/optimizer.h>
#include <Optimization/solvers/SNOPTSolver.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <vector>
#include <tuple>

using namespace Eigen;
using std::vector;
using std::tuple;

struct Polynomial {
    int dimension = 0;
    VectorXd coefficients;
};

struct ModelMatrix {
  double c = 0.0;
  VectorXd g;
  MatrixXd H;
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
    TrustRegionModel(
            VectorXd& lb,
            VectorXd& ub,
            Case *base_case,
            Settings::Optimizer *settings
            );

    void DBG_printPivotPolynomials(string msg);
    void DBG_printPolynomials(string msg, Polynomial polynomial);
    void DBG_printToFile(string fn, string sout);

    void DBG_printHeader(stringstream &ss, string msg="", int htype=0);
    void DBG_printModelData(string msg="");
    void DBG_printSettingsData(string msg="");

  void DBG_printFunctionData(
      string fnm = "none", string msg = "",
      VectorXd v0 = VectorXd::Zero(0),
      VectorXd v1 = VectorXd::Zero(0),
      VectorXd v2 = VectorXd::Zero(0),
      double d0 = -9.99, double d1 = -9.99,
      double d2 = -9.99);

    void DBG_printExchangePoint(string msg="",
        VectorXd v0 = VectorXd::Zero(0),
        VectorXd v1 = VectorXd::Zero(0));

  inline string DBG_printDouble(double argout,
      string frmt="% 10.3e ", string fn="") {
      stringstream ss;
      char buffer [100];
      sprintf(buffer, frmt.c_str(), argout);
      ss << buffer;
      if (fn != "") DBG_printToFile(fn, ss.str() + "\n");
      return ss.str();
    }

    inline string DBG_printVectorXd(VectorXd vec,
        string mv="", string fv="% 10.3e ", string fn="") {

      stringstream ss;
      if ( vec.size() > 0 ) {
        if (mv != "") ss << mv;
        ss << "[";
        for (int ii = 0; ii < vec.size() - 1; ii++) {
          ss << DBG_printDouble(vec(ii), fv);
        }
        ss << DBG_printDouble(vec(vec.size() - 1)) << "]";
      } else {
        ss << "[ Empty ]";
      }

      if (fn != "") DBG_printToFile(fn, ss.str() + "\n");
      return ss.str();
    }

    inline string DBG_printMatrixXd(MatrixXd mat,
                                    string mm="", string fm="% 10.3e ") {

      stringstream ss;
      for (int ii = 0; ii < mat.rows(); ii++) {
        if (mm != "") ss << "\n" << mm << "#" << ii;
        ss << DBG_printVectorXd(mat.row(ii), mm, fm);
      }
      return ss.str();
    }

    string DBG_fn_pivp_;
    string DBG_fn_mdat_;
    string DBG_fn_sdat_;
    string DBG_fn_xchp_;
    string DBG_fn_co2m_;
    string DBG_fn_m2co_;
    string DBG_fn_pcfs_;

    SNOPTSolver *SNOPTSolver_;

    void setXDim(int dim) { dim_ = dim; }
    int getXDim() { return dim_; }

    int getNumPts() {
        return static_cast<int>(points_abs_.cols());
    }

    int getNumFvals() {
        return static_cast<int>(fvalues_.rows());
    }

    double getRadius() { return radius_; }
    void setRadius(double r) { radius_ = r;}

    Matrix<double,Dynamic,Dynamic> getPoints() { return points_abs_; }
    Matrix<double,Dynamic,Dynamic> getCachedPoints() { return cached_points_; }
    Matrix<double,Dynamic,Dynamic> getPointsShifted() { return points_shifted_; }
    VectorXd getCurrentPoint() { return points_abs_.col(tr_center_);}
    double getCurrentFval() { return fvalues_(tr_center_);}

    RowVectorXd getFunctionValues() { return fvalues_; }
    std::vector<Polynomial> getPivotPolynomials() { return pivot_polynomials_; }

    std::vector<Polynomial> getModelingPolynomials() { return modeling_polynomials_ ;}
    RowVectorXd getPivotValues() { return pivot_values_;}

    /*!
   * @brief changes TR center pointer to best point
   * considering lower and upper bounds on variables.
   */
    void moveToBestPoint();

    bool criticalityStep(double initial_criticality_radius);

    /*!
    * @brief gives the gradient of the model, calculated
    * in absolute coordinates in the current point.
    *
    * CG comment 1: In this work, the polynomial model is scaled
    * inside a ball of radius 1. Therefore, it has to be rescaled
    * to absolute coordinates.
    *
    * CG comment 2: Other criticality measures need to be considered
    * if constraint handling
    */
    VectorXd measureCriticality();

    ModelMatrix getModelMatrices(int m);

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
     *                  5 = new points found, returning for function evaluation.
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

    int changeTrCenter(VectorXd new_point, double fvalue);

    /*!
     * brief Approximately solves trust-region subproblem.
     *
     * @return Tuple with the trial point and the trial decrease in the objective function (fval_current-trial_fval)
     */
    std::tuple<VectorXd, double> solveTrSubproblem();


     /*!
    * @brief Try to add new point. Step was not accepted. Smaller effort in including point.
    * @param new_point Point to be added
    * @param new_fvalue Function value of corresponding point
    * @return Tuple with the trial point and the trial decrease in the objective function (fval_current-trial_fval)
    */
    int tryToAddPoint(VectorXd new_point, double new_fvalue);

   /*!
   * @brief Add a point to the trust-region model
   * @param new_point to be added
   * @param fvalue function value of the corresponding point
   * @param relative_pivot_threshold Relative pivot threshold
   * @return Exit flag: 1 if the points was successfully added, or 0 otherwise.
   */
    int addPoint(VectorXd new_point, double fvalue, double relative_pivot_threshold);

   /*!
   * @brief Exchange a point from the trust-region model with a new one.
   * @param new_point Point to be exchanged.
   * @param fvalue Function value of the corresponding point.
   * @param relative_pivot_threshold Relative pivot threshold
   * @return tuple<bool,int> where the boolean indicates whether the exchange was sucessful,
    * and the integer is the point index.
   */

   std::tuple<bool,int> exchangePoint(VectorXd new_point, double new_fvalue, double relative_pivot_threshold);

    /*!
    * brief Choose pivot polynomial
    * @param initial_i
    * @param final_i
    * @param tol tolerance
    * @return pivot value
    */
    std::tuple<double,bool> choosePivotPolynomial(int initial_i, int final_i, double tol);



  void computePolynomialModels();

    /*!
     * Old evaluateNewFvalues description (incorporate somewhere else):
   * @brief evaluates function values at the new points
   * @param new_points_abs new points in absolute coordinates
   * @return map in which the first element is a RowVectorXd with the new function values,
   * and the second element is a boolean indicating whether the function evaluations succeeded.
   */

    // Delete (replaced by similar support methods)
    // void setDim(int dim) { dim_ = dim; }
    // int getDim() { return dim_; }

    // Model methods
    bool isInitialized() const { return is_initialized_; }
    bool hasModelChanged() const { return model_changed_; }
    void setIsInitialized(bool s) { is_initialized_ = s; }
    void setModelChanged(bool s) { model_changed_ = s; }

    // Initialization cases
    bool areInitPointsComputed() const { return init_points_computed_; }
    void addInitializationCase(Case *c) { initialization_cases_.append(c); }

    int getSizeInitCases() {
        return (int)initialization_cases_.size(); }

    void addTempInitCase(Case *c) {
        temp_init_cases_.append(c); }

    void submitTempInitCases();

    void setAreInitPointsComputed(bool s) {
        init_points_computed_ = s; }

    // Improvement cases
    bool areImprovementPointsComputed() const { return impr_points_computed_; };
    void setAreImprovementPointsComputed(bool s) { impr_points_computed_ = s; }
    void addImprovementCase(Case *c) { improvement_cases_.append(c); }

    void addTempImprCase(Case *c) {
        temp_impr_cases_.append(c); }

    int getSizeImprCases() {
        return (int)improvement_cases_.size(); }

    int getSizeTempImprCases() {
        return (int)temp_impr_cases_.size(); }

    void submitTempImprCases();

    void setAreImprPointsComputed(bool s) {
        impr_points_computed_ = s; }

    bool hasOnlyOnePoint() const { return points_abs_.cols() < 2; }
    bool isImprovementNeeded() const { return needs_improvement_; }
    void setIsImprovementNeeded(bool s) { needs_improvement_ = s; }

    void clearImprovementCasesList();

    // Replacement cases
    bool areReplacementPointsComputed() const { return repl_points_computed_; };
    void setAreReplacementPointsComputed(bool s) { repl_points_computed_ = s; }
    void addReplacementCase(Case *c) { replacement_cases_.append(c); }
    bool isReplacementNeeded() const { return needs_replacement_; }
    void setIsReplacementNeeded(bool s) { needs_replacement_ = s; }
    void clearReplacementCasesList();

    void addTempReplCase(Case *c) {
      temp_repl_cases_.append(c); }

    int getSizeReplCases() {
      return (int)replacement_cases_.size(); }

    void submitTempReplCases();

    void setAreReplPointsComputed(bool s) {
      repl_points_computed_ = s; }

    /*!
    * @brief
    * @param
    * @return Returns the cases needed to initialize TRModel
    */
    QList<Case *> getInitializationCases() { return initialization_cases_; };

    QList<Case *> getImprovementCases() { return improvement_cases_;};

    QList<Case *> getReplacementCases() { return replacement_cases_;};

    /*!
    * @brief Method that attempts to initialize TRModel
    * Sets is_model_initialized_ = true if initialization successful
    * @param
    * @return
    */
    void submitInitializationCases(QList<Case *>);

    bool checkDataSize(string context);

    private:
    Settings::Optimizer *settings_;
    std::vector<Polynomial> pivot_polynomials_;
    std::vector<Polynomial> modeling_polynomials_;

    Matrix<double,Dynamic,Dynamic> all_points_;
    Matrix<double,Dynamic,Dynamic> points_abs_;
    Matrix<double,Dynamic,Dynamic> points_shifted_;
    Matrix<double,Dynamic,Dynamic> cached_points_;

    Matrix<double,Dynamic,Dynamic> points_shifted_temp_;

    RowVectorXd all_fvalues_;
    RowVectorXd fvalues_;

    Matrix<double,Dynamic,Dynamic> new_points_;
    RowVectorXd new_fvalues_;

    RowVectorXd cached_fvalues_;
    RowVectorXd pivot_values_;
    VectorXd index_vector_;
    VectorXd piv_order_;
    VectorXd distances_;
    VectorXd lb_;
    VectorXd ub_;

    Case *base_case_;
    CaseHandler *case_handler_;

    // improveModelNfp() saved variables
    Polynomial nfp_polynomial_;
    MatrixXd nfp_new_points_shifted_;
    RowVectorXd nfp_new_pivots_;

    VectorXd nfp_new_point_shifted_;
    VectorXd nfp_new_point_abs_;
    RowVectorXd nfp_new_fvalues_;
    bool nfp_point_found_ = false;

    vector<QUuid> pt_case_uuid_;

    double radius_;
    int tr_center_; //!<index of trust region center point in points_abs>
    int cache_max_;
    int dim_;

    // TRModel status properties
    bool is_initialized_;
    bool model_changed_;

    // Initialization points status properties
    bool init_points_computed_;
    QList<Case *> initialization_cases_;
    QList<Case *> temp_init_cases_;

    // chooseAndReplacePoints() saved variables
    Polynomial repl_polynomial_;
    MatrixXd repl_new_points_shifted_;
    RowVectorXd repl_new_pivots_;

    MatrixXd repl_new_point_shifted_;
    VectorXd repl_new_point_abs_;
    RowVectorXd repl_new_fvalues_;
    bool repl_point_found_ = false;

    vector<QUuid> repl_pt_case_uuid_;

    // Improvement points status properties
    bool impr_points_computed_;
    bool needs_improvement_;
    QList<Case *> improvement_cases_;
    QList<Case *> temp_impr_cases_;

    QHash<QUuid, Case *> improvement_cases_hash_;

    // Replacement points status properties
    bool repl_points_computed_;
    bool needs_replacement_;
    QList<Case *> replacement_cases_;
    QList<Case *> temp_repl_cases_;

    QHash<QUuid, Case *> replacement_cases_hash_;

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
     * @param point point participating in the normalization.
     * @return resulting polynomial.
     */
    Polynomial normalizePolynomial(
            int poly_i,
            VectorXd point);

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
    Polynomial shiftPolynomial(Polynomial polynomial, VectorXd s);

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

    /*!
    * @brief Find a new point using the trust region.
    * @param polynomial polynomial that approximates the function within the tr.
    * @param tr_center_point trust region center point.
    * @param radius_used radius of trust region.
    * @param bl lower bound to the tr.
    * @param bu upper bound to the tr.
    * @param pivot_threshold pivot threshold.
    * @return a map containing:
     * new points found,
     * new pivots and
     * a boolean indicating whether a point was found or not.
    */
    std::tuple<Eigen::MatrixXd, Eigen::RowVectorXd, bool> pointNew(
            Polynomial polynomial,
            Eigen::VectorXd tr_center_point,
            double radius,
            Eigen::VectorXd bl,
            Eigen::VectorXd bu,
            double pivot_threshold);


    /*!
     * @brief Minimizes the trust region.
     * @param polynomial polynomial that approximates the function within the tr.
     * @param tr_center_point trust region center point.
     * @param radius_used radius of trust region.
     * @param bl lower bound to the tr.
     * @param bu upper bound to the tr.
     * @return a map containing:
      * point that minimizes the tr,
      * minimum function value,
      * an exit flag with the result of the minimization problem.
     */
    std::tuple<Eigen::VectorXd, double, int> minimizeTr(
            Polynomial polynomial,
            Eigen::VectorXd x_tr_center,
            double radius,
            Eigen::VectorXd bl,
            Eigen::VectorXd bu);

};

}
}


#endif //FIELDOPT_TRUSTREGIONMODEL_H
