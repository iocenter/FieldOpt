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
    TrustRegionModel(const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>& initial_points, const Eigen::VectorXd& initial_fvalues , Settings::Optimizer *settings);
    int getDimension();

    void moveToBestPoint();
    void criticalityStep();
    double checkInterpolation();
    void rebuildModel();
    void improveModelNfp();
    void ensureImprovement();
    bool isLambdaPoised();
    void changeTrCenter(Eigen::VectorXd new_point, Eigen::VectorXd fvalues);
    std::map<Eigen::VectorXd, Eigen::VectorXd> solveTrSubproblem();
    void computePolynomialModels();


 private:
    Settings::Optimizer *settings_;
    std::vector<Polynomial> pivot_polynomials_;
    std::vector<Polynomial> modeling_polynomials_;
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> points_abs_;
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> points_shitfted_;
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> cached_points_;
    Eigen::VectorXd fvalues_;
    Eigen::VectorXd cached_fvalues_;
    Eigen::VectorXd index_vector_;
    Eigen::VectorXd distances_;
    Eigen::RowVector2d pivot_values_;
    double radius_;
    int tr_center_; //!<index of trust region center point in points_abs>
    int cache_max_;
    int dim_;

    void sortVectorByIndex(Eigen::VectorXd &vec, const Eigen::VectorXd &ind);
    void sortMatrixByIndex(Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> &points, const Eigen::VectorXd &ind);
    void nfpBasis(int dim);
    Polynomial matricesToPolynomial(int c0, const Eigen::VectorXd &g0, const Eigen::MatrixXd &H);
};

}
}


#endif //FIELDOPT_TRUSTREGIONMODEL_H
