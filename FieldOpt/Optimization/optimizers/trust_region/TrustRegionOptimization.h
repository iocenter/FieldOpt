/*********************************************************************
 Created by thiagols on 26.11.18
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

#ifndef FIELDOPT_TRUSTREGION_H
#define FIELDOPT_TRUSTREGION_H

#include "Optimization/optimizer.h"
#include "Optimization/optimizers/trust_region/TrustRegionModel.h"
#include <Eigen/Core>

//#include "Optimization/objective/objective.h"
//#include "Optimization/objective/NPV.h"

using namespace Eigen;

namespace Optimization {
namespace Optimizers {

/*!
 * @brief This class is an implementation of a Trust Region (TR)
 * algorithm for derivative-free optimization
 *
 * \todo Derivative-free Optimization algorithm
 * \todo TR model
 * \todo TR sub-problem optimization
 * \todo Implement tests
 */
class TrustRegionOptimization : public Optimizer {
 public:
    TerminationCondition IsFinished() override;

  TrustRegionOptimization(
                Settings::Optimizer *settings,
                Case *base_case,
                Model::Properties::VariablePropertyContainer *variables,
                Reservoir::Grid::Grid *grid,
                Logger *logger,
                CaseHandler *case_handler=0,
                Constraints::ConstraintHandler *constraint_handler=0
    );

    TrustRegionModel* getTrustRegionModel() { return tr_model_; };

    int GetNumInitPoints() { return n_initial_points_; };

 protected:
    void handleEvaluatedCase(Case *c) override;

    void iterate() override;

 private:
    Matrix<double, Dynamic, Dynamic> initial_points_;
    RowVectorXd initial_fvalues_;
    VectorXd lb_, ub_; //!< Upper and lower bounds
    TrustRegionModel *tr_model_;
    int n_initial_points_;

    Settings::Optimizer *settings_;
    Model::Properties::VariablePropertyContainer *variables_;
    Case *base_case_;
    CaseHandler *case_handler_;

    void computeInitialPoints();

    void setLowerUpperBounds();
    void projectToBounds(VectorXd *point);

    class ConfigurationSummary : public Loggable {
    public:
        ConfigurationSummary(TrustRegionOptimization *opt) { opt_ = opt; }
        LogTarget GetLogTarget() override;
        QUuid GetId() override;
    private:
        TrustRegionOptimization *opt_;
    };
};

}
}
#endif //FIELDOPT_TRUSTREGIONOPTIMIZATION_H
