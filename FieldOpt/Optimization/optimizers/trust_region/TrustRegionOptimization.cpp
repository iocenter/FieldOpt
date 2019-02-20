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

#include <Utilities/verbosity.h>
#include "Utilities/printer.hpp"
#include "Utilities/stringhelpers.hpp"
#include "TrustRegionOptimization.h"

namespace Optimization {
namespace Optimizers {

TrustRegionOptimization::TrustRegionOptimization(
        Settings::Optimizer *settings,
        Case *base_case,
        Model::Properties::VariablePropertyContainer *variables,
        Reservoir::Grid::Grid *grid,
        Logger *logger,
        CaseHandler *case_handler,
        Constraints::ConstraintHandler *constraint_handler
) : Optimizer(settings, base_case, variables, grid, logger, case_handler, constraint_handler) {

    settings_ = settings;
    variables_ = variables;
    base_case_ = base_case;

    setLowerUpperBounds();

    if (enable_logging_) { // Log base case
        logger_->AddEntry(this);
    }

    n_initial_points_ = 1;

    // Construct shell of TRModel, does not initialize
    // model, i.e., is_model_initialized_ = false
    tr_model_ = new TrustRegionModel(// initial_points_,
                                     // initial_fvalues_,
                                     lb_, ub_, settings_);

    // [1] Find 2nd point; make and add cases corresponding
    // to 1st & 2nd points to initialization_cases_ list
    // [2] Fields initial_points_ and initial_fvalues_ are first
    // established here using fields from base_case (thus, these
    // need to be removed from constructor above, where they play
    // no part)
    computeInitialPoints();

    // Log configuration
    if (enable_logging_) {
        logger_->AddEntry(new ConfigurationSummary(this));
    }
}

void TrustRegionOptimization::iterate() {

    if (iteration_ == 0) {

        if (!tr_model_->areInitPointsComputed() && !tr_model_->isInitialized()) {
            cout << "Submit Case List to case_handler" << endl;
            auto init_cases = tr_model_->getInitializationCases();
            case_handler_->AddNewCases(init_cases);
            cout << "# of cases in case_handler: " << init_cases.size() << endl;
            return;
        }

        if (tr_model_->areInitPointsComputed()
            && !tr_model_->areImprovementPointsComputed()
            && !tr_model_->isInitialized()) {
            cout << "Initializing TRModel" << endl;

            tr_model_->setModelChanged(tr_model_->rebuildModel());
            tr_model_->moveToBestPoint();
            tr_model_->computePolynomialModels();

            if (tr_model_->hasOnlyOnePoint()) {
                cout << "Improving TRModel" << endl;

                tr_model_->ensureImprovement();
                auto improvement_cases = tr_model_->getImprovementCases();

                if(tr_model_->isImprovementNeeded()
                && !improvement_cases.size() == 0) {
                    case_handler_->AddNewCases(improvement_cases);
                } else if (improvement_cases.size() == 0) {
                  // # of improvement cases zero!
                }
            }
            return;
        }

        // Continue improvement model process
        if (tr_model_->isImprovementNeeded()
            && tr_model_->areImprovementPointsComputed()
            && !tr_model_->isInitialized()) {

                tr_model_->ensureImprovement();
            return;
        }

        // TRMmodel fully initialized
    } else {

        // do normal iteration stuff; add cases to queue
        iteration_++;
        return;
    }

    // Save for later
    //    Printer::ext_warn(
    //            "Insufficient # of points to create the Trust Region model.",
    //            "Optimization", "TrustRegionOptimization");
    //    throw std::runtime_error(
    //            "Failed to initialize Trust Region Optimizer.");

}

void TrustRegionOptimization::handleEvaluatedCase(Case *c) {

    if (iteration_ == 0) {

        // Collect init points
        if (!tr_model_->areInitPointsComputed() && !tr_model_->isInitialized()) {

            // Collect initialization case
            tr_model_->addTempInitCase(c);

            // All initialization cases have been evaluated
            if (case_handler_->QueuedCases().size() == 0
                && case_handler_->CasesBeingEvaluated().size() == 0) {

                    // points_abs_, fvalues_ fields are updated in submitTempInitCases()
                    tr_model_->submitTempInitCases();
                    tr_model_->setAreInitPointsComputed(true);
            }

            return;

        } else if (tr_model_->areInitPointsComputed() && !tr_model_->isInitialized()) {

            // Collect initialization case
            tr_model_->addTempImprCase(c);

            // All initialization cases have been evaluated
            if (case_handler_->QueuedCases().size() == 0
                && case_handler_->CasesBeingEvaluated().size() == 0) {

                tr_model_->submitTempImprCases();
                tr_model_->setAreImprPointsComputed(true);
            }

        }


    } else { // Initialization is done; handling "normal" case
        // do normal stuff
    }
}

//void TrustRegionOptimization::iterate() {
//    // TODO: implement the main iteration for the TR optimization algorithm
//
//    if (enable_logging_) {
//        logger_->AddEntry(this);
//    }
//
//    if ((n_initial_points_  < 1)
//        && (case_handler_->CasesBeingEvaluated().size() == 0)) {
//
//
//
//        // Just evaluated the second point, so we are ready
//        // to build the quadratic model for the TR
//    } else if (n_initial_points_ == 1 ) {
//
//        initial_points_.col(1) = c->GetRealVarVector();
//        initial_fvalues_(1) = c->objective_function_value();
//        n_initial_points_++;
//
//        // Construct TRModel
//        tr_model_ = new TrustRegionModel(initial_points_, initial_fvalues_,
//                                         lb_, ub_, settings_);
//
//        //TODO: improve trust region with the new point
//
//        if (isImprovement(c)) {
//            updateTentativeBestCase(c);
//            Printer::ext_info("Found new tentative best case: "
//                              + Printer::num2str(c->objective_function_value()),
//                              "Optimization", "Trust Region");
//        }
//    }
//}

void TrustRegionOptimization::computeInitialPoints() {

    tr_model_->setDim(variables_->ContinousVariableSize());
    int n_cont_vars = tr_model_->getDim();

    auto initial_point = base_case_->GetRealVarVector();

    cout << "Adding base case to Case List" << endl;
    tr_model_->addInitializationCase(base_case_);

    //!<Find another point since only one initial guess provided
    if (settings_->parameters().tr_init_guesses == -1) {
        if (settings_->parameters().tr_init_sampling_method == "Random") {

            //!<Find (random) 2nd initial point>
            auto rng = get_random_generator(settings_->parameters().rng_seed);

            VectorXd second_point(n_cont_vars, 1);
            second_point.setZero(n_cont_vars);
            for (int i = 0; i < n_cont_vars; ++i) {
                second_point(i) = random_double(rng, 0, 1);
            }

            //!<Second point must not be too close>
            while (second_point.lpNorm<Infinity>()
                    < settings_->parameters().tr_pivot_threshold) {
                second_point << 2*second_point.array();
            }

            second_point << (second_point.array() - 0.5);
            second_point *= settings_->parameters().tr_initial_radius;
            second_point << initial_point.array() + second_point.array();
            if(lb_.size()>0 && ub_.size() >0) {
                projectToBounds(&second_point);
            }

            //!<Establish initial point matrix (2 cols since only
            //!< two points are needed to build quad model for TR)>
            initial_points_.setZero(n_cont_vars, 2);
            initial_points_.col(0) = initial_point;

            //!<Establish initial feval matrix>
            initial_fvalues_.setZero(2);
            initial_fvalues_(0) = base_case_->objective_function_value();

            //!<Append case corresponding to 2nd init point>
            Case *second_case = new Case(base_case_);
            second_case->SetRealVarValues(second_point);
            tr_model_->addInitializationCase(second_case);
            cout << "Adding 2nd point to Case List" << endl;

            n_initial_points_++;

        } else {
            //TODO: implement other sampling method such as Uniform.
        }

    } else {
        //TODO: get other initial points from the parameters
    }
}

void TrustRegionOptimization::setLowerUpperBounds() {

  if (constraint_handler_->HasBoundaryConstraints()) {

    // Use constraint_handler lower/upper bounds
    lb_ = constraint_handler_->GetLowerBounds(
        base_case_->GetRealVarIdVector());
    ub_ = constraint_handler_->GetUpperBounds(
        base_case_->GetRealVarIdVector());

  } else {

    // Use lower/upper bounds specified in tr-params
    if (settings_->parameters().tr_lower_bound
        && settings_->parameters().tr_upper_bound) {

      lb_.resize(base_case_->GetRealVarIdVector().size());
      ub_.resize(base_case_->GetRealVarIdVector().size());
      lb_.fill(settings_->parameters().tr_lower_bound);
      ub_.fill(settings_->parameters().tr_upper_bound);

    } else {

      Printer::ext_warn(
          "Lower/upper bounds for DF-TR algorithm not specified.",
          "Optimization", "TrustRegionOptimization");
      throw std::runtime_error(
          "Lower/upper bounds for DF-TR algorithm not specified.");

    }
  }
}

Optimization::Optimizer::TerminationCondition
TrustRegionOptimization::IsFinished() {

    TerminationCondition tc = NOT_FINISHED;
    if (case_handler_->CasesBeingEvaluated().size() > 0) {
        return tc;
    }

    if (evaluated_cases_ > max_evaluations_) {
          tc = MAX_EVALS_REACHED;
    }

    if (tc != NOT_FINISHED) {
        if (enable_logging_) {
            logger_->AddEntry(this);
        }
    }
    return tc;
}

void TrustRegionOptimization::projectToBounds(VectorXd *point) {
    if (ub_.size() > 0) {
        *point = ub_.cwiseMin(*point);
    }
    if (lb_.size() > 0) {
        *point = lb_.cwiseMax(*point);
    }
}

Loggable::LogTarget TrustRegionOptimization::ConfigurationSummary::GetLogTarget() {
    return LOG_SUMMARY;
}

QUuid TrustRegionOptimization::ConfigurationSummary::GetId() {
    return QUuid(); // Null UUID
}

}
}
