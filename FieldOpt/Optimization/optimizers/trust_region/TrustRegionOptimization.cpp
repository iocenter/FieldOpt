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

#include <iostream>
#include <fstream>
#include <string>

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

    // Construct shell of TRModel, does not initialize
    // model, i.e., is_model_initialized_ = false
    tr_model_ = new TrustRegionModel(lb_, ub_, base_case_, settings_);

    // [1] Find 2nd point; make and add cases corresponding
    // to 1st & 2nd points to initialization_cases_ list
    // [2] Fields initial_points_ and initial_fvalues_ are first
    // established here using fields from base_case (thus, these
    // need to be removed from constructor above, where they play
    // no part)
    computeInitialPoints();

    // Initial values for algorithm parameters
    rho_ = 0;
    sum_rho_ = 0;
    sum_rho_sqr_ = 0;
    delay_reduction_ = 0;
    gamma_dec_ = settings_->parameters().tr_gamma_dec;

    // Log configuration
    if (enable_logging_) {
        logger_->AddEntry(new ConfigurationSummary(this));
    }

    cout << "iter        fval         rho      radius  pts" << endl;
}

void TrustRegionOptimization::iterate() {
    if (enable_logging_) {
      logger_->AddEntry(this);
    }

    if (iteration_ == 0) {

        if (!tr_model_->areInitPointsComputed() //_if-1
            && !tr_model_->isInitialized()) {
            auto init_cases = tr_model_->getInitializationCases();
            case_handler_->AddNewCases(init_cases);
            return;
        }

        if (tr_model_->areInitPointsComputed() //_if-2
            && !tr_model_->areImprovementPointsComputed()
            && !tr_model_->areReplacementPointsComputed()
            && !tr_model_->isInitialized()) {

            tr_model_->DBG_printSettingsData("");
            bool is_model_changed = tr_model_->rebuildModel();
            tr_model_->setModelChanged(is_model_changed);
            tr_model_->moveToBestPoint();
            tr_model_->computePolynomialModels();

            if (tr_model_->hasOnlyOnePoint()) {

                mchange_flag_ = tr_model_->ensureImprovement();
                auto improvement_cases = tr_model_->getImprovementCases(); //<improve model>
                auto replacement_cases = tr_model_->getReplacementCases(); //<replace point>
                // Might be 0 if point_found in improveModelNfp is false. We also need to handle exit_flag=3 or exit_flag=4

                if (tr_model_->isImprovementNeeded()
                    && !improvement_cases.size() == 0) {
                    case_handler_->AddNewCases(improvement_cases);
                    return;
                }

                if (tr_model_->isReplacementNeeded() && !replacement_cases.size() ==0) {
                    case_handler_->AddNewCases(replacement_cases);
                    return;
                }
            } else {
                tr_model_->setIsInitialized(true);
                iteration_++;
            }
        }

        if (tr_model_->areInitPointsComputed() //_if-3
            && tr_model_->isImprovementNeeded()
            && tr_model_->areImprovementPointsComputed()
            && !tr_model_->isInitialized()) {

          mchange_flag_ = tr_model_->ensureImprovement();

          if (!tr_model_->isImprovementNeeded()) {
              tr_model_->setIsInitialized(true);
              iteration_++;
          }
        }

        if (tr_model_->areInitPointsComputed() //_if-4
            && tr_model_->isReplacementNeeded()
            && tr_model_->areReplacementPointsComputed()
            && !tr_model_->isInitialized()) {

          mchange_flag_ = tr_model_->ensureImprovement();

          if (!tr_model_->isReplacementNeeded()) {
            tr_model_->setIsInitialized(true);
            iteration_++;
          }
        }
    }

    if(tr_model_->isInitialized()) {
      int iter_max = settings_->parameters().tr_iter_max;
      double tol_radius = settings_->parameters().tr_tol_radius;
      double eps_c = settings_->parameters().tr_eps_c;
      double tol_f = settings_->parameters().tr_tol_f;
//      double eta_1 = settings_->parameters().tr_eta_1;
//      double gamma_1 = gamma_dec_;
//      double gamma_2 = settings_->parameters().tr_gamma_inc;
//      double radius_max = settings_->parameters().tr_radius_max;
      double err_model = 0.0;

      auto fval_current = tr_model_->getCurrentFval();
      auto x_current = tr_model_->getCurrentPoint();

      if (improve_model_) {
        mchange_flag_ = tr_model_->ensureImprovement();
        improve_model_ = ensureImprovementPostProcessing();
      }
      
      if ((!tr_model_->isImprovementNeeded() //_if-5
      && !tr_model_->areImprovementPointsComputed())
      && (!tr_model_->isReplacementNeeded()
      && !tr_model_->areReplacementPointsComputed())) {

        if ((tr_model_->getRadius() < tol_radius)
            || (iteration_ == iter_max)) {

          tr_model_->DBG_printModelData("Iterate-Y");
          return; //end of the algorithm

        } else {
          //<Move among points that are part of the model>
          if (true || tr_model_->isLambdaPoised()) {
            tr_model_->DBG_printModelData("Move among model points");
            tr_model_->moveToBestPoint();
            tr_model_->computePolynomialModels();

            fval_current = tr_model_->getCurrentFval();
            x_current = tr_model_->getCurrentPoint();
            err_model = tr_model_->checkInterpolation();
          }

          //!<Criticality step -- if we are possibly close to the optimum>
          criticality_step_performed_ = false;
          auto model_criticality = tr_model_->measureCriticality();
          tr_model_->DBG_printVectorXd(model_criticality,
                                       "model_criticality @ iterate:  ", "% 10.3e ",
                                       tr_model_->DBG_fn_mdat_);

          if (model_criticality.norm() <= eps_c) {
            if (!criticality_step_execution_ongoing_) {
              criticality_init_radius_ = tr_model_->getRadius();
            }
            criticality_step_execution_ongoing_ = tr_model_->criticalityStep(criticality_init_radius_);
            if (criticality_step_execution_ongoing_) {
              ensureImprovementPostProcessing();
              return;
            }
            criticality_step_performed_ = true;
            if (model_criticality.norm() < tol_f) {
              Printer::ext_warn("Model criticality < tol_f.", "Optimization", "TrustRegionOptimization");
              return;
            }
          } else {
	          criticality_step_execution_ongoing_ = false;
          }

          iteration_model_fl_ = tr_model_->isLambdaPoised();
          //!<Print summary>
          printIteration(fval_current);

          //!<Compute step>
          tie(trial_point_, predicted_red_) = tr_model_->solveTrSubproblem();

          trial_step_ = trial_point_ - x_current;

          if ((predicted_red_ < tol_radius * 1e-2) ||
              (predicted_red_ < tol_radius * abs(fval_current) &&
                  (trial_step_.norm()) < tol_radius) ||
              (predicted_red_ < tol_f * abs(fval_current) * 1e-3)) {

            rho_ = -std::numeric_limits<double>::infinity();
            mchange_flag_ = tr_model_->ensureImprovement();

            improve_model_ = ensureImprovementPostProcessing();
            tr_model_->DBG_printModelData("Iterate-X");

          } else {
            //!<Evaluate objective at trial point>
            Case *new_case = new Case(base_case_);
            new_case->SetRealVarValues(trial_point_);
            case_handler_->AddNewCase(new_case);
            return;
          }
        }
      } else {
        auto improvement_cases = tr_model_->getImprovementCases(); //<improve model>
        auto replacement_cases = tr_model_->getReplacementCases(); //<replace point>

        if ((tr_model_->isReplacementNeeded() && tr_model_->areReplacementPointsComputed()) ||
            (tr_model_->isImprovementNeeded() && tr_model_->areImprovementPointsComputed())) {


          if (!improvement_cases.size() == 0){
            for (Case* c: improvement_cases) {
              if (isImprovement(c)) {
                updateTentativeBestCase(c);
              }
            }
          }

          if (!replacement_cases.size() == 0){
            for (Case* c: replacement_cases) {
              if (isImprovement(c)) {
                updateTentativeBestCase(c);
              }
            }
          }

          //!<continue with model improvement
          mchange_flag_ = tr_model_->ensureImprovement();
          if ((mchange_flag_ ==1) || (mchange_flag_ == 2)) {
            iteration_--;
          }
          updateRadius();

          return;
        } else {
          if (tr_model_->isImprovementNeeded() && !improvement_cases.size() ==0) {
            case_handler_->AddNewCases(improvement_cases);
            return;
          }

          if (tr_model_->isReplacementNeeded() && !replacement_cases.size() == 0) {
            case_handler_->AddNewCases(replacement_cases);
            return;
          }
        }
      }
    }
    return;
}

void TrustRegionOptimization::printIteration(double fval_current) {
  stringstream ss;
  ss << setw(4) << right << iteration_ << setprecision(3)
     << setw(12) << scientific << right << fval_current
     << setw(12) << scientific << right << rho_
     << setw(12) << scientific << right << tr_model_->getRadius()
     << setw(5) << right << tr_model_->getNumPts();
  cout << ss.str() << endl;

  tr_model_->DBG_printToFile(tr_model_->DBG_fn_pivp_, "\n" + ss.str() + "\n");
}


void TrustRegionOptimization::handleEvaluatedCase(Case *c) {
    if (enable_logging_) {
      logger_->AddEntry(this);
    }

    if (iteration_ == 0) {

        // Collect init points
        if (!tr_model_->areInitPointsComputed()
           && !tr_model_->isInitialized()) {

            // Collect initialization case
            tr_model_->addTempInitCase(c);

            // All initialization cases have been evaluated
            if (case_handler_->QueuedCases().size() == 0
                && case_handler_->CasesBeingEvaluated().size() == 0) {

                    tr_model_->submitTempInitCases();
                    tr_model_->setAreInitPointsComputed(true);
            }
            return;

        } else if (tr_model_->areInitPointsComputed()
                   && !tr_model_->isInitialized()) {

            // Collect improvement case
            tr_model_->addTempImprCase(c);

            // All improvement cases have been evaluated
            if (case_handler_->QueuedCases().size() == 0
                && case_handler_->CasesBeingEvaluated().size() == 0) {

              if (tr_model_->isImprovementNeeded()) {
                tr_model_->submitTempImprCases();
                tr_model_->setAreImprPointsComputed(true);
              }

              if (tr_model_->isReplacementNeeded()) {
                tr_model_->submitTempReplCases();
                tr_model_->setAreReplPointsComputed(true);
              }
            }
            return;
        }

    } else { // Initialization is done; handling "normal" case
      if (tr_model_->isImprovementNeeded() || tr_model_->isReplacementNeeded()) {
        // All improvement cases have been evaluated
        if (tr_model_->isImprovementNeeded()) {
          // Collect case
          tr_model_->addTempImprCase(c);

          if (case_handler_->QueuedCases().size() == 0
              && case_handler_->CasesBeingEvaluated().size() == 0) {
            tr_model_->submitTempImprCases();
            tr_model_->setAreImprPointsComputed(true);
          }
        }

        if (tr_model_->isReplacementNeeded()) {
          tr_model_->addTempReplCase(c);

          if (case_handler_->QueuedCases().size() == 0
              && case_handler_->CasesBeingEvaluated().size() == 0) {
            tr_model_->submitTempReplCases();
            tr_model_->setAreReplPointsComputed(true);
          }
        }
      } else {
          VectorXd x_current = tr_model_->getCurrentPoint();
          double fval_current = tr_model_->getCurrentFval();
          double eta_1 = settings_->parameters().tr_eta_1;

          if (settings_->mode() == Settings::Optimizer::OptimizerMode::Maximize) {
            fval_trial_ = -c->objective_function_value();
          } else {
            fval_trial_ = c->objective_function_value();
          }

          //!<Actual reduction>
          ared_ = fval_current - fval_trial_;

          //!<Agreement factor>
          rho_ = ared_ / (predicted_red_);

          //!<Acceptance of the trial point>
          if (rho_ > eta_1) {

            //!<Successful iteration>
            //TODO: we are considering a minimization problem. The default is a maximization problem.
            if (isImprovement(c)) {
              updateTentativeBestCase(c);
            }

            fval_current = fval_trial_;
            x_current = trial_point_;

            //!<Including this new point as the TR center>
            mchange_flag_ = tr_model_->changeTrCenter(trial_point_, fval_trial_);

          } else {
            auto point_added = tr_model_->tryToAddPoint(trial_point_, fval_trial_);
            if (!point_added) {
              improve_model_ = true;
            }
          }
          sum_rho_ += rho_;
          sum_rho_sqr_ += pow(rho_, 2);

        if (!improve_model_) {
          updateRadius();
        }
      }
    }
}

void TrustRegionOptimization::updateRadius() {
  double tol_radius = settings_->parameters().tr_tol_radius;
  double eta_1 = settings_->parameters().tr_eta_1;
  double gamma_1 = gamma_dec_;
  double gamma_2 = settings_->parameters().tr_gamma_inc;
  double radius_max = settings_->parameters().tr_radius_max;

  //!<From time to time a step may end a bit outside the TR>
  auto step_size = min(tr_model_->getRadius(), trial_step_.lpNorm<Infinity>());

  //!<Radius update>
  if (rho_ > eta_1) {
    auto radius_inc = max(double(1), gamma_2 * (step_size / tr_model_->getRadius()));
    tr_model_->setRadius(min(radius_inc * tr_model_->getRadius(), radius_max));

  } else if (iteration_model_fl_
      && (rho_ == -std::numeric_limits<double>::infinity() || mchange_flag_ == 4 || criticality_step_performed_)) {
    //!<A good model should have provided a better point.
    //!< We reduce the radius, since the error is related to the radius
    //!<        | f(x) - m(x) | < K*(radius)^2
    //!<        rho == -inf -> too short step size
    //!<        mchange_flag == 4 -> Couldn't add point, had to rebuild model>
    if (tr_model_->getRadius() <= 2 * tol_radius / gamma_1) {
      delay_reduction_ = delay_reduction_ + 1;
    } else {
      delay_reduction_ = 0;
    }

    if (delay_reduction_ >= 3 || delay_reduction_ == 0 || criticality_step_performed_) {
      gamma_dec_ = gamma_1;
      tr_model_->setRadius(gamma_dec_ * tr_model_->getRadius());
      delay_reduction_ = 0;
    }
  }
  iteration_++;
}

void TrustRegionOptimization::computeInitialPoints() {

    tr_model_->setXDim(variables_->ContinousVariableSize());
    int n_cont_vars = tr_model_->getXDim();

    auto initial_point = base_case_->GetRealVarVector();

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

            //!<Append case corresponding to 2nd init point>
            Case *second_case = new Case(base_case_);
            second_case->SetRealVarValues(second_point);

            // Hack: Comment this line to test improveModelNfp()
            // -> currently, this leads to a crash
            // This will be done later in the test code
            tr_model_->addInitializationCase(second_case);

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

      lb_.conservativeResize(base_case_->GetRealVarIdVector().size());
      ub_.conservativeResize(base_case_->GetRealVarIdVector().size());
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

bool TrustRegionOptimization::ensureImprovementPostProcessing(){
  auto improvement_cases = tr_model_->getImprovementCases(); //<improve model>
  auto replacement_cases = tr_model_->getReplacementCases(); //<replace point>

//  cout << "criticality_step_execution_ongoing_A: " << criticality_step_execution_ongoing_ << endl;
  if (tr_model_->isImprovementNeeded() && !improvement_cases.size() == 0) {
    case_handler_->AddNewCases(improvement_cases);
    return true;
  }

  if (tr_model_->isReplacementNeeded() && !replacement_cases.size() == 0) {
    case_handler_->AddNewCases(replacement_cases);
    return true;
  }

  if (criticality_step_execution_ongoing_) {
    Printer::ext_warn(
          "criticality step did not generate a case.",
          "ensureImprovementPostProcessing", "TrustRegionOptimization");
  }

  if ((improvement_cases.size() == 0)
      && (replacement_cases.size() == 0)
      && !criticality_step_execution_ongoing_) {
    updateRadius();
    return false;
  }
}

Optimization::Optimizer::TerminationCondition
TrustRegionOptimization::IsFinished() {
    TerminationCondition tc = NOT_FINISHED;

    if (tr_model_->isInitialized() && !tr_model_->getModelingPolynomials().empty()) {

      auto model_criticality = tr_model_->measureCriticality();
      tr_model_->DBG_printVectorXd(model_criticality,
          "model_criticality @ isFinished:  ", "% 10.3e ",
          tr_model_->DBG_fn_mdat_);

      if (model_criticality.norm() < settings_->parameters().tr_tol_f) {
        tc =  OPTIMALITY_CRITERIA_REACHED;
      }
    }

    if (tr_model_->getRadius() < settings_->parameters().tr_tol_radius) {
      tc = MINIMUM_STEP_LENGTH_REACHED;
    }

    if (iteration_ == settings_->parameters().tr_iter_max) {
      tc = MAX_ITERATIONS_REACHED;
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
