/***********************************************************
 Created by Mathias C. Bellout on 27.11.18
 Copyright (C) 2018-2019 Mathias Bellout
 <mathias.bellout@petroleumcyberneticsgroup.no>
 <chakibbb-pcg@gmail.com>

 Modified 2018-2019 Thiago Lima Silva
 <thiagolims@gmail.com>

 This file is part of the FieldOpt project.

 FieldOpt is free software: you can redistribute it and/or
 modify it under the terms of the GNU General Public License
 as published by the Free Software Foundation, either version
 3 of the License, or (at your option) any later version.

 FieldOpt is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty
 of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See
 the GNU General Public License for more details.

 You should have received a copy of the
 GNU General Public License along with FieldOpt.
 If not, see <http://www.gnu.org/licenses/>.
***********************************************************/

#include <gtest/gtest.h>

// Define settings for tr-dfo
#include "Optimization/tests/test_resource_optimizer.h"
#include "Optimization/tests/test_resource_cases.h"

#include "Runner/tests/test_resource_runner.hpp"
#include "Reservoir/tests/test_resource_grids.h"
#include "Optimization/tests/test_resource_test_functions.h"

#include "Optimization/optimizers/trust_region/TrustRegionOptimization.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include "Utilities/math.hpp"
#include "Utilities/colors.hpp"
#include "Utilities/stringhelpers.hpp"

#include "test_tr-model-data.hpp"
#include "test_tr-support.hpp"

using namespace TestResources::TestFunctions;
using namespace Optimization::Optimizers;
using namespace std;

namespace {

    class TrustRegionTest : public ::testing::Test,
                            public TestResources::TestResourceOptimizer,
                            public TestResources::TestResourceGrids {

    protected:
        TrustRegionTest() {}

        virtual ~TrustRegionTest() {}
        virtual void SetUp() {}

        TrustRegionOptimization *tr_dfo_;
        Optimization::Case *test_case_tr_dfo_probs_;
        VariablePropertyContainer *varcont_tr_dfo_probs_;
        TestResources::TrustRegionModelData tr_mdata;

        void SetUpOptimizer(TestResources::TrustRegionModelData::prob &prob,
                            double (*tr_dfo_prob)(VectorXd xs)){

            VectorXd x0 = prob.xm.col(0);

            // Dummy var container based on initial point
            varcont_tr_dfo_probs_ = new VariablePropertyContainer();
            QString base_varname = "BHP#PRODUCER#"; // dummy var name

            for (int i = 0; i < x0.rows(); ++i) {
                // Use initial point values to construct container
                auto *prop = new ContinousProperty(x0(i));
                prop->setName(base_varname + QString::number(i));
                varcont_tr_dfo_probs_->AddVariable(prop);
            }

            // Set up base case using dummy var containter
            test_case_tr_dfo_probs_ = new Optimization::Case(
                    QHash<QUuid, bool>(), QHash<QUuid, int>(),
                    varcont_tr_dfo_probs_->GetContinousVariableValues());

            TestResources::FindVarSequence(prob,
                                           *test_case_tr_dfo_probs_);

            VectorXd ordered_vec(test_case_tr_dfo_probs_->GetRealVarVector().size());
            auto vec =test_case_tr_dfo_probs_->GetRealVarVector();
            for (int i = 0; i < prob.idx.size(); i++) {
                ordered_vec(prob.idx[i]) = vec(i);
            }
            test_case_tr_dfo_probs_->SetRealVarValues(ordered_vec);

            // Use initial point from Matlab data
            test_case_tr_dfo_probs_->set_objective_function_value(tr_dfo_prob(x0));
            settings_tr_opt_max_->setTRProbName(prob.name);

            tr_dfo_ = new TrustRegionOptimization(
                    settings_tr_opt_max_,
                    test_case_tr_dfo_probs_,
                    varcont_tr_dfo_probs_,
                    grid_5spot_,
                    logger_);

        }

        bool RunnerSubs(TestResources::TrustRegionModelData::prob prob,
                        double (*tr_dfo_prob)(VectorXd xs)){

            stringstream ss; ss << "[          ] " << FMAGENTA;
            double tol = 1e-06;
            int p_count = 0;

            while (tr_dfo_->IsFinished()
            == Optimization::Optimizer::TerminationCondition::NOT_FINISHED) {

                auto next_case = tr_dfo_->GetCaseForEvaluation();
                while (next_case == nullptr) {
                  if (tr_dfo_->IsFinished()) {
                  break;
                  } else {
                    next_case = tr_dfo_->GetCaseForEvaluation();
                  }
                }

                if (tr_dfo_->IsFinished()) {
                  break;
                }

                // Compute obj.function value for case
                next_case->set_objective_function_value(
                        tr_dfo_prob(next_case->GetRealVarVector()));

                // Override 2nd point
                if (p_count == 1 && prob.xm.cols() > 1) {
                    TestResources::OverrideSecondPoint(prob, *next_case);
                }

                // Test if current point is equal to matlab data -----
                if (p_count < 2) {

                  // Print case data (id, x, f)
                  TestResources::PrintCaseData(*next_case);

                  string msg = "Checking point " +
                      to_string(p_count + 1) + " is equal";

                  TestResources::CheckSameVector(
                      next_case->GetRealVarVector(),
                      prob.xm.col(p_count), tol, msg);
                }

                // Finish Runner
                tr_dfo_->SubmitEvaluatedCase(next_case);

                if (tr_dfo_->getTrustRegionModel()->isInitialized() && (p_count < 2)) {

                }
                p_count++;
            }

            stringstream sx;
            string cc;

            // ---------------------------------------------------
            if (tr_dfo_->IsFinished() == Optimization::Optimizer::
            TerminationCondition::OPTIMALITY_CRITERIA_REACHED) {
                cc = "OPTIMALITY_CRITERIA_REACHED";

            } else if (tr_dfo_->IsFinished() == Optimization::Optimizer::
            TerminationCondition::MINIMUM_STEP_LENGTH_REACHED) {
                cc = "MINIMUM_STEP_LENGTH_REACHED";

            } else if (tr_dfo_->IsFinished() == Optimization::Optimizer::
            TerminationCondition::MAX_ITERATIONS_REACHED) {
                cc = "MAX_ITERATIONS_REACHED";
            }

            sx << setw(12) << scientific << right << setprecision(6)
               << "---------------------------------------------" << endl
               << "x* = " << tr_dfo_->getTrustRegionModel()->getCurrentPoint().transpose() << endl
               << "f* = " << tr_dfo_->getTrustRegionModel()->getCurrentFval() << endl
               << "tc: " << cc.c_str();
            cout << sx.str();

            return true;
        }
      void TearDown() override {
	/*
        TrustRegionOptimization *tr_dfo_;
        Optimization::Case *test_case_tr_dfo_probs_;
        VariablePropertyContainer *varcont_tr_dfo_probs_;
        TestResources::TrustRegionModelData tr_mdata;
	 */
	delete(tr_dfo_);
	delete(test_case_tr_dfo_probs_);
	delete (varcont_tr_dfo_probs_);
      }
    };

    // TEST_F(TrustRegionTest, trHS1) {
    //     cout << endl << FMAGENTA << "[          ] =============="
    //          << "=========================================== " << endl
    //          << "[ HS1 ] "
    //          << " 100*pow(x(1) - pow(x(0), 2), 2) + pow(1 - x(0), 2)" << END << endl;

    //     SetUpOptimizer(tr_mdata.prob_hs1, hs1);
    //     EXPECT_TRUE(RunnerSubs(tr_mdata.prob_hs1, hs1));
    // }

//   TEST_F(TrustRegionTest, trDfoProb1) {
//       cout << endl << FMAGENTA << "[          ] =============="
//            << "=========================================== " << endl
//            << "[ CG.prob1 ] "
//            << " f = @(x) (1 - x(0))^2; x0=[-1.2 2.0]" << END << endl;
//
//       SetUpOptimizer(tr_mdata.prob1, tr_dfo_prob1);
//       EXPECT_TRUE(RunnerSubs(tr_mdata.prob1, tr_dfo_prob1));
//   }
//
//  TEST_F(TrustRegionTest, trDfoProb2) {
//      cout << endl << FMAGENTA << "[          ] =============="
//           << "=========================================== " << endl
//           << "[ CG.prob2 ] "
//           << "f = @(x) log1p(x(0)^2) + x(1)^2; x0=[2.0 2.0]"
//           << END << endl;
//
//      SetUpOptimizer(tr_mdata.prob2, tr_dfo_prob2);
//      EXPECT_TRUE(RunnerSubs(tr_mdata.prob2, tr_dfo_prob2));
//  }

//    TEST_F(TrustRegionTest, trDfoProb3) {
//        cout << endl << FMAGENTA << "[          ] =============="
//             << "=========================================== " << endl
//             << "[ CG.prob3 ] "
//             << "f = @(x) sin(pi*x(0)/12) * cos(pi*x(1)/16); x0=[0.0 0.0]"
//             << END << endl;
//
//        SetUpOptimizer(tr_mdata.prob3, tr_dfo_prob3);
//        EXPECT_TRUE(RunnerSubs(tr_mdata.prob3, tr_dfo_prob3));
//    }

//   TEST_F(TrustRegionTest, trDfoProb4) {
//       cout << endl << FMAGENTA << "[          ] =============="
//            << "=========================================== " << endl
//            << "[ CG.prob4 ] "
//            << "f = @(x) 0.01*(x(0) - 1)^2 + (x(1) - x(0)^2)^2; x0=[2.0 2.0 2.0]"
//            << END << endl;
//
//       // -------------------------------------------------------
//       SetUpOptimizer(tr_mdata.prob4, tr_dfo_prob4);
//       EXPECT_TRUE(RunnerSubs(tr_mdata.prob4, tr_dfo_prob4));
//   }

   TEST_F(TrustRegionTest, trDfoProb5) {
       cout << endl << FMAGENTA << "[          ] =============="
            << "=========================================== " << endl
            << "[ CG.prob5 ] "
            << "f = @(x) (x(0)-x(1))^2 + (x(1) - x(2))^4; x0=[-2.6 2.0 2.0]"
            << END << endl;

       // -------------------------------------------------------
       SetUpOptimizer(tr_mdata.prob5, tr_dfo_prob5);
       auto success_runnersubs = RunnerSubs(tr_mdata.prob5, tr_dfo_prob5);
       cout << "success runnersubs: " << success_runnersubs << endl;
       EXPECT_TRUE(true);
   }

//   TEST_F(TrustRegionTest, trDfoProb6) {
//       cout << endl << FMAGENTA << "[          ] =============="
//            << "=========================================== " << endl
//            << "[ CG.prob6 ] "
//            << "f = @(x) (x(0) + x(1))^2 + (x(1) + x(2))^2; x0=[-4.0 1.0 1.0]"
//            << END << endl;
//
//       // -------------------------------------------------------
//       SetUpOptimizer(tr_mdata.prob6, tr_dfo_prob6);
//       EXPECT_TRUE(RunnerSubs(tr_mdata.prob6, tr_dfo_prob6));
//   }
//
//   TEST_F(TrustRegionTest, trDfoProb7) {
//       cout << endl << FMAGENTA << "[          ] =============="
//            << "=========================================== " << endl
//            << "[ CG.prob7 ] "
//            << "f = @(x) log1p(x(0)^2) + log1p((x(0) - x(1))^2) " << endl
//            << " + log1p((x(1) - x(2))^2) + log1p((x(2) - x(3))^2); "
//               "x0=[2.0 2.0 2.0 2.0]"
//            << END << endl;
//
//       // -------------------------------------------------------
//       SetUpOptimizer(tr_mdata.prob7, tr_dfo_prob7);
//       EXPECT_TRUE(RunnerSubs(tr_mdata.prob7, tr_dfo_prob7));
//   }
//
//   TEST_F(TrustRegionTest, trDfoProb8) {
//       cout << endl << FMAGENTA << "[          ] =============="
//            << "=========================================== " << endl
//            << "[ CG.prob8 ] "
//            << "f = @(x) (x(0)*x(1)*x(2)*x(3))^2; x0=[0.8 0.8 0.8 0.8]"
//            << END << endl;
//
//       // -------------------------------------------------------
//       SetUpOptimizer(tr_mdata.prob8, tr_dfo_prob8);
//       EXPECT_TRUE(RunnerSubs(tr_mdata.prob8, tr_dfo_prob8));
//   }
//
//   TEST_F(TrustRegionTest, trDfoProb9) {
//       cout << endl << FMAGENTA << "[          ] =============="
//            << "=========================================== " << endl
//            << "[ CG.prob9 ] "
//            << "f = @(x) (x(0)-1)^2 + (x(1)-2)^2 + (x(2)-3)^2 + (x(3)-4)^2; x0=[1.0 1.0 1.0 1.0]"
//            << END << endl;
//
//       // -------------------------------------------------------
//       SetUpOptimizer(tr_mdata.prob9, tr_dfo_prob9);
//       EXPECT_TRUE(RunnerSubs(tr_mdata.prob9, tr_dfo_prob9));
//   }
//
//    TEST_F(TrustRegionTest, trDfoProb10) {
//        cout << endl << FMAGENTA << "[          ] =============="
//             << "=========================================== " << endl
//             << "[ CG.prob10 ] "
//             << "f = @(x) (x(0) - x(1))^2 + (x(1) - x(2))^2 + " << endl
//             << "(x(2) - x(3))^4 + (x(3) - x(4))^4; x0=[2.0 sqrt(2) -1.0 2-sqrt(2) 0.5]"
//             << END << endl;
//
//        // -------------------------------------------------------
//        SetUpOptimizer(tr_mdata.prob10, tr_dfo_prob10);
//        EXPECT_TRUE(RunnerSubs(tr_mdata.prob10, tr_dfo_prob10));
//    }
//
//   TEST_F(TrustRegionTest, trDfoProb11) {
//       cout << endl << FMAGENTA << "[          ] =============="
//            << "=========================================== " << endl
//            << "[ CG.prob11 ] "
//            << "f = @(x) sum(2*x./(x.*x + 1));; x0=[1.0 1.0 1.0 1.0]"
//            << END << endl;
//
//       // -------------------------------------------------------
//       SetUpOptimizer(tr_mdata.prob11, tr_dfo_prob11);
//       EXPECT_TRUE(RunnerSubs(tr_mdata.prob11, tr_dfo_prob11));
//   }

    // TEST_F(TrustRegionTest, trDfoInitialModelCreated) {
    //     bool is_model_present = false;
    //     SetUpOptimizer(tr_mdata.prob1, tr_dfo_prob1);

    //     // Compute obj.function value for case
    //     auto next_case = tr_dfo_->GetCaseForEvaluation();
    //     next_case->set_objective_function_value(
    //             tr_dfo_prob1(next_case->GetRealVarVector()));

    //     cout << "----------------------" << END << endl;
    //     cout << "Case id:" << next_case << END << endl;
    //     cout << "x:" << next_case->GetRealVarVector().transpose() << END << endl;
    //     cout << "f:" << next_case->objective_function_value() << END << endl;

    //     // Finish Runner
    //     tr_dfo_->SubmitEvaluatedCase(next_case);

    //     // Compute obj.function value for case
    //     auto scnd_case = tr_dfo_->GetCaseForEvaluation();
    //     TestResources::OverrideSecondPoint(tr_mdata.prob1, *next_case);
    //     scnd_case->set_objective_function_value(
    //             tr_dfo_prob1(scnd_case->GetRealVarVector()));

    //     cout << "----------------------" << END << endl;
    //     cout << "Case id:" << scnd_case << END << endl;
    //     cout << "x:" << scnd_case->GetRealVarVector().transpose() << END << endl;
    //     cout << "f:" << scnd_case->objective_function_value() << END << endl;

    //     // Finish Runner
    //     tr_dfo_->SubmitEvaluatedCase(scnd_case);

    //     auto tr_model = tr_dfo_->getTrustRegionModel();

    //     if (tr_model) {
    //         if (tr_model->getNumPts() >=2) { //<!at least 2 points are needed in the
    //             is_model_present = true;
    //         } else {
    //             cout << "Not enough points in the model." << endl;
    //         }
    //     } else {
    //         cout << "model not created" << endl;
    //     }
    //     EXPECT_TRUE(is_model_present);
    // }

    // TEST_F(TrustRegionTest, trDfoInitialRebuildModel) {
    //     bool is_model_present = false;
    //     SetUpOptimizer(tr_mdata.prob1, tr_dfo_prob1);

    //     // RUNNER CALL (START)
    //     auto next_case = tr_dfo_->GetCaseForEvaluation();

    //     // COMPUTE OBJ.FUNCTION VALUE FOR CASE
    //     next_case->set_objective_function_value(
    //             tr_dfo_prob1(next_case->GetRealVarVector()));

    //     if (tr_dfo_->GetNumInitPoints() == 1 ) {
    //         TestResources::OverrideSecondPoint(tr_mdata.prob1, *next_case);
    //     }

    //     // RUNNER CALL (FINISH)
    //     tr_dfo_->SubmitEvaluatedCase(next_case);

    //     auto tr_model = tr_dfo_->getTrustRegionModel();
    //     auto pivot_polynomials = tr_model->getPivotPolynomials();
    //     auto pivot_values = tr_model->getPivotValues();
    //     double tol = 1e-06;

    //     auto same_pivot_values =  true;
    //     for (int i=0; i<pivot_values.size(); i++) {
    //         if (abs(pivot_values(i) - tr_mdata.prob1.vm(i,0)) > tol) {
    //             same_pivot_values = false;
    //         }
    //     }
    //     auto same_pivot_coeff = true;
    //     for (int i=0; i<pivot_polynomials.size(); i++) {
    //         if (pivot_polynomials[i].dimension != tr_mdata.prob1.pdm(i)) {
    //             same_pivot_coeff = false;
    //         }

    //         for (int j=0; j<pivot_polynomials.size(); j++) {
    //             if (abs(pivot_polynomials[i].coefficients(j) -  tr_mdata.prob1.pcm(j,i)) > tol) {
    //                 same_pivot_coeff = false;
    //             }
    //         }
    //     }
    //     EXPECT_TRUE(same_pivot_values &&  same_pivot_coeff);
    // }


    // TEST_F(TrustRegionTest, trDfoInitialComputePolynomialModels) {
    //     bool is_model_present = false;
    //     SetUpOptimizer(tr_mdata.prob1, tr_dfo_prob1);

    //     // RUNNER CALL (START)
    //     auto next_case = tr_dfo_->GetCaseForEvaluation();

    //     // COMPUTE OBJ.FUNCTION VALUE FOR CASE
    //     next_case->set_objective_function_value(
    //             tr_dfo_prob1(next_case->GetRealVarVector()));

    //     if (tr_dfo_->GetNumInitPoints() == 1 ) {
    //         TestResources::OverrideSecondPoint(tr_mdata.prob1, *next_case);
    //     }

    //     // RUNNER CALL (FINISH)
    //     tr_dfo_->SubmitEvaluatedCase(next_case);

    //     auto tr_model = tr_dfo_->getTrustRegionModel();
    //     auto modeling_polynomials = tr_model->getModelingPolynomials();

    //     double tol = 1e-06;
    //     auto same_modeling_polynomials = true;

    //     for (int i=0; i<modeling_polynomials.size(); i++)  {
    //         if (modeling_polynomials[i].dimension != tr_mdata.prob1.mdm(i)) {
    //             same_modeling_polynomials = false;
    //         }

    //         for (int j=0; j<modeling_polynomials[i].coefficients.size(); j++) {
    //             if (abs(modeling_polynomials[i].coefficients(j) - tr_mdata.prob1.mcm(j,i)) > tol) {
    //                 same_modeling_polynomials = false;
    //             }
    //         }
    //     }
    //     EXPECT_TRUE(same_modeling_polynomials);
    // }

}
