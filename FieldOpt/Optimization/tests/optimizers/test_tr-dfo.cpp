//
// Created by bellout on 11/27/18.
//

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

            FindVarSequence(prob);

            // Set up base case using dummy var containter
            test_case_tr_dfo_probs_ = new Optimization::Case(
                    QHash<QUuid, bool>(), QHash<QUuid, int>(),
                    varcont_tr_dfo_probs_->GetContinousVariableValues());

            // Use initial point from Matlab data
            test_case_tr_dfo_probs_->set_objective_function_value(tr_dfo_prob(x0));

            // cout << "SetUpOptimizer" << endl;
            tr_dfo_ = new TrustRegionOptimization(
                    settings_tr_opt_max_,
                    test_case_tr_dfo_probs_,
                    varcont_tr_dfo_probs_,
                    grid_5spot_,
                    logger_);
        }

        void CheckSameX(VectorXd xa, VectorXd xb,
                        vector<int> idx, double tol,
                        std::string msg) {

          VectorXd xbr(xb.size());
          for (int ii=0; ii<xb.rows(); ii++) {
            xbr.row(ii) << xb(idx[ii]);
          }
          // cout << "xa: [" << xa.transpose() << "]" << endl; // Matlab tr model data
          // cout << "xb: [" << xb.transpose() << "]" << endl; // FO tr model data
          // cout << "xbr: [" << xbr.transpose() << "]" << endl; // FO tr model data (reordered)

          ASSERT_TRUE(xa.isApprox(xbr,tol));

          if( ~msg.empty() ) {
            stringstream ss; ss << "[          ] " << FMAGENTA;
            cout << ss.str() << msg.c_str() << " -> ok" << END << endl;
          }
        }

        void FindVarSequence(TestResources::TrustRegionModelData::prob &prob) {

          // First vector (Eigen + std formats)
          VectorXd va = prob.xm.col(0); // <- va: VectorXd
          vector<double> v1;
          v1.resize(va.size());
          VectorXd::Map(&v1[0], va.size()) = va; // <- v1: std

          // Second vector (Eigen + std formats)
          // QHash<QUuid,double> vbu = varcont_tr_dfo_probs_->GetContinousVariableValues();
          list<double> vbl = varcont_tr_dfo_probs_->GetContinousVariableValues().values().toStdList();
          vector<double> v2{ std::begin(vbl), std::end(vbl) }; // <- v2: std
          Eigen::VectorXd vb = Eigen::Map<VectorXd>(v2.data(), v2.size()); // <- vb: VectorXd

          // Eigen approach
          double tol = 1e-6;
          VectorXi idxa(va.rows(),1);

          for (int ii=0; ii<va.size(); ii++) {
            for (int jj=0; jj<vb.size(); jj++) {
              if ( norm( va(ii) - vb(jj) ) < tol ) {
                idxa.row(ii) << jj;
              } else {
                continue;
              }
            }
          }
          // cout << "ia:" << endl << ia << endl; // dbg

          // std approah
          // vector<int> idx1; // <- defined as function param

          for (int ii=0; ii<v2.size(); ii++) {

            auto it =
                find_if(v1.begin(), v1.end(), [&](const double& x) {
                  return norm( x - v2[ii] ) < tol;
                } );

            prob.idx.push_back( (int)distance(v1.begin(), it) );
          }
          // dbg
          // cout << "idx:";
          // for (int ii=0; ii<idx1.size(); ii++) { cout << idx1[ii] << " "; }
          // cout << endl;

        } // End of void FindVarSequence

        void PrintCaseData(Optimization::Case &c,
                           TrustRegionOptimization &tr_dfo){

          // PRINT CASE DATA (ID, X, F)
          stringstream ss; ss << "[          ] " << FMAGENTA;
          cout << ss.str()
          << "---------------------------------------------------------" << END << endl;
          cout << ss.str() << "Case id: " << c.GetId().toString().toStdString() << END << endl;
          cout << ss.str() << "# initial points: " << tr_dfo.GetNumInitPoints() << END << endl;
          cout << ss.str() << "x: [" << c.GetRealVarVector().transpose() << "]" << END << endl;
          cout << ss.str() << "f: [" << c.objective_function_value() << "]" << END << endl;

        }

        void OverrideSecondPoint(TestResources::TrustRegionModelData::prob &prob,
                                 Optimization::Case &c) {

            VectorXd xbr(prob.xm.rows(),1);
            for (int ii=0; ii<prob.xm.rows(); ii++) {
              xbr.row(ii) << prob.xm(prob.idx[ii],1);
            }

            c.SetRealVarValues(xbr.col(0));
            c.set_objective_function_value(prob.fm(0,1));
        }



        void RunnerSubs(TestResources::TrustRegionModelData::prob prob,
                        double (*tr_dfo_prob)(VectorXd xs)){

            stringstream ss; ss << "[          ] " << FMAGENTA;

            // Tentative best case should be equal to base case at this point
            PrintCaseData(*tr_dfo_->GetTentativeBestCase(), *tr_dfo_);
            CheckSameX(tr_dfo_->GetTentativeBestCase()->GetRealVarVector(),
                       prob.xm.col(0), prob.idx, 1e-6,
                       "Check 1st point equal");

            while (!tr_dfo_->IsFinished()) {

                // RUNNER CALL (START)
                auto next_case = tr_dfo_->GetCaseForEvaluation();

                // COMPUTE OBJ.FUNCTION VALUE FOR CASE
                next_case->set_objective_function_value(
                        tr_dfo_prob(next_case->GetRealVarVector()));

                if (tr_dfo_->GetNumInitPoints() == 1 ) {
                  OverrideSecondPoint(prob, *next_case);
                }

                // RUNNER CALL (FINISH)
                tr_dfo_->SubmitEvaluatedCase(next_case);

                // PRINT CASE DATA (ID, X, F)
                PrintCaseData(*next_case, *tr_dfo_);
                CheckSameX(next_case->GetRealVarVector(),
                           prob.xm.col(1), prob.idx, 1e-6,
                           "Check 2nd point equal");

            }

            cout << ss.str() << "----------------------" << END << endl;
            auto best_case = tr_dfo_->GetTentativeBestCase();
            cout << best_case->objective_function_value() << endl;
        }

    };

    //  auto values = variables->GetContinousVariableValues();
    //  VectorXd test_vec = VectorXd::Zero(values.size());
    //  Case * init_case = new Case(base_case);
    //  init_case->SetRealVarValues(test_vec);
    //  case_handler_->AddNewCase(init_case);

    TEST_F(TrustRegionTest, tr_dfo_prob1) {
        cout << FMAGENTA
             << "[ CG.prob1 ] f = @(x) (1 - x(1))^2; x0=[-1.2 2.0]"
             << END << endl;
        SetUpOptimizer(tr_mdata.prob1, tr_dfo_prob1);
        RunnerSubs(tr_mdata.prob1, tr_dfo_prob1);
    }

    TEST_F(TrustRegionTest, tr_dfo_prob2) {
        cout << FMAGENTA
             << "[ CG.prob2 ] f = @(x) log1p(x(1)^2) + x(2)^2; x0=[2.0 2.0]"
             << END << endl;
        SetUpOptimizer(tr_mdata.prob2, tr_dfo_prob2);
        RunnerSubs(tr_mdata.prob2, tr_dfo_prob2);
    }

    TEST_F(TrustRegionTest, tr_dfo_prob3) {
        cout << FMAGENTA
             << "[ CG.prob3 ] f = @(x) sin(pi*x(1)/12) * cos(pi*x(2)/16); x0=[0.0 0.0]"
             << END << endl;
        SetUpOptimizer(tr_mdata.prob3, tr_dfo_prob3);
        RunnerSubs(tr_mdata.prob3, tr_dfo_prob3);
    }

    TEST_F(TrustRegionTest, tr_dfo_initial_model_created) {
        bool is_model_present = false;
        SetUpOptimizer(tr_mdata.prob1, tr_dfo_prob1);

        // RUNNER CALL (START)
        auto next_case = tr_dfo_->GetCaseForEvaluation();

        next_case->set_objective_function_value(
                tr_dfo_prob1(next_case->GetRealVarVector()));

        cout << "----------------------" << END << endl;
        cout << "Case id:" << next_case << END << endl;
        cout << "x:" << next_case->GetRealVarVector().transpose() << END << endl;
        cout << "f:" << next_case->objective_function_value() << END << endl;

        // RUNNER CALL (FINISH)
        tr_dfo_->SubmitEvaluatedCase(next_case);

        auto tr_model = tr_dfo_->getTrustRegionModel();

        if (tr_model) {
            if (tr_model->getDimension() >=2) { //<!at least 2 points are needed in the
                is_model_present = true;
            } else {
                cout << "Not enough points in the model." << endl;
            }
        } else {
            cout << "model not created" << endl;
        }
        EXPECT_TRUE(is_model_present);
    }
}
