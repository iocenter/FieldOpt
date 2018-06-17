//
// Created by morderbrage on 07.05.18.
//

#include "NPV.h"


#include <iostream>
#include <iomanip>
#include "weightedsum.h"
#include <stdlib.h>
#include <cmath>

using std::cout;
using std::endl;
using std::fixed;
using std::setprecision;
using std::scientific;

namespace Optimization {
namespace Objective {

NPV::NPV(Settings::Optimizer *settings,
         Simulation::Results::Results *results) {
    settings_ = settings;
    results_ = results;
    components_ = new QList<NPV::Component *>();
    auto report_time = results->GetValueVector(results->Time);

    for (int i = 0; i < settings->objective().NPV_sum.size(); ++i) {
        auto *comp = new NPV::Component();
        comp->property_name = settings->objective().NPV_sum.at(i).property;
        comp->property = results_->GetPropertyKeyFromString(comp->property_name);
        comp->coefficient = settings->objective().NPV_sum.at(i).coefficient;
        comp->time_step = settings->objective().NPV_sum.at(i).time_step;


        for (int j = 0; j < report_time.size(); j++) {
            if (abs(report_time.at(j) - settings->objective().NPV_sum.at(i).time_step) < 0.1) {
                comp->time_step = j;
                break;
            }
        }


        if (settings->objective().NPV_sum.at(i).is_well_prop) {
            comp->is_well_property = true;
            comp->well = settings->objective().NPV_sum.at(i).well;
        } else {
            comp->is_well_property = false;
            components_->append(comp);
        }
    }
}
double NPV::value() const
{
    double value = 0;

    auto report_times = results_->GetValueVector(results_->Time);
    for (int k = 0; k < components_->size(); ++k){
        for (int j = 0; j < report_times.size(); j++) {
            if (abs(report_times.at(j) - components_->at(k)->time_step) < 0.1) {
                components_->at(k)->time_step = j;
                cout << "Time_step :" << j << "    Report_time: " << report_times.at(j) << endl;
                break;
            }
        }
    }
    for (int i = 0; i < components_->size(); ++i) {
        value += components_->at(i)->resolveValue(results_);
        if (settings_->verb_vector()[5] > 1) { // idx:6 -> mod (Model)
            QString prop_name = components_->at(i)->property_name;
            double prop_coeff = components_->at(i)->coefficient;
            cout << scientific << setprecision(8);
            cout << "ObjFunctionProp[i=" << i << "]: -> (Coeff.) "
                 << prop_coeff << " * (" << prop_name.toStdString() << ") "
                 << results_->GetValue(results_->GetPropertyKeyFromString(prop_name))
                 << endl;
        }
    }
    return value;
}

double NPV::Component::resolveValue(Simulation::Results::Results *results)
{
    if (is_well_property) {
        if (time_step < 0) { // Final time step well property
            return coefficient * results->GetValue(property, well);
        }
        else { // Non-final time step well property
            return coefficient * results->GetValue(property, well, time_step);
        }
    }
    else {
        if (time_step < 0) { // Final time step field/misc property
            return coefficient * results->GetValue(property);
        }
        else { // Non-final time step field/misc property
            return coefficient * results->GetValue(property, time_step);
        }
    }
}

    }
}
