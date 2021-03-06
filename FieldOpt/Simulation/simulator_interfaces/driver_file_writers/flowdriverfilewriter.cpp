#include <simulator_interfaces/driver_file_writers/driver_parts/ecl_driver_parts/welspecs.h>
#include <simulator_interfaces/driver_file_writers/driver_parts/ecl_driver_parts/compdat.h>
#include <simulator_interfaces/driver_file_writers/driver_parts/ecl_driver_parts/wellcontrols.h>
#include <Utilities/filehandling.hpp>
#include "flowdriverfilewriter.h"

namespace Simulation {
FlowDriverFileWriter::FlowDriverFileWriter(::Settings::Settings *settings,
                                           Model::Model *model)
{
    model_ = model;
    settings_ = settings;
}

void FlowDriverFileWriter::WriteDriverFile(QString output_dir) {
    auto welspecs = ECLDriverParts::Welspecs(model_->wells());
    auto compdat = ECLDriverParts::Compdat(model_->wells());
    auto wellcontrols = ECLDriverParts::WellControls(model_->wells(), settings_->model()->control_times());
    model_->SetCompdatString(compdat.GetPartString());

    if (!Utilities::FileHandling::FileExists(output_dir+"/include/wells.in")
        || !Utilities::FileHandling::FileExists(output_dir+"/include/welspecs.in"))
        throw std::runtime_error("Unable to find include/wells.in or include/welspecs.in file to write to.");
    else Utilities::FileHandling::WriteStringToFile(welspecs.GetPartString(), output_dir+"/include/welspecs.in");

    if (!Utilities::FileHandling::FileExists(output_dir+"/include/compdat.in"))
        throw std::runtime_error("Unable to find include/compdat.in file to write to.");
    else Utilities::FileHandling::WriteStringToFile(compdat.GetPartString(), output_dir+"/include/compdat.in");

    if (!Utilities::FileHandling::FileExists(output_dir+"/include/controls.in"))
        throw std::runtime_error("Unable to find include/controls.in file to write to.");
    else Utilities::FileHandling::WriteStringToFile(wellcontrols.GetPartString(), output_dir+"/include/controls.in");
}

QString FlowDriverFileWriter::GetCompdatString() {
    return ECLDriverParts::Compdat(model_->wells()).GetPartString();
}

}
