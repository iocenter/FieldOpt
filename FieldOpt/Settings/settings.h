/****************************************************************
 *
 * settings.h
 *
 * Created: 28.09.2015 2015 by einar
 *
 * This file is part of the FieldOpt project.
 *
 * Copyright (C) 2015-2015
 * Einar J.M. Baumann <einar.baumann@ntnu.no>
 *
 * This program is free software; you can redistribute it
 * and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be
 * useful, but WITHOUT ANY WARRANTY; without even the implied
 * warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 * PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public
 * License along with this program; if not, write to the Free
 * Software Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA  02110-1301 USA
 *
 ****************************************************************/

// ---------------------------------------------------------
#ifndef SETTINGS_H
#define SETTINGS_H

// ---------------------------------------------------------
// STD / QT
#include <QString>
#include <QJsonObject>
#include <vector>

// ---------------------------------------------------------
#include "simulator.h"
#include "optimizer.h"
#include "optimization.h"
#include "model.h"
#include "Utilities/colors.hpp"

// ---------------------------------------------------------
namespace Settings {

// ---------------------------------------------------------
using std::vector;

// ---------------------------------------------------------
class Simulator;
class Model;
class Optimizer;
class Optimization;

// =========================================================
/*!
 * \brief The Settings class contains both general settings
 * for a FieldOpt run and pointers to objects containing
 * specific settings for the Model, Simulator and Optimizer.
 * Settings takes as input the path to a "driver file" in
 * the JSON format.
 *
 * \todo Remove bool verbose_, and associated
 * functions, since this functionality is now
 * handled by int verbosity_level_ in runtime_settings
 */
class Settings
{
 public:
  // -------------------------------------------------------
  Settings(){}
  Settings(QString driver_path,
           QString output_directory,
           vector<int> verb_vector);

  // -------------------------------------------------------
  QString driver_path() const { return driver_path_; }

  // -------------------------------------------------------
  // The name to be used for the run. Output file
  // and folder names are derived from this.
  QString name() const { return name_; }

  // -------------------------------------------------------
  // Simulation include directory
  QString sim_incl_dir_path() const { return sim_include_dir_path_; }

  // -------------------------------------------------------
  // Path to directory in which output files are to be placed.
  QString output_directory() const { return output_directory_; }

  // -------------------------------------------------------
  // Get the value for the bookkeeper tolerance.
  // Used by the Bookkeeper in the Runner library.
  double bookkeeper_tolerance() const
  { return bookkeeper_tolerance_; }

  // -------------------------------------------------------
  QString runner_type() const { return runner_type_; }

  // -------------------------------------------------------
  // Object containing model specific settings.
  Model *model() const { return model_; }

  // Object containing optimizer specific settings.
  Optimizer *optimizer() const { return optimizer_; }

  // Object containing simulator specific settings.
  Simulator *simulator() const { return simulator_; }

  // -------------------------------------------------------
  // Get string containing CSV header and contents for log.
  QString GetLogCsvString() const;

  // -------------------------------------------------------
  // Get the to the FieldOpt build directory.
  const QString &build_path() const { return build_path_; }

  // Set path of FieldOpt build directory.
  void set_build_path(const QString &build_path);

 public:

  // -------------------------------------------------------
  QString output_directory_;
  vector<int> verb_vector_; // = vector<int>(11,0);

 private:
  // -------------------------------------------------------
  QString driver_path_;
  QJsonObject *json_driver_;
  QString name_;
  double bookkeeper_tolerance_;
  QString runner_type_;
  QString sim_include_dir_path_;
  QString build_path_;


  // -------------------------------------------------------
  Model *model_;
  Simulator *simulator_;
  Optimizer *optimizer_;
  Optimization *optimization_;

  // -------------------------------------------------------
  void readDriverFile();
  void readGlobalSection();
  void readSimulatorSection();
  void readModelSection();

  void readOptimizationSection();
  void readOptimizerSection();
};

}

#endif // SETTINGS_H
