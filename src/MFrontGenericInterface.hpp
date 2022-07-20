/** \file MFrontGenericInterface.hpp
 */

/* This file is part of MoFEM.
 * MoFEM is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * MoFEM is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 * License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with MoFEM. If not, see <http://www.gnu.org/licenses/>. */
#pragma once

#ifndef __MFRONTGENERICINTERFACE_HPP__
#define __MFRONTGENERICINTERFACE_HPP__

struct MFrontGenericInterface : public GenericElementInterface {

  MoFEM::Interface &mField;
  MFrontGenericInterface(MoFEM::Interface &m_field) : mField(m_field) {}

  MoFEMErrorCode addElementFields() override { return 0; };
  MoFEMErrorCode createElements() override { return 0; };
  MoFEMErrorCode setOperators() override { return 0; };
  MoFEMErrorCode addElementsToDM(SmartPetscObj<DM> dm) override { return 0; };

  MoFEMErrorCode setupSolverJacobianTS(const TSType type) override {
    return 0;
  };
  MoFEMErrorCode setupSolverFunctionTS(const TSType type) override {
    return 0;
  };

  MoFEMErrorCode updateElementVariables() override { return 0; };
  MoFEMErrorCode postProcessElement(int step) override { return 0; };
};

#endif // __MFRONTGENERICINTERFACE_HPP__