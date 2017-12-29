/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2012 - 2013 Heiko Strathmann
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 * FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are
 * those
 * of the authors and should not be interpreted as representing official
 * policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#ifndef _CALIBRATION_H__
#define _CALIBRATION_H__

#include <shogun/lib/config.h>

#include <shogun/evaluation/CalibrationMethod.h>
#include <shogun/machine/Machine.h>

namespace shogun
{

	class CCalibration : public CMachine
	{
	public:
		/** constructor
		 */
		CCalibration();

		/** destructor
		*/
		virtual ~CCalibration();

		/** get name
		 *
		 * @return Calibration
		 */
		virtual const char* get_name() const
		{
			return "Calibration";
		}

		/** returns problem type of the machine to be calibrated
		 * @return problem type
		 */
		virtual EProblemType get_machine_problem_type() const;

		/** calibrate machine using given data
		* @param data on which to calibrate the machine
		* @return whether calibration was successful
		*/
		virtual bool train(CFeatures* data = NULL);

		/** calibrate locked machine 
		* @param subset_indices on which to calibrate the machine
		* @return whether calibration was successful
		*/
		virtual bool train_locked(SGVector<index_t> subset_indices);

		/** returns calibrated predictions 
		* @param features on which to apply the machine
		* @return binary labels
		*/
		virtual CBinaryLabels* apply_binary(CFeatures* features);

		/** returns calibrated multiclass predictions
		* @param features on which to apply machine
		* @return multiclass labels
		*/
		virtual CMulticlassLabels* apply_multiclass(CFeatures* features);

		/** returns calibrated multiclass predictions for a locked machine
		* @param subset_indices on which to apply machine
		* @return multiclass labels
		*/
		virtual CMulticlassLabels*
		apply_locked_multiclass(SGVector<index_t> subset_indices);

		/** returns calibrated predictions for a locked machine
		* @param features on which to apply the machine
		* @return binary labels
		*/
		virtual CBinaryLabels*
		apply_locked_binary(SGVector<index_t> subset_indices);

		/** returns current learning machine
		* @return learning machine
		*/
		virtual CMachine* get_machine();

		/** set learning machine
		* @param machine the learning machine to be calibrated
		*/
		virtual void set_machine(CMachine* machine);

		/** set method to be used for calibration
		* @param calibration_method the calibration method to be used
		*/
		virtual void
		set_calibration_method(CCalibrationMethod* calibration_method);

		/** get current calibration method
		* @return calibration method
		*/
		virtual CCalibrationMethod* get_calibration_method();

	private:
		/** helper method to get machine result for given data
		* @param features on which to apply the machine
		* @return predicted labels
		*/
		CLabels* apply_once(CFeatures* features);

		/** helper method to get machine result for given data
		* @param subset_indices on which to apply the machine
		* @return predicted labels
		*/
		CLabels* apply_once(SGVector<index_t> subset_indices);

		/** helper method to train both unlocked and locked machines
		* @param training_data on which to train the machine
		* @return whether training was successful
		*/
		template <typename T>
		bool train_calibration_machine(T training_data);

		/** helper method to get calibrated multiclass labels
		* @param result_labels predicted labels to calibrate
		* @param num_calibration_machines number of calibration method instances
		* @return calibrated multiclass labels
		*/
		CMulticlassLabels* get_multiclass_result(
		    CMulticlassLabels* result_labels, index_t num_calibration_machines);

		/** helper method to get calibrated binary labels
		* @param data on which to get calibrated predictions
		* @return calibrated labels
		*/
		template <typename T>
		CBinaryLabels* get_binary_result(T data);

		/** initialize variables and register them
		*/
		void init();

		/** helper method to train locked machine on given indices
		* @param subset_indices on which to train the machine
		* @return whether training was successful
		*/
		bool train_one_machine(SGVector<index_t> subset_indices);

		/** helper method to train machine on given features
		* @param features on which to train the machine
		* @return whether training was successful
		*/
		bool train_one_machine(CFeatures* features);

	private:
		/** learning machine to be calibrated */
		CMachine* m_machine;
		/** array of calibration method instances */
		CDynamicObjectArray* m_calibration_machines;
		/** calibration method */
		CCalibrationMethod* m_method;
	};
}
#endif