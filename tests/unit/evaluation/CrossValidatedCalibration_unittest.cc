#include "environments/MultiLabelTestEnvironment.h"
#include <shogun/evaluation/MulticlassAccuracy.h>
#include <shogun/features/DataGenerator.h>
#include <shogun/features/DenseFeatures.h>
#include <gtest/gtest.h>
#include <shogun/multiclass/MulticlassOCAS.h>
#include <shogun/evaluation/CrossValidatedCalibration.h>
#include <shogun/evaluation/SigmoidCalibrationMethod.h>
#include <shogun/evaluation/StratifiedCrossValidationSplitting.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/evaluation/MulticlassAccuracy.h>
#include <shogun/distance/EuclideanDistance.h>

using namespace shogun;

extern MultiLabelTestEnvironment* multilabel_test_env;

#ifdef HAVE_LAPACK
TEST(CrossValidatedCalibrationTest, check_probability_sum)
{
  CMath::init_random(5);
  float64_t C = 1.0;
  std::shared_ptr<GaussianCheckerboard> mockData =
	  multilabel_test_env->getMulticlassFixture();

  CDenseFeatures<float64_t>* train_feats = mockData->get_features_train();
  CDenseFeatures<float64_t>* test_feats = mockData->get_features_test();
  CMulticlassLabels* ground_truth =
	  (CMulticlassLabels*)mockData->get_labels_test();
    index_t n_folds=4;
  CStratifiedCrossValidationSplitting* splitting=
      new CStratifiedCrossValidationSplitting(ground_truth, n_folds);
  CMulticlassOCAS* mocas = new CMulticlassOCAS(C, train_feats, ground_truth);
  mocas->set_epsilon(1e-5);
  mocas->parallel->set_num_threads(1);

  CCrossValidatedCalibration* cross=new CCrossValidatedCalibration(mocas, ground_truth,
      splitting, new CSigmoidCalibrationMethod);
  
  cross->train(train_feats);

  CMulticlassLabels* pred = (CMulticlassLabels*)cross->apply(test_feats);

  index_t num_classes = pred->get_num_classes();

  SGVector<float64_t> confidence_sums;
  for (index_t i=0; i<num_classes; ++i)
  {
    SGVector<float64_t> scores = pred->get_multiclass_confidences(i);
    if (i==0)
      confidence_sums = scores;
    else 
      confidence_sums += scores;
  }

  //the sum of predicted probabilities for one sample should be one
  for (index_t i=0; i<confidence_sums.vlen; ++i)
  {
    EXPECT_EQ(confidence_sums[i], 1);
  }

  SG_UNREF(pred);
  SG_UNREF(cross);
}
#endif // HAVE_LAPACK