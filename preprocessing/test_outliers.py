import warnings

warnings.filterwarnings('ignore')
import unittest
from preprocessing.data_explorer import outliers_detector
import numpy as np

np.random.seed(0)

data_points = np.random.randn(2000, 2)
avg_codisp, is_outlier = outliers_detector(data_points)


class TestOutliers(unittest.TestCase):

    def test_outliers_score(self):
        self.assertEqual(round(avg_codisp.quantile(0.99), 4), 25.3697)


if __name__ == '__main__':
    unittest.main()
