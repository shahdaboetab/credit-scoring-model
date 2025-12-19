"""
Unit tests for data preprocessing functions.
"""

import unittest
import pandas as pd
import numpy as np
from src.data.preprocess import preprocess_data, split_data

class TestPreprocess(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.df = pd.DataFrame({
            'numerical1': np.random.randn(100),
            'numerical2': np.random.randint(1, 100, 100),
            'categorical1': np.random.choice(['A', 'B', 'C'], 100),
            'categorical2': np.random.choice(['X', 'Y'], 100),
            'target': np.random.randint(0, 2, 100)
        })

    def test_preprocess_data(self):
        """Test data preprocessing."""
        X_processed, y, preprocessor, feature_names = preprocess_data(self.df)

        # Check shapes
        self.assertEqual(X_processed.shape[0], 100)
        self.assertEqual(len(y), 100)

        # Check that preprocessor is fitted
        self.assertIsNotNone(preprocessor)

        # Check feature names
        self.assertIsInstance(feature_names, list)
        self.assertGreater(len(feature_names), 0)

    def test_split_data(self):
        """Test data splitting."""
        X_processed, y, _, _ = preprocess_data(self.df)
        X_train, X_test, y_train, y_test = split_data(pd.DataFrame(X_processed), y)

        # Check that data is split
        self.assertGreater(len(X_train), len(X_test))
        self.assertGreater(len(y_train), len(y_test))

        # Check that test size is approximately 0.2
        test_ratio = len(X_test) / len(X_processed)
        self.assertAlmostEqual(test_ratio, 0.2, delta=0.05)

if __name__ == '__main__':
    unittest.main()
