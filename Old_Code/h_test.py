import unittest
import numpy as np 
from Old_Code.city_image import haversine_distance_claire, haversine_distance_ashton, haversine_distance_david

class Test_haversine(unittest.TestCase): 

    def test_haversine_small(self): 
        ann_arb = (42.279, -83.732)
        detroit = (42.331, -83.046)
        
        res_claire = haversine_distance_claire(ann_arb[0], ann_arb[1], detroit[0], detroit[1])
        res_ashton = haversine_distance_ashton(ann_arb[0], ann_arb[1], detroit[0], detroit[1])
        res_david = haversine_distance_david(ann_arb[0], ann_arb[1], detroit[0], detroit[1])

        correct_distance = 56.71

        self.assertAlmostEqual(res_claire, correct_distance, delta=1e-1, msg="The values are not close enough for david")
        # np.testing.assert_allclose(res_david, correct_distance, atol = 1e-1)
        self.assertAlmostEqual(res_ashton, correct_distance, delta=1e-1, msg="The values are not close enough for claire")
        # np.testing.assert_allclose(res_claire, correct_distance, atol = 1e-1)
        self.assertAlmostEqual(res_david, correct_distance, delta=1e-1, msg="The values are not close enough for ashton")
        # np.testing.assert_allclose(res_ashton, correct_distance, atol = 1e-1)

    def test_haversine_large(self): 
        los_angel = (34.0549, -118.2426)
        new_york = (40.7128, -74.0060)
        
        res_claire = haversine_distance_claire(los_angel[0], los_angel[1], new_york[0], new_york[1])
        res_ashton = haversine_distance_ashton(los_angel[0], los_angel[1], new_york[0], new_york[1])
        res_david = haversine_distance_david(los_angel[0], los_angel[1], new_york[0], new_york[1])

        correct_distance = 3935.54

        # np.testing.assert_allclose(res_david, correct_distance, atol = 1e-1)
        self.assertAlmostEqual(res_david, correct_distance, delta=1e-1, msg="The values are not close enough for david")
        # np.testing.assert_allclose(res_claire, correct_distance, atol = 1e-1)
        self.assertAlmostEqual(res_claire, correct_distance, delta=1e-1, msg="The values are not close enough for claire")
        # np.testing.assert_allclose(res_ashton, correct_distance, atol = 1e-1)
        self.assertAlmostEqual(res_ashton, correct_distance, delta=1e-1, msg="The values are not close enough for ashton")

    def test_haversine_large_meters(self): 
        los_angel = (34.0549, -118.2426)
        new_york = (40.7128, -74.0060)
        
        res_claire = haversine_distance_claire(los_angel[0], los_angel[1], new_york[0], new_york[1], True)
        res_ashton = haversine_distance_ashton(los_angel[0], los_angel[1], new_york[0], new_york[1], True)
        res_david = haversine_distance_david(los_angel[0], los_angel[1], new_york[0], new_york[1], True)

        # correct_distance = 3935540
        correct_distance = 3935.53 * 1000

        # np.testing.assert_allclose(res_david, correct_distance, atol = 1e-1)
        self.assertAlmostEqual(res_david, correct_distance, delta=1e-1, msg="The values are not close enough for david")

        # np.testing.assert_allclose(res_claire, correct_distance, atol = 1e-1)
        self.assertAlmostEqual(res_claire, correct_distance, delta=1e-1, msg="The values are not close enough for claire")

        # np.testing.assert_allclose(res_ashton, correct_distance, atol = 1e-1)
        self.assertAlmostEqual(res_ashton, correct_distance, delta=1e-1, msg="The values are not close enough for ashton")


if __name__ == '__main__': 
    unittest.main()
    


