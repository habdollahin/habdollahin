import unittest

from dvh_tool import dose_at_volume, dvh_curve, mean_dose, volume_at_dose

DOSES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


class TestDVH(unittest.TestCase):
    def test_mean_dose(self):
        self.assertEqual(mean_dose(DOSES), 55)

    def test_d_values(self):
        self.assertEqual(dose_at_volume(DOSES, 10), 100)  # hottest 10% = 1 voxel
        self.assertEqual(dose_at_volume(DOSES, 50), 60)
        self.assertEqual(dose_at_volume(DOSES, 100), 10)  # whole volume = the minimum

    def test_v_values(self):
        self.assertEqual(volume_at_dose(DOSES, 0), 100)
        self.assertEqual(volume_at_dose(DOSES, 55), 50)
        self.assertEqual(volume_at_dose(DOSES, 101), 0)

    def test_curve_starts_at_full_volume(self):
        self.assertEqual(dvh_curve(DOSES)[0], (0, 100))

    def test_empty_structure_is_rejected(self):
        with self.assertRaises(ValueError):
            mean_dose([])


if __name__ == "__main__":
    unittest.main()
