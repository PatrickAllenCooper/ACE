import unittest
from pathlib import Path
from delivery_attribution_batch import cells
from chambers_delivery_validation import partition


class StagingTests(unittest.TestCase):
    def test_matrix_complete_and_exclusive(self):
        z=cells('123',Path('/scratch/example'))
        self.assertEqual(len(z),40);self.assertEqual(len({c['path'] for c in z}),40)
        self.assertEqual(sum(c['arm']=='scm' for c in z),18)
        self.assertEqual(sum(c.get('matched_cpu',False) for c in z),1)

    def test_duplicate_commands_and_blocks_never_split(self):
        test,blocks=partition([[-89,-89],[-88,-88],[-89,-89],[89,89]])
        self.assertEqual(test[0],test[1]);self.assertEqual(test[0],test[2])
        self.assertTrue((blocks[0]==blocks[1]).all())


if __name__=='__main__':unittest.main()
