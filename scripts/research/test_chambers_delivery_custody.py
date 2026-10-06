import tempfile
import unittest
from pathlib import Path
from chambers_delivery_validation import physics_features, validate_fit_seal
from runner_delivery_confirmation import write, sha


class PhysicalValidationTests(unittest.TestCase):
    def test_relative_angle_physics_has_rotation_symmetry(self):
        f=physics_features([[0,0],[30,30],[0,90],[45,-45],[0,45],[20,65]])
        self.assertAlmostEqual(f[0,1],1.)
        self.assertAlmostEqual(f[1,1],f[0,1])
        self.assertAlmostEqual(f[2,1],0.)
        self.assertAlmostEqual(f[3,1],f[2,1])
        self.assertAlmostEqual(f[4,1],.5)
        self.assertAlmostEqual(f[5,1],f[4,1])

    def test_auxiliary_model_and_protocol_tampering_prevents_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);p={'conditions':[f'c{i}.csv' for i in range(11)],'selected_delivery':'tree'}
            write(out/'protocol.json',p)
            write(out/'started.json',{'protocol_sha256':sha(out/'protocol.json')})
            summaries={}
            for name in p['conditions']:
                dest=out/Path(name).stem;dest.mkdir()
                for file in ('models.pt','linear_coefficients.json','predictions.npz','selected_regressor.pkl'):
                    (dest/file).write_bytes(b'analytical fixture; no archived responses')
                summaries[dest.name]={'rows':60,'train_rows':50,'test_rows':10,'train_variance':1.,
                                      'artifact_hashes':{file.name:sha(file) for file in dest.iterdir()}}
            write(out/'fit_seal.json',{'protocol_sha256':sha(out/'protocol.json'),'conditions':summaries})
            write(out/'fit_complete.json',{'seal_sha256':sha(out/'fit_seal.json')})
            _,verified=validate_fit_seal(out,p)
            self.assertEqual(len(verified),11)
            for file in ('linear_coefficients.json','selected_regressor.pkl'):
                artifact=out/'c0'/file;original=artifact.read_bytes();artifact.write_bytes(b'tampered')
                with self.assertRaises(ValueError):validate_fit_seal(out,p)
                artifact.write_bytes(original)
            write(out/'started.json',{'protocol_sha256':'tampered'})
            with self.assertRaises(ValueError):validate_fit_seal(out,p)


if __name__=='__main__':unittest.main()
