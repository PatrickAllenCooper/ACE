"""Artificial device-map records only. No CUDA, model, device I/O or runtime."""
import copy
import hashlib
from pathlib import Path
import stat
import unittest

import llm_intervention_gpu_mappings as g


def fixture():
    text="Character devices:\n  1 mem\n195 nvidia-frontend\n509 nvidia-uvm\n\nBlock devices:\n  8 sd\n"
    records={}
    for i,(name,major,minor) in enumerate((('/dev/nvidia0',195,0),('/dev/nvidiactl',195,255),
                                          ('/dev/nvidia-uvm',509,0),('/dev/zero',1,5))):
        records[name]={'uid':0,'mode':stat.S_IFCHR|0o666,'inode':100+i,
            'dev_major':0,'dev_minor':6,'rdev_major':major,'rdev_minor':minor}
    inventory={'schema':g.VERSION,'devices':records,'proc_devices':text,
               'proc_devices_sha256':hashlib.sha256(text.encode()).hexdigest()}
    maps='\n'.join(['1000-2000 rw-s 00000000 00:06 100 /dev/nvidia0',
        '2000-3000 r--s 00000000 00:06 101 /dev/nvidiactl',
        '3000-4000 -w-s 00000000 00:06 102 /dev/nvidia-uvm',
        '4000-5000 rw-s 00000000 00:01 900 /dev/zero (deleted)'])+'\n'
    return {'process_maps':maps,'mixed_native_paths':sorted(g.PATHS),'mixed_module_paths':['unchanged']},inventory


class GpuMappingTests(unittest.TestCase):
    def test_exact_four_types_and_raw_immutability(self):
        s,i=fixture();before=copy.deepcopy((s,i))
        view,evidence=g.checked_view(s,i,'after-model-load')
        self.assertEqual((s,i),before)
        self.assertEqual(view['mixed_native_paths'],[])
        self.assertEqual(view['mixed_module_paths'],['unchanged'])
        self.assertEqual(view['process_maps'],s['process_maps'])
        self.assertEqual(len(evidence['classified_rows']),4)
        self.assertFalse(evidence['gpu_assignment_proved'])

    def test_unknown_library_executable_private_deleted_gpu_rejected(self):
        s,i=fixture()
        for old,new in (('/dev/nvidia0','/dev/evil'),('1000-2000 rw-s','1000-2000 rwxp'),
                        ('1000-2000 rw-s','1000-2000 rw-p'),
                        ('100 /dev/nvidia0','100 /dev/nvidia0 (deleted)'),
                        ('00:06 100','00:07 100'),('00:06 100','00:06 999')):
            bad=copy.deepcopy(s);bad['process_maps']=bad['process_maps'].replace(old,new)
            with self.subTest(new=new),self.assertRaises(ValueError):g.checked_view(bad,i,'after-model-load')
        bad=copy.deepcopy(s);bad['mixed_native_paths']=sorted([*g.PATHS,'/tmp/untrusted.so'])
        with self.assertRaises(ValueError):g.checked_view(bad,i,'after-requests')

    def test_device_identity_and_driver_registry(self):
        s,i=fixture()
        for key,value in (('uid',1000),('mode',stat.S_IFREG|0o666),('mode',stat.S_IFLNK|0o777),
                          ('rdev_major',1),('rdev_minor',1),('uid',False)):
            bad=copy.deepcopy(i);bad['devices']['/dev/nvidia0'][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):g.checked_view(s,bad,'after-model-load')
        bad=copy.deepcopy(i);bad['devices']['/dev/nvidia-uvm']['rdev_major']=508
        with self.assertRaises(ValueError):g.checked_view(s,bad,'after-model-load')
        with self.assertRaises(ValueError):g.character_registry('Character devices:\n508 nvidia-uvm\n509 nvidia-uvm\n')

    def test_anonymous_zero_is_separate_from_current_device_inode(self):
        s,i=fixture();g.checked_view(s,i,'after-requests')
        for old,new in (('00:01 900','00:06 103'),('/dev/zero (deleted)','/dev/zero'),
                        ('4000-5000 rw-s','4000-5000 r--s'),
                        ('rw-s 00000000 00:01','rw-s 00001000 00:01'),('00:01 900','00:01 0')):
            bad=copy.deepcopy(s);bad['process_maps']=bad['process_maps'].replace(old,new)
            with self.subTest(new=new),self.assertRaises(ValueError):g.checked_view(bad,i,'after-model-load')

    def test_early_phase_or_missing_row_rejected(self):
        s,i=fixture()
        for phase in ('before-imports','after-service-imports','after-cpu-tokenization','unknown'):
            with self.assertRaises(ValueError):g.checked_view(s,i,phase)
        bad=copy.deepcopy(s);bad['process_maps']='\n'.join(bad['process_maps'].splitlines()[1:])
        with self.assertRaises(ValueError):g.checked_view(bad,i,'after-model-load')

    def test_malformed_map_rejected(self):
        for raw in ('broken','2000-1000 rw-s 0 00:06 1 /dev/nvidia0',
                    '1000-2000 abcd 0 00:06 1 /dev/nvidia0','1000-2000 rw-s x 00:06 1 /dev/nvidia0'):
            with self.assertRaises(ValueError):g.mapping_rows(raw)


if __name__=='__main__':unittest.main()
