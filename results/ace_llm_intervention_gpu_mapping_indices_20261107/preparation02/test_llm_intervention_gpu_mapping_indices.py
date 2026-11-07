"""Physical GPU indices on fabricated records, without device/model access."""
import copy
import hashlib
import io
import os
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import llm_intervention_gpu_mappings as g
import test_llm_intervention_gpu_mappings as fixtures


def indexed(index):
    snapshot, inventory = fixtures.fixture()
    name = '/dev/nvidia'+str(index)
    snapshot['process_maps'] = snapshot['process_maps'].replace('/dev/nvidia0', name)
    snapshot['mixed_native_paths'] = sorted(g.SPECIAL_PATHS | {name})
    record = inventory['devices'].pop('/dev/nvidia0')
    record['rdev_minor'] = index
    inventory['devices'][name] = record
    return snapshot, inventory


class PhysicalIndexTests(unittest.TestCase):
    def test_nonzero_physical_indices_keep_exact_device_checks(self):
        for index in (0, 1, 3, 253):
            snapshot, inventory = indexed(index)
            before = copy.deepcopy((snapshot, inventory))
            view, proof = fixtures.check(snapshot, inventory, 'after-model-load')
            self.assertEqual((snapshot, inventory), before)
            self.assertEqual(view['mixed_native_paths'], [])
            self.assertIn('/dev/nvidia'+str(index), proof['device_inventory']['devices'])
            self.assertFalse(proof['gpu_assignment_proved'])

    def test_reserved_malformed_multiple_and_unknown_nodes_rejected(self):
        snapshot, inventory = indexed(1)
        for name in ('/dev/nvidia01', '/dev/nvidia-1', '/dev/nvidia254', '/dev/nvidia255',
                     '/dev/nvidia256', '/dev/nvidia1000', '/dev/nvidia1/child',
                     '/dev/nvidia1.so', '/tmp/nvidia1', '/dev/nvidia-modeset'):
            bad = copy.deepcopy(snapshot)
            bad['mixed_native_paths'] = sorted(g.SPECIAL_PATHS | {name})
            with self.subTest(name=name), self.assertRaises(ValueError):
                fixtures.check(bad, inventory, 'after-model-load')
        for paths in (sorted(g.SPECIAL_PATHS), sorted(g.SPECIAL_PATHS | {'/dev/nvidia0', '/dev/nvidia1'})):
            with self.assertRaises(ValueError): g.device_paths(paths)
        bad = copy.deepcopy(snapshot)
        bad['process_maps'] += '5000-6000 r-xp 0 00:06 900 /tmp/untrusted.so\n'
        with self.assertRaises(ValueError): fixtures.check(bad, inventory, 'after-model-load')

    def test_inventory_cannot_substitute_logical_zero_for_physical_one(self):
        snapshot, inventory = indexed(1)
        _, wrong = indexed(0)
        with self.assertRaises(ValueError): fixtures.check(snapshot, wrong, 'after-requests')
        for key, value in (('rdev_minor', 0), ('rdev_major', 1), ('inode', 999), ('uid', 1000)):
            bad = copy.deepcopy(inventory); bad['devices']['/dev/nvidia1'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                fixtures.check(snapshot, bad, 'after-requests')
        bad = copy.deepcopy(inventory)
        bad['proc_devices'] = bad['proc_devices'].replace('195 nvidia-frontend', '196 nvidia-frontend')
        bad['proc_devices_sha256'] = hashlib.sha256(bad['proc_devices'].encode()).hexdigest()
        with self.assertRaises(ValueError): fixtures.check(snapshot, bad, 'after-requests')

    def test_nonzero_node_still_rejects_executable_private_or_deleted_mapping(self):
        snapshot, inventory = indexed(1)
        for old, new in (('1000-2000 rw-s', '1000-2000 rwxs'), ('1000-2000 rw-s', '1000-2000 rw-p'),
                         ('100 /dev/nvidia1', '100 /dev/nvidia1 (deleted)')):
            bad = copy.deepcopy(snapshot); bad['process_maps'] = bad['process_maps'].replace(old, new)
            with self.subTest(new=new), self.assertRaises(ValueError):
                fixtures.check(bad, inventory, 'after-requests')

    def test_capture_uses_observed_node_and_metadata_only(self):
        snapshot, expected = indexed(1)
        calls = []
        # Platform-independent fake dev_t. This tests capture wiring, not Linux
        # device-number encoding or real metadata syscalls on this Mac.
        def metadata(path):
            name = str(path); calls.append(name); row = expected['devices'][name]
            return types.SimpleNamespace(st_uid=row['uid'], st_mode=row['mode'], st_ino=row['inode'],
                st_dev=(row['dev_major'] << 20) | row['dev_minor'],
                st_rdev=(row['rdev_major'] << 20) | row['rdev_minor'])
        def opened(path, mode):
            self.assertEqual((str(path), mode), ('/proc/devices', 'rb'))
            return io.BytesIO(expected['proc_devices'].encode())
        with patch.object(Path, 'resolve', lambda path, strict=False: path), \
             patch.object(Path, 'lstat', metadata), patch.object(Path, 'open', opened), \
             patch.object(g.os, 'major', lambda value: value >> 20), \
             patch.object(g.os, 'minor', lambda value: value & ((1 << 20)-1)):
            observed = g.capture(snapshot)
        self.assertEqual(observed, expected)
        self.assertNotIn('/dev/nvidia0', calls)
        self.assertEqual(len(calls), 8)
        self.assertEqual(set(calls), set(expected['devices']))


if __name__ == '__main__': unittest.main()
