"""A real worker refusal must reach its parent with every option and context."""
import concurrent.futures as cf
import multiprocessing as mp
import pickle
import unittest
from woof.hex.physics_suites import PhysicsSelectionRefusal, OptionRefusal

def refuse():
    raise PhysicsSelectionRefusal([
        OptionRefusal('config_convection_scheme', 'requested', 'native role mismatch', 'inspect authority'),
        OptionRefusal('config_pbl_scheme', 'requested', 'source mismatch', 'inspect source'),
    ], context='final-native-stage')

class RefusalTransport(unittest.TestCase):
    def test_pickle_roundtrip(self):
        try:refuse()
        except PhysicsSelectionRefusal as original:
            restored=pickle.loads(pickle.dumps(original))
            self.assertEqual(restored.refusals,original.refusals)
            self.assertEqual(restored.context,original.context)
            self.assertEqual(str(restored),str(original))
    def test_actual_worker_pool(self):
        with cf.ProcessPoolExecutor(max_workers=1,mp_context=mp.get_context('spawn'))as pool:
            future=pool.submit(refuse)
            with self.assertRaises(PhysicsSelectionRefusal)as caught:future.result(timeout=10)
        self.assertEqual(caught.exception.context,'final-native-stage')
        self.assertEqual(len(caught.exception.refusals),2)
        self.assertIn('native role mismatch',str(caught.exception))
if __name__=='__main__':unittest.main()
