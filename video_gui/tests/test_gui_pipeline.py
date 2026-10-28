"""Contract tests: real saved scores, QC, timestamps, export, and cancellation."""
from pathlib import Path
import sys
import tempfile
import unittest
import json
from fractions import Fraction
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import FixedPredictor
from backend import (map_world, local_score, analyze, Cancelled, oriented_rgb)

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = FixedPredictor()

    def test_original_reference_scores(self):
        self.assertLess(self.predictor.reference_max_error, 1e-6)

    def test_mapping_uses_world_and_rejects_low_visibility(self):
        landmarks = [SimpleNamespace(x=float(i),y=2.,z=3.,visibility=1.,presence=1.) for i in range(33)]
        landmarks[23].visibility = .1
        mapped = map_world(landmarks,.5)
        np.testing.assert_equal(mapped[1], [24.,-2.,3.])
        self.assertTrue(np.isnan(mapped[4]).all())

    def test_scores_do_not_carry_across_invalid_gaps(self):
        rows=[dict(start_s=0.,end_s=1.5,score=.2),dict(start_s=1.,end_s=2.5,score=.8),
              dict(start_s=4.,end_s=5.5,score=.9)]
        self.assertAlmostEqual(local_score(1.1,rows),.5)
        self.assertIsNone(local_score(3.,rows))
        self.assertIsNone(local_score(5.5,rows))

    def test_rotation(self):
        rgb=np.zeros((4,6,3),np.uint8);rgb[0,0]=255
        frame=SimpleNamespace(rotation=90,to_ndarray=lambda **kw:rgb)
        stream=SimpleNamespace(metadata={})
        np.testing.assert_equal(oriented_rgb(frame,stream),np.rot90(rgb))

    def make_video(self, root):
        import av
        path=root/"input.mp4"
        # Nonuniform PTS exercises source-time based rendering.
        times=np.array([0.,.04,.1,.18,.22,.3])
        with av.open(str(path),"w") as container:
            s=container.add_stream("libx264",rate=25);s.width=160;s.height=120;s.pix_fmt="yuv420p"
            for i,t in enumerate(times):
                rgb=np.full((120,160,3),i*30,np.uint8)
                f=av.VideoFrame.from_ndarray(rgb,format="rgb24")
                f.pts=round(t*1000);f.time_base=Fraction(1,1000)
                for packet in s.encode(f):container.mux(packet)
            for packet in s.encode():container.mux(packet)
        with av.open(str(path)) as c:
            decoded=list(c.decode(video=0))
            times=np.array([float(f.pts*f.time_base) for f in decoded]);times-=times[0]
        return path,times

    def fixture_extractor(self,times,people):
        def extractor(path,cfg,progress,preview,cancel):
            n=len(times)
            return (np.full((n,17,3),np.nan),times,np.full((n,2,33,5),np.nan,np.float32),
                    np.full(n,people),np.zeros(n,bool),
                    dict(frames=n,nominal_fps=25.,median_effective_fps=25.,
                         duration_s=float(times[-1]+np.median(np.diff(times))),
                         detected_single_person_fraction=0.,required_joints_valid_fraction=0.,
                         multiple_person_frames=n if people==2 else 0,missing_pose_frames=n if people==0 else 0,
                         timing="test_pts",timestamp_nudges=0))
        return extractor

    def test_abstention_video_and_artifact_export(self):
        import av
        for people,reason in [(0,"invalid_leg_scale"),(2,"multiple_people_detected")]:
            with self.subTest(people=people),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);video,times=self.make_video(root)
                folder,report=analyze(video,root/"results",predictor=self.predictor,
                                      extractor=self.fixture_extractor(times,people))
                self.assertEqual(report["prediction"]["reason"],reason)
                self.assertIsNone(report["prediction"]["score"])
                self.assertEqual(json.loads((folder/"status.json").read_text())["status"],"complete")
                for name in ["annotated.mp4","prediction.json","pose.npz","features.npz","scores.png",
                             "window_scores.csv","rejected_windows.csv","frame_quality.csv"]:
                    self.assertTrue((folder/name).is_file(),name)
                with av.open(str(folder/"annotated.mp4")) as c:
                    frames=list(c.decode(video=0))
                    self.assertEqual(len(frames),report["video_export"]["frames"])
                    self.assertEqual(float(c.streams.video[0].average_rate),30.)
                    self.assertFalse(c.streams.audio)
                    self.assertGreater(frames[0].height,120) # banner is burned into video
                    pts=[float(f.pts*f.time_base) for f in frames]
                    np.testing.assert_allclose(np.diff(pts),1/30,atol=1e-6)

    def test_cancellation_does_not_publish_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);video,times=self.make_video(root)
            with self.assertRaises(Cancelled):
                analyze(video,root/"results",cancel=lambda:True,predictor=self.predictor)
            folder=next((root/"results").iterdir())
            self.assertEqual(json.loads((folder/"status.json").read_text())["status"],"cancelled")
            self.assertFalse((folder/"annotated.mp4").exists())

    def test_short_sequence_abstains(self):
        # A finite, plausible arrangement: too few frames for a 45-sample window.
        a=np.zeros((20,17,3))
        for hip,knee,ankle,shoulder,elbow,wrist,x in [(1,2,3,14,15,16,-.2),(4,5,6,11,12,13,.2)]:
            a[:,hip]=[x,1,0];a[:,knee]=[x,.5,.05];a[:,ankle]=[x,0,0]
            a[:,shoulder]=[x,1.7,0];a[:,elbow]=[x,1.3,.1];a[:,wrist]=[x,1.,.2]
        p,rows,rejected,windows,signals=self.predictor.predict(a,np.arange(20)/30)
        self.assertEqual(p["status"],"unable_to_assess")
        self.assertEqual(rejected[0]["reason"],"too_short")
        self.assertFalse(rows)

if __name__ == "__main__":
    unittest.main()
