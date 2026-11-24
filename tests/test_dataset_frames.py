"""PNG de un render reemplazados por los cuadros del dataset LeRobot (splatsim/utils/data/dataset_frames.py).

    python -m unittest discover -s tests -p 'test_dataset_frames.py'

Arma un dataset mínimo con dos episodios en un solo video, como lerobot_conversion.py: H.264 en vez de AV1, para
no depender del codificador.
"""
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from splatsim.utils.data import dataset_frames as df

CAMS = ["base_rgb", "wrist_rgb"]
LENGTHS = {"000": 8, "001": 6}
FPS = 10


def frame(ep: int, k: int, cam: int) -> np.ndarray:
    """Imagen suave que cambia con el cuadro: un desfase de un cuadro se nota en el PSNR."""
    y, x = np.mgrid[0:48, 0:64]
    a = np.stack([np.sin((x + 6 * k + 20 * ep) / 9.0 + cam), np.cos((y + 4 * k) / 7.0 + ep),
                  np.sin((x + y + 3 * k) / 11.0)], axis=-1)
    return ((a + 1) * 120 + 5).astype(np.uint8)


def build(root: Path) -> tuple[Path, Path]:
    import av
    render, dataset = root / "demo_10hz", root / "ds" / "demo_ds"
    rows = []
    start = 0
    videos = {}
    for cam in CAMS:
        out = dataset / "videos" / f"observation.images.{cam}" / "chunk-000" / "file-000.mp4"
        out.parent.mkdir(parents=True)
        videos[cam] = (av.open(str(out), "w"), out)
        stream = videos[cam][0].add_stream("libx264", rate=FPS, options={"crf": "10"})
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv444p"
    for e, (name, n) in enumerate(LENGTHS.items()):
        ep = render / name
        (ep / "image_1").mkdir(parents=True)
        (ep / "meta.json").write_text("{}")
        for k in range(n):
            (ep / f"{k + 1:05d}.pkl").write_bytes(pickle.dumps({"state": np.zeros(7)}))
            for c, cam in enumerate(CAMS):
                img = frame(e, k, c)
                Image.fromarray(img).save(ep / "image_1" / f"{cam}_{k + 1:05d}.png")
                container = videos[cam][0]
                for packet in container.streams.video[0].encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                    container.mux(packet)
        row = {"episode_index": e, "length": n}
        for cam in CAMS:
            key = f"videos/observation.images.{cam}"
            row.update({f"{key}/chunk_index": 0, f"{key}/file_index": 0,
                        f"{key}/from_timestamp": start / FPS, f"{key}/to_timestamp": (start + n) / FPS})
        rows.append(row)
        start += n
    for container, _ in videos.values():
        for packet in container.streams.video[0].encode():
            container.mux(packet)
        container.close()
    (dataset / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(dataset / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    features = {f"observation.images.{c}": {"dtype": "video", "shape": [3, 48, 64]} for c in CAMS}
    features["observation.state"] = {"dtype": "float32", "shape": [6]}
    (dataset / "meta" / "info.json").write_text(json.dumps({
        "fps": FPS, "features": features,
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}))
    (dataset / "meta" / "splatsim_sources.yaml").write_text(yaml.safe_dump(
        {"sources": [{"folder": render.name, "episodes": list(LENGTHS), "dataset_episodes": [0, 1]}]}))
    (dataset / "DONE").touch()
    (render / "DONE").touch()
    return render, dataset


class DatasetFramesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.render, self.dataset = build(Path(self.tmp.name))
        df._episodes.cache_clear()
        df._decode.cache_clear()

    def tearDown(self):
        self.tmp.cleanup()

    def test_find_and_check(self):
        dataset, mapping = df.find_dataset(self.render, [self.dataset.parent])
        self.assertEqual((dataset, mapping), (self.dataset, {"000": 0, "001": 1}))
        self.assertEqual(df.check(self.render, dataset, mapping), [])

    def test_frames_match_png(self):
        frames = df.DatasetFrames(self.dataset, 1, "wrist_rgb")
        self.assertEqual(len(frames), LENGTHS["001"])
        for k in range(len(frames)):
            self.assertGreater(df._psnr(frames.array(k), frame(1, k, 1)), df.MIN_PSNR_DB)
        self.assertEqual(frames[-1].size, (64, 48))

    def test_wrong_mapping_fails(self):
        # mismo largo, otro episodio: el PSNR delata la mezcla
        (self.render / "001").rename(self.render / "tmp")
        for p in sorted((self.render / "000").glob("*.pkl"))[6:]:
            p.unlink()
        for p in (self.render / "000" / "image_1").glob("*_0000[78].png"):
            p.unlink()
        problems = df.check(self.render, self.dataset, {"000": 1})
        self.assertTrue(problems)

    def test_shifted_fails(self):
        orig = df.DatasetFrames.__init__

        def shifted(self, *a):
            orig(self, *a)
            self.first += 1
        df.DatasetFrames.__init__ = shifted
        try:
            problems = df.check(self.render, self.dataset, {"000": 0})
        finally:
            df.DatasetFrames.__init__ = orig
        self.assertTrue(any("vecinos" in p or "PSNR" in p for p in problems), problems)

    def test_prune_then_cameras(self):
        mapping = {"000": 0}   # "001" fuera del dataset (descartado): conserva sus PNG
        size, kept, extra = df.prune(self.render, self.dataset, mapping)
        self.assertGreater(size, 0)
        self.assertTrue(df.png_files(self.render / "000"))   # sin delete no borra
        df.prune(self.render, self.dataset, mapping, delete=True)
        self.assertEqual(df.png_files(self.render / "000"), {})
        self.assertEqual(sorted(df.png_files(self.render / "001")), CAMS)
        self.assertEqual(len(list((self.render / "000").glob("*.pkl"))), LENGTHS["000"])
        cams = df.dataset_cameras(self.render, "000")
        self.assertEqual(sorted(cams), CAMS)
        self.assertEqual(len(cams["base_rgb"]), LENGTHS["000"])
        self.assertEqual(df.dataset_cameras(self.render, "001"), {})
        # ya sin PNG, el chequeo sigue pasando (se puede volver a correr)
        self.assertEqual(df.check(self.render, self.dataset, mapping), [])


if __name__ == "__main__":
    unittest.main()
