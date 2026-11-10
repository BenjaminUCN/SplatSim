import tempfile
import unittest
from pathlib import Path

from splatsim.utils.data.registry import find_model, models_table


class TestFindModel(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)

        def model(folder, name="", finished=True, saved="2026-11-01 10:00"):
            (root / folder).mkdir()
            return {"name": name, "suggested_name": f"2cam-{folder}", "path": str(root / folder),
                    "finished": finished, "cameras": ["base_rgb", "wrist_rgb"], "dataset": "ds", "saved": saved}

        self.registry = {"models": {
            "rlt/gello_benjamin_200_2cam_1106_1648": model("gello_benjamin_200_2cam_1106_1648",
                                                           "gello benjamin 200 2cam", saved="2026-11-07 06:06"),
            "rlt/300-planner-trajectories-2cam_1104_1706": model("300-planner-trajectories-2cam_1104_1706",
                                                                "300-planner-trajectories-2cam"),
            "rlt/diffusion_0726_1024": model("diffusion_0726_1024"),
            "rlt/unfinished_0101": model("unfinished_0101", "unfinished", finished=False),
        }}

    def tearDown(self):
        self.tmp.cleanup()

    def key(self, query):
        return find_model(query, self.registry)[0]

    def test_exact_name_folder_or_key(self):
        self.assertEqual(self.key("gello benjamin 200 2cam"), "rlt/gello_benjamin_200_2cam_1106_1648")
        self.assertEqual(self.key("gello_benjamin_200_2cam_1106_1648"), "rlt/gello_benjamin_200_2cam_1106_1648")
        self.assertEqual(self.key("rlt/diffusion_0726_1024"), "rlt/diffusion_0726_1024")

    def test_spaces_underscores_and_case_are_the_same(self):
        self.assertEqual(self.key("GELLO-Benjamin_200 2CAM"), "rlt/gello_benjamin_200_2cam_1106_1648")

    def test_unique_part_of_a_name(self):
        self.assertEqual(self.key("planner"), "rlt/300-planner-trajectories-2cam_1104_1706")

    def test_a_part_shared_by_several_models_is_ambiguous(self):
        with self.assertRaisesRegex(ValueError, "coincide con"):
            self.key("2cam")

    def test_unknown_or_unfinished_models_list_the_choices(self):
        for query in ("nothing", "unfinished"):
            with self.assertRaisesRegex(ValueError, "Modelos disponibles"):
                self.key(query)

    def test_table_lists_finished_models_newest_first(self):
        lines = models_table(self.registry).splitlines()
        self.assertTrue(lines[0].startswith("name"))
        self.assertTrue(lines[1].startswith("gello benjamin 200 2cam"))
        self.assertEqual(len(lines), 4)   # cabecera + 3 terminados


if __name__ == "__main__":
    unittest.main()
