from cray_infra.training.get_latest_model import get_latest_model
import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path


class TestGetLatestModel(unittest.TestCase):
    @patch("cray_infra.training.get_latest_model.get_config")
    def test_training_job_directory_path(self, mock_get_config):
        mock_get_config.return_value = {
            "training_job_directory" : ""
        }   

        with self.assertRaises(FileNotFoundError):
            get_latest_model()


    @patch("cray_infra.training.get_latest_model.get_config")
    def test_empty_training_job_directory(self, mock_get_config):
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_get_config.return_value = {
                "training_job_directory" : temp_dir
            } 

        with self.assertRaises(FileNotFoundError):
            get_latest_model()

    @patch("cray_infra.training.get_latest_model.get_config")
    def test_return_latest_model_single_model(self, mock_get_config):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            mock_get_config.return_value = {
                "training_job_directory" : temp_dir
            }  

            model0 = temp_path / "model0"
            model0.mkdir()
            (model0 / "status.json").write_text('{"start_time": 2000, "history":[]}')
    
            latest_model = get_latest_model()
            self.assertEqual(latest_model, "model0")

    @patch("cray_infra.training.get_latest_model.get_config")
    def test_return_latest_model_multiple_models(self, mock_get_config):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            mock_get_config.return_value = {
                "training_job_directory" : temp_dir
            } 

            models = [
                ("model0", 2000),
                ("model1", 6000),
                ("model2", 300),
                ("model3", 0)
            ]

            for model_name, start_time in models:
                model_dir = temp_path / model_name
                model_dir.mkdir()
                (model_dir / "status.json").write_text(f'{{"start_time": {start_time}, "history": []}}')

            latest_model = get_latest_model()
            assert latest_model == "model1"



    


