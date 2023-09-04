import pytest

from sviro.dataset import SVIRODetection


class TestSVIRODetectionDataset:
    @pytest.mark.parametrize('car', [['x5'], ['aclass'], ['hilux']])
    def test_len_train_per_car(self, car):
        dataset = SVIRODetection(dataroot='data', car=car, split='train')
        assert len(dataset) == 2000

    def test_len_train_total(self):
        dataset = SVIRODetection(dataroot='data', car='all', split='train')
        assert len(dataset) == 22000

    @pytest.mark.parametrize('car', [['x5'], ['aclass'], ['hilux']])
    def test_len_test_per_car(self, car):
        dataset = SVIRODetection(dataroot='data', car=car, split='test')
        assert len(dataset) == 500

    def test_len_test_total(self):
        dataset = SVIRODetection(dataroot='data', car='all', split='test')
        assert len(dataset) == 5000
