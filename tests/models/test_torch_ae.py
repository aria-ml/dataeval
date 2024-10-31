import pytest
import torch
from torch.utils.data import DataLoader

from dataeval._internal.models.pytorch.autoencoder import (
    AETrainer,
    AriaAutoencoder,
    Decoder,
    Encoder,
    get_images_from_batch,
)
from tests.utils.data import DataEvalDataset


@pytest.fixture
def dataset(images=None, labels=None, bboxes=None):
    return DataEvalDataset(images, labels, bboxes)


@pytest.mark.parametrize(
    "channels",
    [
        1,
        2,
        3,
        5,
    ],
)
class TestChannels:
    """Tests the functionality of the channels parameter"""

    def test_encoder(self, channels):
        """For any given channel size, output becomes 64 channels"""
        encoder = Encoder(channels=channels)
        # NCHW
        images = torch.ones(size=[1, channels, 32, 32])
        result = encoder(images)

        # channels = autoencoder output (64)
        assert result.shape[1] == 64

    def test_decoder(self, channels):
        """The output image has correct channel size"""
        decoder = Decoder(channels=channels)
        # NCHW
        images = torch.ones(size=[1, 64, 1, 1])  #
        result = decoder(images)

        # Decode should output any channel size
        assert result.shape[1] == channels

    def test_autoencoder(self, channels):
        """The channel size matches after encoding and decoding"""
        ae = AriaAutoencoder(channels=channels)
        # NCHW
        images = torch.ones(size=[1, channels, 32, 32])
        result = ae(images)

        assert result.shape[1] == channels


class TestTrainer:
    """Tests the AETrainer class"""

    def test_train_aria_ae(self):
        """Aria provided autoencoder can be trained"""
        images = torch.ones(size=(5, 3, 32, 32))
        dataset = DataEvalDataset(images)
        ae = AriaAutoencoder(channels=3)
        trainer = AETrainer(model=ae)
        trainer.train(dataset, epochs=5)

    def test_train_bad_model(self):
        """
        If model output image shape != input image shape, it's likely not an AE
        """
        images = torch.ones(size=(5, 3, 32, 32))
        dataset = DataEvalDataset(images)
        model = Encoder()
        trainer = AETrainer(model)

        # RuntimeError for unmatching imgs/preds during Loss calculation
        with pytest.raises(RuntimeError):
            trainer.train(dataset, epochs=5)

    def test_eval_aria_ae(self):
        """Aria provided autoencoder has evaluate on new data"""
        images = torch.ones(size=[5, 3, 32, 32])
        dataset = DataEvalDataset(images)
        ae = AriaAutoencoder(channels=3)
        trainer = AETrainer(model=ae)
        loss = trainer.eval(dataset)
        assert loss > 0

    def test_encode_aria_ae(self):
        images = torch.ones(size=(5, 3, 32, 32))
        dataset = DataEvalDataset(images)
        ae = AriaAutoencoder(channels=3)
        trainer = AETrainer(model=ae)
        embeddings = trainer.encode(dataset)
        assert embeddings.shape == (5, 64, 7, 7)

    def test_encode_batch(self):
        images = torch.ones(size=(20, 3, 32, 32))
        dataset = DataEvalDataset(images)
        ae = AriaAutoencoder(channels=3)
        trainer = AETrainer(model=ae)
        embeddings = trainer.encode(dataset)
        # Checks batch stacking functionality
        assert embeddings.shape == (20, 64, 7, 7)

    def test_encode_missing_encode(self):
        images = torch.ones(size=(5, 3, 32, 32))
        dataset = DataEvalDataset(images)
        ae = Encoder(channels=3)
        trainer = AETrainer(model=ae)
        embeddings = trainer.encode(dataset)

        assert embeddings.shape == (5, 64, 7, 7)

    # Parameterizing the 3 following tests causes errors, so split into 3 tests
    def test_images_from_batch_imgs(self):
        ds = DataEvalDataset(torch.ones(size=(8, 3, 32, 32)))
        imgs = []
        for batch in DataLoader(dataset=ds, batch_size=8):
            imgs = get_images_from_batch(batch)
            assert isinstance(imgs, torch.Tensor)
            assert imgs.shape == (8, 3, 32, 32)

    def test_images_from_batch_lbls(self):
        ds = DataEvalDataset(torch.ones(size=(8, 3, 32, 32)), torch.ones(size=(8, 1)))
        imgs = []
        for batch in DataLoader(dataset=ds, batch_size=8):
            # print("Batch:", batch)
            imgs = get_images_from_batch(batch)
            assert isinstance(imgs, torch.Tensor)
            assert imgs.shape == (8, 3, 32, 32)

    def test_images_from_batch_bxs(self):
        ds = DataEvalDataset(
            torch.ones(size=(8, 3, 32, 32)),
            torch.ones(size=(8, 1)),
            torch.ones(size=(8, 2)),
        )
        imgs = []
        for batch in DataLoader(dataset=ds, batch_size=8):
            imgs = get_images_from_batch(batch)
            assert isinstance(imgs, torch.Tensor)
            assert imgs.shape == (8, 3, 32, 32)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        torch.device("cpu"),
        pytest.param(
            0,
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available"),
        ),
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available"),
        ),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available"),
        ),
    ],
)
class TestGPU:
    def test_trainer_device(self, device):
        model = AriaAutoencoder()
        trainer = AETrainer(model, device=device)
        # Check trainer device set properly
        assert trainer.device == device

        # Check if all params moved to device
        m = trainer.model
        assert isinstance(m, torch.nn.Module)

        # Need to check device.type as tensor's device automatically selects an index
        # i.e. param.to("cuda"), param.device equals device(type="cuda", index=0)
        for param in m.parameters():
            assert param.device.type == torch.device(device).type
