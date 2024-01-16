import pytest
import torch

from daml._internal.models.pytorch.autoencoder import (
    AERunner,
    AETrainer,
    AriaAutoencoder,
    Decoder,
    Encoder,
)

# from torch.utils.data import DataLoader


# from daml._internal.datasets import DamlDataset


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

    def test_train_default_model(self):
        images = torch.ones(size=[1, 3, 32, 32])
        trainer = AETrainer()
        trainer.train(images, epochs=1)

    def test_train_good_model(self):
        images = torch.ones(size=[1, 3, 32, 32])
        model = AriaAutoencoder()
        trainer = AETrainer(model)
        trainer.train(images, epochs=1)

    def test_train_bad_model(self):
        """
        If model output image shape != input image shape, it's likely not an AE"""
        images = torch.ones(size=[1, 3, 32, 32])
        model = Encoder()
        trainer = AETrainer(model)

        # RuntimeError for unmatching imgs/preds during Loss calculation
        with pytest.raises(RuntimeError):
            trainer.train(images, epochs=1)

    def test_batch(self):
        """Image control logic"""
        images = torch.ones(size=[10, 3, 32, 32])
        trainer = AETrainer()
        trainer.train(images, epochs=1)

    def test_model(self):
        pass


class TestRunner:
    """Tests the AERunner class"""

    def test_call(self):
        """
        Calls runner as if it was the forward pass of the given model
        For an AE, it should reconstruct the image
        """
        images = torch.ones(size=[1, 3, 32, 32])
        model = AriaAutoencoder(3)
        runner = AERunner(model=model)
        result = runner(images)

        assert result.shape == images.shape

    def test_encode(self):
        """
        Calls encode on a model with the encode function.
        Returns an encoded image shape (model specific)
        """
        images = torch.ones(size=[1, 3, 32, 32])
        model = AriaAutoencoder(3)
        runner = AERunner(model=model)
        result = runner.encode(images)

        # Height/Width based on current AE & given image size
        assert result.shape == (1, 64, 7, 7)

    def test_no_encode(self):
        """
        Calls encode on a model that does not have an encode function
        Returns normal model forward behavior
        """
        images = torch.ones(size=[1, 64, 7, 7])
        model = Decoder(3)  # Decoder does not have an encode function
        runner = AERunner(model=model)
        result = runner.encode(images)

        # # Height/Width based on current AE & given image size
        assert result.shape == (1, 3, 32, 32)
