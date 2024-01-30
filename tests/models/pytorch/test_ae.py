import pytest
import torch

from daml._internal.models.pytorch.autoencoder import (
    AERunner,
    AETrainer,
    AriaAutoencoder,
    Decoder,
    Encoder,
)


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

    def test_no_model_no_channels(self):
        with pytest.raises(TypeError):
            AETrainer()

    def test_train_default_model(self):
        images = torch.ones(size=[1, 3, 32, 32])
        trainer = AETrainer(channels=3)
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
        trainer = AETrainer(channels=3)
        trainer.train(images, epochs=1, batch_size=10)

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


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        torch.device("cpu"),
        pytest.param(
            0,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="cuda not available"
            ),
        ),
        pytest.param(
            torch.device(0),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="cuda not available"
            ),
        ),
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="cuda not available"
            ),
        ),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="cuda not available"
            ),
        ),
    ],
)
class TestGPU:
    def test_runner_device(self, device):
        model = AriaAutoencoder()
        runner = AERunner(model=model, device=device)

        # Check runner device set properly
        assert runner._device == device

        # Check if all params moved to device
        m = runner._model
        assert isinstance(m, torch.nn.Module)

        # Need to check device.type as tensor's device automatically selects an index
        # i.e. param.to("cuda"), param.device equals device(type="cuda", index=0)
        for param in m.parameters():
            print(param.device.type)
            assert param.device.type == torch.device(device).type

    def test_trainer_device(self, device):
        trainer = AETrainer(channels=1, device=device)
        # Check trainer device set properly
        assert trainer._device == device

        # Check if all params moved to device
        m = trainer._model
        assert isinstance(m, torch.nn.Module)

        # Need to check device.type as tensor's device automatically selects an index
        # i.e. param.to("cuda"), param.device equals device(type="cuda", index=0)
        for param in m.parameters():
            assert param.device.type == torch.device(device).type
