"""
Unit tests for CNN model.
"""

import torch

from models.cnn import SimpleCNN


# ---------------------------------------------------------
# Model Creation
# ---------------------------------------------------------

def test_model_created(model):
    """
    Verify model is created successfully.
    """

    assert model is not None

    assert isinstance(model, SimpleCNN)


# ---------------------------------------------------------
# Forward Pass
# ---------------------------------------------------------

def test_forward_pass(model):
    """
    Verify forward pass executes.
    """

    dummy_input = torch.randn(
        8,
        3,
        128,
        128,
    )

    output = model(dummy_input)

    assert output is not None


# ---------------------------------------------------------
# Output Shape
# ---------------------------------------------------------

def test_output_shape(model):
    """
    Verify output tensor shape.
    """

    dummy_input = torch.randn(
        8,
        3,
        128,
        128,
    )

    output = model(dummy_input)

    assert output.shape == (8, 2)


# ---------------------------------------------------------
# Output Type
# ---------------------------------------------------------

def test_output_dtype(model):
    """
    Verify output datatype.
    """

    dummy_input = torch.randn(
        4,
        3,
        128,
        128,
    )

    output = model(dummy_input)

    assert output.dtype == torch.float32


# ---------------------------------------------------------
# Batch Size Preservation
# ---------------------------------------------------------

def test_batch_size_preserved(model):
    """
    Output batch size should match input.
    """

    dummy_input = torch.randn(
        16,
        3,
        128,
        128,
    )

    output = model(dummy_input)

    assert output.shape[0] == 16


# ---------------------------------------------------------
# Number of Classes
# ---------------------------------------------------------

def test_number_of_output_classes(model):
    """
    Verify final classifier output.
    """

    dummy_input = torch.randn(
        2,
        3,
        128,
        128,
    )

    output = model(dummy_input)

    assert output.shape[1] == 2


# ---------------------------------------------------------
# Model Parameters
# ---------------------------------------------------------

def test_model_has_parameters(model):
    """
    Verify model contains trainable parameters.
    """

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    assert total_parameters > 0


# ---------------------------------------------------------
# Trainable Parameters
# ---------------------------------------------------------

def test_trainable_parameters(model):
    """
    Verify parameters require gradients.
    """

    trainable_parameters = sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad

    )

    assert trainable_parameters > 0


# ---------------------------------------------------------
# Gradient Flow
# ---------------------------------------------------------

def test_backward_pass(model):
    """
    Verify gradients are computed.
    """

    dummy_input = torch.randn(
        8,
        3,
        128,
        128,
    )

    labels = torch.randint(
        0,
        2,
        (8,),
    )

    criterion = torch.nn.CrossEntropyLoss()

    output = model(dummy_input)

    loss = criterion(
        output,
        labels,
    )

    loss.backward()

    gradients_exist = False

    for parameter in model.parameters():

        if parameter.grad is not None:

            gradients_exist = True

            break

    assert gradients_exist


# ---------------------------------------------------------
# Model Training Mode
# ---------------------------------------------------------

def test_model_train_mode(model):
    """
    Verify training mode.
    """

    model.train()

    assert model.training is True


# ---------------------------------------------------------
# Model Evaluation Mode
# ---------------------------------------------------------

def test_model_eval_mode(model):
    """
    Verify evaluation mode.
    """

    model.eval()

    assert model.training is False


# ---------------------------------------------------------
# Inference Without Gradients
# ---------------------------------------------------------

def test_inference_no_grad(model):
    """
    Verify inference works without gradients.
    """

    dummy_input = torch.randn(
        4,
        3,
        128,
        128,
    )

    model.eval()

    with torch.no_grad():

        output = model(dummy_input)

    assert output.shape == (4, 2)


# ---------------------------------------------------------
# Output Contains Finite Values
# ---------------------------------------------------------

def test_output_is_finite(model):
    """
    Verify no NaN or Inf values.
    """

    dummy_input = torch.randn(
        4,
        3,
        128,
        128,
    )

    output = model(dummy_input)

    assert torch.isfinite(output).all()


# ---------------------------------------------------------
# Model on CPU
# ---------------------------------------------------------

def test_model_device_cpu(model):
    """
    Verify model defaults to CPU.
    """

    device = next(
        model.parameters()
    ).device

    assert device.type == "cpu"