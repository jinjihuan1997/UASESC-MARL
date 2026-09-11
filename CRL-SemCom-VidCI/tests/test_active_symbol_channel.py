import torch
from experiment_scripts.comm.channel import power_normalize, apply_channel, build_rate_mask


def test_active_power_and_snr_independent_of_retained_fraction():
    torch.manual_seed(18)
    for level in (1, 2, 3, 4):
        mask = build_rate_mask(torch.full((4, 1, 32, 32), float(level)))
        symbols = torch.randn_like(mask) * mask
        x = power_normalize(symbols, rate_mask=mask)
        active = mask.bool()
        torch.testing.assert_close(x[active].square().mean(), torch.tensor(1.), rtol=1e-5, atol=1e-5)
        y = apply_channel(symbols, 10., training=False, rate_mask=mask)
        assert torch.count_nonzero(y[~active]) == 0
        empirical = 10 * torch.log10(x[active].square().mean() / (y-x)[active].square().mean())
        assert abs(float(empirical) - 10.) < .2


def test_mask_counts_transmitted_zeros_and_zero_padding_has_no_gradient():
    mask = torch.tensor([[[[1., 1., 0., 0.]]]])
    x = torch.tensor([[[[2., 0., 7., 8.]]]], requires_grad=True)
    y = power_normalize(x, rate_mask=mask)
    torch.testing.assert_close(y, torch.tensor([[[[2.**.5, 0., 0., 0.]]]]))
    y.sum().backward()
    assert torch.count_nonzero(x.grad[mask == 0]) == 0
