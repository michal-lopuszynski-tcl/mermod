import collections

import torch


class SimpleModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(10, 20, bias=True)
        self.l2 = torch.nn.Linear(20, 10, bias=True)
        self.l3 = torch.nn.Linear(10, 20, bias=True)
        self.l4 = torch.nn.Linear(20, 10, bias=True)
        self.l5 = torch.nn.Linear(10, 20, bias=True)
        self.l6 = torch.nn.Linear(20, 10, bias=True)
        self.l7 = torch.nn.Linear(10, 20, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        return x


def gen_state_dict(seed: int) -> collections.OrderedDict[str, torch.Tensor]:
    device = torch.device("cpu")
    gen = torch.Generator(device=device)
    d = collections.OrderedDict()
    d["l1.weight"] = torch.rand((20, 10), generator=gen, device=device)
    d["l1.bias"] = torch.rand((20,), generator=gen, device=device)
    d["l2.weight"] = torch.rand((10, 20), generator=gen, device=device)
    d["l2.bias"] = torch.rand((10,), generator=gen, device=device)
    d["l3.weight"] = torch.rand((20, 10), generator=gen, device=device)
    d["l3.bias"] = torch.rand((20,), generator=gen, device=device)
    d["l4.weight"] = torch.rand((10, 20), generator=gen, device=device)
    d["l4.bias"] = torch.rand((10,), generator=gen, device=device)
    d["l5.weight"] = torch.rand((20, 10), generator=gen, device=device)
    d["l5.bias"] = torch.rand((20,), generator=gen, device=device)
    d["l6.weight"] = torch.rand((10, 20), generator=gen, device=device)
    d["l6.bias"] = torch.rand((10,), generator=gen, device=device)
    d["l7.weight"] = torch.rand((20, 10), generator=gen, device=device)
    d["l7.bias"] = torch.rand((20,), generator=gen, device=device)
    return d


CONFIG_ABS_DIFF_TIES0 = {
    "sd_base_path": 621345,
    "sd_merged_paths": {
        "tmp_01": 123456,
        "tmp_02": 245145,
        "tmp_03": 423155,
    },
    "sd_output_path": "tests/data_pth/abs_diff_ties0.pth",
    "method": "abs_diff",
    "seed_dict": None,
    "lambda_param": 0.4,
    "sparsity": 0.93,
    "use_ties": False,
    "weight_batch_size": 96,
    "weight_batches_custom": None,
    "merge_device": "cpu",
}


CONFIG_ABS_DIFF_TIES1 = {
    "sd_base_path": 711244,
    "sd_merged_paths": {
        "tmp_01": 190122,
        "tmp_02": 871235,
        "tmp_03": 912341,
    },
    "sd_output_path": "tests/data_pth/abs_diff_ties1.pth",
    "method": "abs_diff",
    "seed_dict": None,
    "lambda_param": 0.4,
    "sparsity": 0.93,
    "use_ties": True,
    "weight_batch_size": 96,
    "weight_batches_custom": None,
    "merge_device": "cpu",
}
