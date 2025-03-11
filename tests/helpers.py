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
    gen.manual_seed(seed)
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
    "weight_batch_size": 2,
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
    "weight_batch_size": 2,
    "weight_batches_custom": None,
    "merge_device": "cpu",
}

CONFIG_JOINT_ABS_DIFF_TIES0 = {
    "sd_base_path": 621345,
    "sd_merged_paths": {
        "tmp_01": 123456,
        "tmp_02": 245145,
        "tmp_03": 423155,
    },
    "sd_output_path": "tests/data_pth/joint_abs_diff_ties0.pth",
    "method": "joint_abs_diff",
    "seed_dict": None,
    "lambda_param": 0.4,
    "sparsity": 0.93,
    "use_ties": False,
    "weight_batch_size": 2,
    "weight_batches_custom": None,
    "merge_device": "cpu",
}


CONFIG_JOINT_ABS_DIFF_TIES1 = {
    "sd_base_path": 122353,
    "sd_merged_paths": {
        "tmp_01": 771351,
        "tmp_02": 587123,
        "tmp_03": 245912,
    },
    "sd_output_path": "tests/data_pth/joint_abs_diff_ties1.pth",
    "method": "joint_abs_diff",
    "seed_dict": None,
    "lambda_param": 0.4,
    "sparsity": 0.93,
    "use_ties": True,
    "weight_batch_size": 2,
    "weight_batches_custom": None,
    "merge_device": "cpu",
}

CONFIG_DARE_TIES1_CPU = {
    "sd_base_path": 711244,
    "sd_merged_paths": {
        "tmp_01": 190122,
        "tmp_02": 871235,
        "tmp_03": 912341,
    },
    "sd_output_path": "tests/data_pth/dare_ties1_cpu.pth",
    "method": "dare",
    "lambda_param": 0.4,
    "sparsity": 0.93,
    "use_ties": True,
    "weight_batch_size": 2,
    "weight_batches_custom": None,
    "merge_device": "cpu",
    "seed_dict": {
        "l7.bias@tmp_01:_cpu": 17784003615498595655,
        "l7.bias@tmp_02:_cpu": 7404201825816571778,
        "l7.bias@tmp_03:_cpu": 9795502066293766854,
        "l7.weight@tmp_01:_cpu": 928492489769832310,
        "l7.weight@tmp_02:_cpu": 7755617984131704863,
        "l7.weight@tmp_03:_cpu": 14860239426471831184,
        "l6.bias@tmp_01:_cpu": 17285555995511593865,
        "l6.bias@tmp_02:_cpu": 8870264517112729447,
        "l6.bias@tmp_03:_cpu": 11282941290745447577,
        "l6.weight@tmp_01:_cpu": 7854268808609121291,
        "l6.weight@tmp_02:_cpu": 15249473777399268520,
        "l6.weight@tmp_03:_cpu": 12086524735816622992,
        "l5.bias@tmp_01:_cpu": 10616863180781513320,
        "l5.bias@tmp_02:_cpu": 14118046335413047683,
        "l5.bias@tmp_03:_cpu": 380972430311930486,
        "l5.weight@tmp_01:_cpu": 9899457818882785998,
        "l5.weight@tmp_02:_cpu": 16833757855925207608,
        "l5.weight@tmp_03:_cpu": 11911973522854882639,
        "l4.bias@tmp_01:_cpu": 13296511645086465325,
        "l4.bias@tmp_02:_cpu": 2152564176237348655,
        "l4.bias@tmp_03:_cpu": 7154680797565651019,
        "l4.weight@tmp_01:_cpu": 13468673235753838751,
        "l4.weight@tmp_02:_cpu": 10037537643080306818,
        "l4.weight@tmp_03:_cpu": 8692562376070472717,
        "l3.bias@tmp_01:_cpu": 935513488004124568,
        "l3.bias@tmp_02:_cpu": 137467038650034386,
        "l3.bias@tmp_03:_cpu": 9060709452800408761,
        "l3.weight@tmp_01:_cpu": 2669193672527560105,
        "l3.weight@tmp_02:_cpu": 5639238667971127654,
        "l3.weight@tmp_03:_cpu": 12774520007147071024,
        "l2.bias@tmp_01:_cpu": 4768204281676269284,
        "l2.bias@tmp_02:_cpu": 11601644324586831791,
        "l2.bias@tmp_03:_cpu": 1830274020188513711,
        "l2.weight@tmp_01:_cpu": 15573237941484929537,
        "l2.weight@tmp_02:_cpu": 5099897434150979340,
        "l2.weight@tmp_03:_cpu": 17560940847204700124,
        "l1.bias@tmp_01:_cpu": 7272725779766596975,
        "l1.bias@tmp_02:_cpu": 40661292051765614,
        "l1.bias@tmp_03:_cpu": 857018380269195192,
        "l1.weight@tmp_01:_cpu": 15612749561639675023,
        "l1.weight@tmp_02:_cpu": 8801382066848050573,
        "l1.weight@tmp_03:_cpu": 13713699276447734432,
    },
}

CONFIG_DARE_TIES1_CUDA = {
    "sd_base_path": 711244,
    "sd_merged_paths": {
        "tmp_01": 235622,
        "tmp_02": 498235,
        "tmp_03": 713341,
    },
    "sd_output_path": "tests/data_pth/dare_ties1_cuda.pth",
    "method": "dare",
    "lambda_param": 0.4,
    "sparsity": 0.93,
    "use_ties": True,
    "weight_batch_size": 2,
    "weight_batches_custom": None,
    "merge_device": "cuda",
    "seed_dict": {
        "l7.bias@tmp_01:_cuda": 11262379629731004144,
        "l7.bias@tmp_02:_cuda": 7470293834762534261,
        "l7.bias@tmp_03:_cuda": 8062465717820707091,
        "l7.weight@tmp_01:_cuda": 17815836381085143450,
        "l7.weight@tmp_02:_cuda": 8043757290364747955,
        "l7.weight@tmp_03:_cuda": 16570002594012103738,
        "l6.bias@tmp_01:_cuda": 9450985108464699594,
        "l6.bias@tmp_02:_cuda": 12761177678137984591,
        "l6.bias@tmp_03:_cuda": 10062960983824319812,
        "l6.weight@tmp_01:_cuda": 2697666990541111930,
        "l6.weight@tmp_02:_cuda": 9828100084367740348,
        "l6.weight@tmp_03:_cuda": 12558046792785246346,
        "l5.bias@tmp_01:_cuda": 3984989281046440853,
        "l5.bias@tmp_02:_cuda": 14175501620977530004,
        "l5.bias@tmp_03:_cuda": 16280171136160255367,
        "l5.weight@tmp_01:_cuda": 2298764051101003173,
        "l5.weight@tmp_02:_cuda": 9978805944123614295,
        "l5.weight@tmp_03:_cuda": 9804987375067244802,
        "l4.bias@tmp_01:_cuda": 8059697744769285346,
        "l4.bias@tmp_02:_cuda": 16794419863405606831,
        "l4.bias@tmp_03:_cuda": 1740899175180104906,
        "l4.weight@tmp_01:_cuda": 4028045814499437478,
        "l4.weight@tmp_02:_cuda": 4493571174309258414,
        "l4.weight@tmp_03:_cuda": 8693758502437397098,
        "l3.bias@tmp_01:_cuda": 326176784737553547,
        "l3.bias@tmp_02:_cuda": 11021727640139347692,
        "l3.bias@tmp_03:_cuda": 2414340309477185406,
        "l3.weight@tmp_01:_cuda": 6574385511971661816,
        "l3.weight@tmp_02:_cuda": 5254762686172374416,
        "l3.weight@tmp_03:_cuda": 15028112410401788388,
        "l2.bias@tmp_01:_cuda": 1643075040453797470,
        "l2.bias@tmp_02:_cuda": 4847542213627584989,
        "l2.bias@tmp_03:_cuda": 12006722169889691697,
        "l2.weight@tmp_01:_cuda": 574362251922033701,
        "l2.weight@tmp_02:_cuda": 1226348894113837268,
        "l2.weight@tmp_03:_cuda": 5851869513799547089,
        "l1.bias@tmp_01:_cuda": 13764129569894208041,
        "l1.bias@tmp_02:_cuda": 10154535458483573161,
        "l1.bias@tmp_03:_cuda": 16164967312520000707,
        "l1.weight@tmp_01:_cuda": 12862676105375032849,
        "l1.weight@tmp_02:_cuda": 13045529039499391803,
        "l1.weight@tmp_03:_cuda": 13398558236456217823,
    },
}

CONFIG_DARE_DISJOINT_TIES1_CPU = {
    "sd_base_path": 711244,
    "sd_merged_paths": {
        "tmp_01": 235622,
        "tmp_02": 498235,
        "tmp_03": 713341,
    },
    "sd_output_path": "tests/data_pth/dare_disjoint_ties1_cpu.pth",
    "method": "dare_disjoint",
    "lambda_param": 0.4,
    "sparsity": 0.93,
    "use_ties": True,
    "weight_batch_size": 2,
    "weight_batches_custom": None,
    "merge_device": "cpu",
    "seed_dict": {
        "l7.bias@PERMUTATION:_cpu": 2162373395208094919,
        "l7.bias@tmp_01:_cpu": 7191397968257941805,
        "l7.bias@tmp_02:_cpu": 4214196314146985430,
        "l7.bias@tmp_03:_cpu": 17456168674928492095,
        "l7.weight@PERMUTATION:_cpu": 97528326056444312,
        "l7.weight@tmp_01:_cpu": 9680104493501248446,
        "l7.weight@tmp_02:_cpu": 8973581098706447572,
        "l7.weight@tmp_03:_cpu": 14379631313972717999,
        "l6.bias@PERMUTATION:_cpu": 7346665847600021615,
        "l6.bias@tmp_01:_cpu": 11073293763631692991,
        "l6.bias@tmp_02:_cpu": 1948987271808233637,
        "l6.bias@tmp_03:_cpu": 18227396744452486810,
        "l6.weight@PERMUTATION:_cpu": 15582988562473469415,
        "l6.weight@tmp_01:_cpu": 10829408723561149670,
        "l6.weight@tmp_02:_cpu": 4500232068258453304,
        "l6.weight@tmp_03:_cpu": 2554339886940451816,
        "l5.bias@PERMUTATION:_cpu": 18061188843863160616,
        "l5.bias@tmp_01:_cpu": 1492528569562334329,
        "l5.bias@tmp_02:_cpu": 14071432904713036225,
        "l5.bias@tmp_03:_cpu": 7253268356198535662,
        "l5.weight@PERMUTATION:_cpu": 7279186812790435029,
        "l5.weight@tmp_01:_cpu": 15858082865457292614,
        "l5.weight@tmp_02:_cpu": 7046951101188563553,
        "l5.weight@tmp_03:_cpu": 5708892982093580542,
        "l4.bias@PERMUTATION:_cpu": 7504653615842286040,
        "l4.bias@tmp_01:_cpu": 440437549950131792,
        "l4.bias@tmp_02:_cpu": 2501723909323623518,
        "l4.bias@tmp_03:_cpu": 9092949033819014929,
        "l4.weight@PERMUTATION:_cpu": 14899497832443616743,
        "l4.weight@tmp_01:_cpu": 5243098016659792220,
        "l4.weight@tmp_02:_cpu": 5693395418897665961,
        "l4.weight@tmp_03:_cpu": 1561964547295283278,
        "l3.bias@PERMUTATION:_cpu": 8503513188289177520,
        "l3.bias@tmp_01:_cpu": 6694382510077039714,
        "l3.bias@tmp_02:_cpu": 8679771790593752463,
        "l3.bias@tmp_03:_cpu": 1376888356622769513,
        "l3.weight@PERMUTATION:_cpu": 4107643840997258442,
        "l3.weight@tmp_01:_cpu": 15266239771705433839,
        "l3.weight@tmp_02:_cpu": 11850810861138751659,
        "l3.weight@tmp_03:_cpu": 12908921732832449768,
        "l2.bias@PERMUTATION:_cpu": 7760346017254920686,
        "l2.bias@tmp_01:_cpu": 13714942719600512852,
        "l2.bias@tmp_02:_cpu": 14511138504191747046,
        "l2.bias@tmp_03:_cpu": 1777284622182025863,
        "l2.weight@PERMUTATION:_cpu": 16040302337023434743,
        "l2.weight@tmp_01:_cpu": 17403488564483059391,
        "l2.weight@tmp_02:_cpu": 10929452644391790019,
        "l2.weight@tmp_03:_cpu": 4735975861204360136,
        "l1.bias@PERMUTATION:_cpu": 7472051926754276617,
        "l1.bias@tmp_01:_cpu": 9706651031592892071,
        "l1.bias@tmp_02:_cpu": 1403671784012127205,
        "l1.bias@tmp_03:_cpu": 5739922109398273593,
        "l1.weight@PERMUTATION:_cpu": 13279650090715803989,
        "l1.weight@tmp_01:_cpu": 6370559973695197432,
        "l1.weight@tmp_02:_cpu": 3269979856447593260,
        "l1.weight@tmp_03:_cpu": 1556950194805472179,
    },
}

CONFIG_DARE_DISJOINT_TIES1_CUDA = {
    "sd_base_path": 711244,
    "sd_merged_paths": {
        "tmp_01": 235622,
        "tmp_02": 498235,
        "tmp_03": 713341,
    },
    "sd_output_path": "tests/data_pth/dare_disjoint_ties1_cuda.pth",
    "method": "dare_disjoint",
    "lambda_param": 0.4,
    "sparsity": 0.93,
    "use_ties": True,
    "weight_batch_size": 2,
    "weight_batches_custom": None,
    "merge_device": "cuda",
    "seed_dict": {
        "l7.bias@PERMUTATION:_cuda": 13627145580628697451,
        "l7.bias@tmp_01:_cuda": 12639469056006059335,
        "l7.bias@tmp_02:_cuda": 38476546806984663,
        "l7.bias@tmp_03:_cuda": 16958736660636992629,
        "l7.weight@PERMUTATION:_cuda": 4724678618133121516,
        "l7.weight@tmp_01:_cuda": 15411514603262612709,
        "l7.weight@tmp_02:_cuda": 10108625636668705499,
        "l7.weight@tmp_03:_cuda": 15470041691103312174,
        "l6.bias@PERMUTATION:_cuda": 13564450540069631360,
        "l6.bias@tmp_01:_cuda": 3187527220445683845,
        "l6.bias@tmp_02:_cuda": 15724048495558145419,
        "l6.bias@tmp_03:_cuda": 16641792721957661076,
        "l6.weight@PERMUTATION:_cuda": 14559343211174064499,
        "l6.weight@tmp_01:_cuda": 10956870014658862164,
        "l6.weight@tmp_02:_cuda": 480238990238116162,
        "l6.weight@tmp_03:_cuda": 1064166206609372243,
        "l5.bias@PERMUTATION:_cuda": 7509027204673523440,
        "l5.bias@tmp_01:_cuda": 9361085114040681934,
        "l5.bias@tmp_02:_cuda": 8453655862058450767,
        "l5.bias@tmp_03:_cuda": 8931970222386262734,
        "l5.weight@PERMUTATION:_cuda": 1622729829068329830,
        "l5.weight@tmp_01:_cuda": 1539572799336810741,
        "l5.weight@tmp_02:_cuda": 15988106402761420023,
        "l5.weight@tmp_03:_cuda": 8003368595672627933,
        "l4.bias@PERMUTATION:_cuda": 7349434621025723240,
        "l4.bias@tmp_01:_cuda": 5574462368387529796,
        "l4.bias@tmp_02:_cuda": 8920964640545589538,
        "l4.bias@tmp_03:_cuda": 1985595326750782147,
        "l4.weight@PERMUTATION:_cuda": 10196902028991334473,
        "l4.weight@tmp_01:_cuda": 560512598009046620,
        "l4.weight@tmp_02:_cuda": 17495412158687025691,
        "l4.weight@tmp_03:_cuda": 2395653389287357549,
        "l3.bias@PERMUTATION:_cuda": 12477975059912578273,
        "l3.bias@tmp_01:_cuda": 3942761916943177232,
        "l3.bias@tmp_02:_cuda": 7806721248687474248,
        "l3.bias@tmp_03:_cuda": 18280237848942431278,
        "l3.weight@PERMUTATION:_cuda": 4970482025381602005,
        "l3.weight@tmp_01:_cuda": 1091248569956911843,
        "l3.weight@tmp_02:_cuda": 4992084937734427057,
        "l3.weight@tmp_03:_cuda": 10679603338303654219,
        "l2.bias@PERMUTATION:_cuda": 11748068382809249208,
        "l2.bias@tmp_01:_cuda": 8601206695915879091,
        "l2.bias@tmp_02:_cuda": 207076128928871727,
        "l2.bias@tmp_03:_cuda": 11387247721104550726,
        "l2.weight@PERMUTATION:_cuda": 18111576218503767610,
        "l2.weight@tmp_01:_cuda": 1237369697423556424,
        "l2.weight@tmp_02:_cuda": 15454203668153466811,
        "l2.weight@tmp_03:_cuda": 15554335975216545945,
        "l1.bias@PERMUTATION:_cuda": 10088646120833517274,
        "l1.bias@tmp_01:_cuda": 8537392746781925246,
        "l1.bias@tmp_02:_cuda": 12169446733870722341,
        "l1.bias@tmp_03:_cuda": 15216353639000307125,
        "l1.weight@PERMUTATION:_cuda": 5651297784296027235,
        "l1.weight@tmp_01:_cuda": 4709787273439132332,
        "l1.weight@tmp_02:_cuda": 8605774705415113877,
        "l1.weight@tmp_03:_cuda": 2281385119557049123,
    },
}
