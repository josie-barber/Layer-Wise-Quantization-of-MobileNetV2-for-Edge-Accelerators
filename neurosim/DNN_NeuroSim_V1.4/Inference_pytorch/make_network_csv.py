"""Write NeuroSIM/NetWork_<model>.csv from a single forward pass.

Columns (same as NetWork_VGG8.csv):
IFM height, IFM width, IFM channels, kernel height, kernel width, OFM channels, max-pool after (0/1), stride
Grouped (depthwise) convs are listed as dense convs (--dw_mapping dense, block-diagonal weights in utee/hook.py)
or with IFM depth 1 (--dw_mapping depth1, the real k*k x C weights).
Padded convs list the padded IFM size so NeuroSIM's (H-k+1)/stride gives the true number of output positions.
"""
import argparse
import os
from types import SimpleNamespace
import torch
from utee import misc

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='MobileNetV2')
parser.add_argument('--input_size', type=int, default=64)
parser.add_argument('--dw_mapping', default='dense', help='dense|depth1, must match inference.py --dw_mapping')
cli = parser.parse_args()

misc.logger.init(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log/default'), 'make_network_csv')
args = SimpleNamespace(model=cli.model, mode='WAGE', wl_weight=8, wl_activate=8, wl_error=8, inference=0,
                       subArray=128, parallelRead=128, ADCprecision=5, cellBit=1, onoffratio=10,
                       vari=0, t=0, v=0, detect=0, target=0, cuda=False)

assert cli.model == 'MobileNetV2', 'only MobileNetV2 is wired up here'
from models import MobileNetV2
from modules.quantization_cpu_np_infer import QConv2d, QLinear
model = MobileNetV2.mobilenetv2(args=args, logger=misc.logger.info,
                                pretrained=os.path.expanduser('~/neurosim/mobilenetv2_cifar10.pth')).eval()

rows, fired = [], []
def record(layer, inp, out):
    fired.append(layer.name)
    x = inp[0]
    if isinstance(layer, QConv2d):
        # padded IFM size: NeuroSIM counts output positions as (H-k+1)/stride and ignores padding
        depth = 1 if (layer.groups > 1 and cli.dw_mapping == 'depth1') else layer.in_channels
        rows.append([x.shape[2] + 2*layer.padding[0], x.shape[3] + 2*layer.padding[1], depth, layer.kernel_size[0], layer.kernel_size[1],
                     layer.out_channels, 0, layer.stride[0]])
    else:
        rows.append([1, 1, layer.in_features, 1, 1, layer.out_features, 0, 1])

layers = [m for m in model.modules() if isinstance(m, (QConv2d, QLinear))]
handles = [m.register_forward_hook(record) for m in layers]
with torch.no_grad():
    model(torch.zeros(1, 3, cli.input_size, cli.input_size))
for h in handles:
    h.remove()

# utee/hook.py lists layer files in execution order; NeuroSIM pairs CSV rows with them by position
assert fired == [m.name for m in layers], 'execution order differs from module order'

out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'NeuroSIM', 'NetWork_' + cli.model + ('_depth1' if cli.dw_mapping == 'depth1' else '') + '.csv')
with open(out_file, 'w') as f:
    for r in rows:
        f.write(','.join(str(int(v)) for v in r) + '\n')
print('wrote {} layers to {}'.format(len(rows), out_file))
