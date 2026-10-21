from utee import misc
print = misc.logger.info
import torch.nn as nn
from modules.quantization_cpu_np_infer import QConv2d,  QLinear
from modules.floatrange_cpu_np_infer import FConv2d, FLinear
import torch
from torchvision.models import mobilenet_v2


def quantize_layers(module, args, logger, counter=None):
    # Swap every nn.Conv2d / nn.Linear for its NeuroSim counterpart in place,
    # keeping the attribute names so torchvision's state_dict keys still match.
    if counter is None:
        counter = [0]
    for child_name, child in module.named_children():
        i = counter[0]
        if isinstance(child, nn.Conv2d):
            counter[0] += 1
            if args.mode == "WAGE":
                new = QConv2d(child.in_channels, child.out_channels, kernel_size=child.kernel_size, stride=child.stride,
                              padding=child.padding, groups=child.groups, bias=child.bias is not None,
                              logger=logger,wl_input = args.wl_activate,wl_activate=args.wl_activate,
                              wl_error=args.wl_error,wl_weight= args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                              subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target,
                              name = 'Conv'+str(i)+'_', model = args.model, parallelRead=args.parallelRead)
            elif args.mode == "FP":
                new = FConv2d(child.in_channels, child.out_channels, kernel_size=child.kernel_size, stride=child.stride,
                              padding=child.padding, groups=child.groups, bias=child.bias is not None,
                              logger=logger,wl_input = args.wl_activate,wl_weight= args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                              subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target,
                              name = 'Conv'+str(i)+'_' )
            setattr(module, child_name, new)
        elif isinstance(child, nn.Linear):
            counter[0] += 1
            if args.mode == "WAGE":
                new = QLinear(in_features=child.in_features, out_features=child.out_features, bias=child.bias is not None,
                              logger=logger, wl_input = args.wl_activate,wl_activate=args.wl_activate,wl_error=args.wl_error,
                              wl_weight=args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                              subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target,
                              name='FC'+str(i)+'_', model = args.model, parallelRead=args.parallelRead)
            elif args.mode == "FP":
                new = FLinear(in_features=child.in_features, out_features=child.out_features, bias=child.bias is not None,
                              logger=logger,wl_input = args.wl_activate,wl_weight= args.wl_weight,inference=args.inference,onoffratio=args.onoffratio,cellBit=args.cellBit,
                              subArray=args.subArray,ADCprecision=args.ADCprecision,vari=args.vari,t=args.t,v=args.v,detect=args.detect,target=args.target,
                              name='FC'+str(i)+'_')
            setattr(module, child_name, new)
        else:
            quantize_layers(child, args, logger, counter)
    return module


def mobilenetv2( args, logger, pretrained=None):
    # torchvision mobilenet_v2 with classifier[1] = nn.Linear(1280, 10), as in 662_final_proj.ipynb
    model = mobilenet_v2(weights=None, num_classes=10)
    model = quantize_layers(model, args, logger)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained, map_location='cpu'))
    return model
