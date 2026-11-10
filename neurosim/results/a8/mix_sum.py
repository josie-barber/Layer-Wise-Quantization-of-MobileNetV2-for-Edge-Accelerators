# mix_sum.py: mixed-precision NeuroSim numbers from the W8A8 and W4A8 logs (split-and-sum)
# energy=sum of per-layer energy (exact, same as NeuroSim's pipeline total)
# latency=max per-layer latency (NeuroSim's pipeline clock cycle)
# area=estimate: W4 chip area + share of the extra W8 tiles for the layers kept at 8 bits
# run from ~/team-repo:  python neurosim/results/a8/mix_sum.py
import re,json,csv,os

R='neurosim/results/a8'
TAG='unscaled'  # change to 'scaled' after the dec2bin fix
LILLY='ECE662_results_A8/mixed_precision_configs.json'
LILLY_ACC='ECE662_results_A8/mixed_precision_A8.csv'
AMANDA='results/amanda_neurosim_configs.json'

def parse(f):
    t=open(f).read()
    lat,en={},{}
    # pipeline per-layer blocks: "layerN's readLatency is: Xns" / "readDynamicEnergy is: XpJ"
    for blk in t.split('Estimation of Layer ')[1:]:
        i=int(blk.split()[0])
        lat[i]=float(re.search(r"'s readLatency is: ([\d.e+-]+)ns",blk).group(1))
        en[i]=float(re.search(r"'s readDynamicEnergy is: ([\d.e+-]+)pJ",blk).group(1))
    tiles={int(a):float(b) for a,b in re.findall(r'^layer(\d+): ([\d.e+-]+)$',t.split('# of tile used for each layer')[1].split('Speed-up')[0],re.M)}
    area=float(re.search(r'ChipArea : ([\d.e+-]+)um\^2',t).group(1))
    chip_en=float(re.search(r'pipeline-system readDynamicEnergy \(per image\) is: ([\d.e+-]+)pJ',t).group(1))
    chip_lat=float(re.search(r'pipeline-system-clock-cycle \(per image\) is: ([\d.e+-]+)ns',t).group(1))
    return dict(lat=lat,en=en,tiles=tiles,area=area,chip_en=chip_en,chip_lat=chip_lat)

def combine(bits,order,L8,L4):
    n=len(order)
    pick=lambda i:L8 if bits[order[i-1]]==8 else L4  # NeuroSim layer i = order[i-1]
    en=sum(pick(i)['en'][i] for i in range(1,n+1))
    lat=max(pick(i)['lat'][i] for i in range(1,n+1))
    extra=[L8['tiles'][i]-L4['tiles'][i] for i in range(1,n+1)]
    frac=sum(e for i,e in enumerate(extra,1) if bits[order[i-1]]==8)/sum(extra)
    area=L4['area']+frac*(L8['area']-L4['area'])
    return area/1e6,lat/1e3,en/1e6,1e9/lat  # mm2, us, uJ, fps

# configs: (source, name, bits, accuracy)
lj=json.load(open(LILLY))
order=lj['layer_order']
acc={r['config']:r['test_accuracy'] for r in csv.DictReader(open(LILLY_ACC))}
cfgs=[('lilly',k,v,acc.get(k,'')) for k,v in lj['configs'].items()]
for k,v in json.load(open(AMANDA)).items():
    cfgs.append(('amanda',k,v['weight_bits'],v['a8_test_accuracy']))

rows=[]
for m in ['depth1','dense']:
    L8=parse(f'{R}/mbv2_w8a8_{m}_{TAG}.txt')
    L4=parse(f'{R}/mbv2_w4a8_{m}_{TAG}.txt')
    # sanity check: all-8 and all-4 must reproduce NeuroSim's own chip totals
    for L,b in [(L8,8),(L4,4)]:
        a,lat,en,_=combine({k:b for k in order},order,L8,L4)
        print(f'check {m} W{b}: energy {en:.4f} vs {L["chip_en"]/1e6:.4f} uJ, latency {lat:.3f} vs {L["chip_lat"]/1e3:.3f} us')
    for src,name,bits,a8 in cfgs:
        a,lat,en,fps=combine(bits,order,L8,L4)
        n8=sum(1 for k in order if bits[k]==8)
        rows.append([src,name,n8,m,round(a,2),round(lat,2),round(en,3),round(fps,1),a8,TAG])

out=f'{R}/mbv2_mixed_a8_{TAG}.csv'
with open(out,'w',newline='') as f:
    w=csv.writer(f)
    w.writerow(['source','config','layers_w8','depthwise_mapping','chip_area_mm2_est','latency_us','dynamic_energy_uJ','fps','a8_test_accuracy','act_encoding'])
    w.writerows(rows)
print('wrote',out,len(rows),'rows')
