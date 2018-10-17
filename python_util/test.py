import datetime
import tifffile

print(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))

iomat = tifffile.imread('/Users/changyunpeng/CODE_BACKUP/GF2_PMS1_E113.4_N23.3_20170915_L1A0002600401/GF2_PMS1_E113.4_N23.3_20170915_L1A0002600401-MSS1.tif')
print(iomat.shape)
"block_lat1": 22.70506056480798, "block_lon1": 113.89879130296187, 
"block_lat2": 22.69854275204969, "block_lon2": 113.93504680614278,
"block_lat3": 22.668499103466488, "block_lon3": 113.92848666057257,
"block_lat4": 22.675014653967132, "block_lon4": 113.89223841300341