import datetime
import tifffile

print(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))

iomat = tifffile.imread('/Users/changyunpeng/CODE_BACKUP/GF2_PMS1_E113.4_N23.3_20170915_L1A0002600401/GF2_PMS1_E113.4_N23.3_20170915_L1A0002600401-MSS1.tif')
print(iomat.shape)
