# Phase 2 dump — deployed_moondream05b_asshipped_grounding.sample20s7

source: `results/raw/deployed/deployed_moondream05b_asshipped_grounding.sample20s7.jsonl`  items: 20

**cell summary: parse 20/20  correct 6/20**


---
**item** `297363:216168`  image `COCO_train2014_000000297363.jpg`  (640x480)
**prompt:** 'yellow shirt guy'
**raw output:** '[{"x_min": -0.00390625, "y_min": 0.66162109375, "x_max": 0.19140625, "y_max": 0.99853515625}, {"x_min": 0.6005859375, "y_min": 0.4892578125, "x_max": 1.0107421875, "y_max": 1.0029296875}]'
**parsed:** `[[0.0, 317.578125, 122.5, 479.296875], [384.375, 234.84375, 640.0, 480.0]]`
**gt:** `{'box_xyxy': [401.989990234375, 91.83000183105469, 639.2799835205078, 473.22999572753906], 'category': 'person'}`
**score:** `{'iou': 0.0, 'hit_at_50': False, 'center_in_box': False, 'area_ratio': 0.21889526066903678, 'miss_kind': 'found_but_loose'}`  ❌
**latency:** 3692.6 ms

---
**item** `68812:1259562`  image `COCO_train2014_000000068812.jpg`  (640x427)
**prompt:** 'man in very front'
**raw output:** '[{"x_min": 0.11279296875, "y_min": 0.19287109375, "x_max": 0.52978515625, "y_max": 1.00048828125}]'
**parsed:** `[[72.1875, 82.35595703125, 339.0625, 427.0]]`
**gt:** `{'box_xyxy': [87.31999969482422, 105.55000305175781, 327.20999908447266, 420.28001403808594], 'category': 'person'}`
**score:** `{'iou': 0.8208647976658217, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 1.2182274143605134, 'miss_kind': None}`  ✅
**latency:** 3121.8 ms

---
**item** `495344:1394064`  image `COCO_train2014_000000495344.jpg`  (640x426)
**prompt:** 'white bench on right all of the bench is shown'
**raw output:** '[]'
**parsed:** `[]`
**gt:** `{'box_xyxy': [379.6000061035156, 305.9800109863281, 601.8899993896484, 426.00000762939453], 'category': 'bench'}`
**score:** `{'iou': 0.0, 'hit_at_50': False, 'center_in_box': False, 'area_ratio': None, 'miss_kind': 'not_found'}`  ❌
**latency:** 2229.3 ms

---
**item** `182347:459025`  image `COCO_train2014_000000182347.jpg`  (500x333)
**prompt:** 'man wearing black shirt'
**raw output:** '[{"x_min": -0.1279296875, "y_min": 0.2451171875, "x_max": 0.8662109375, "y_max": 0.9736328125}]'
**parsed:** `[[0.0, 81.6240234375, 433.10546875, 324.2197265625]]`
**gt:** `{'box_xyxy': [141.42999267578125, 81.33999633789062, 236.4699935913086, 326.7899932861328], 'category': 'person'}`
**score:** `{'iou': 0.218873374108167, 'hit_at_50': False, 'center_in_box': True, 'area_ratio': 4.504092583200669, 'miss_kind': 'found_but_loose'}`  ❌
**latency:** 1776.9 ms

---
**item** `463505:458610`  image `COCO_train2014_000000463505.jpg`  (500x375)
**prompt:** 'woman'
**raw output:** '[{"x_min": 0.05517578125, "y_min": 0.130859375, "x_max": 0.58740234375, "y_max": 0.9921875}]'
**parsed:** `[[27.587890625, 49.072265625, 293.701171875, 372.0703125]]`
**gt:** `{'box_xyxy': [32.09000015258789, 50.5099983215332, 283.78000259399414, 368.9200019836426], 'category': 'person'}`
**score:** `{'iou': 0.9323655588852093, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 1.0725406901512518, 'miss_kind': None}`  ✅
**latency:** 1936.8 ms

---
**item** `562063:286973`  image `COCO_train2014_000000562063.jpg`  (468x640)
**prompt:** 'right umbrella'
**raw output:** '[{"x_min": 0.30859375, "y_min": 0.0322265625, "x_max": 1.302734375, "y_max": 0.3017578125}]'
**parsed:** `[[144.421875, 20.625, 468.0, 193.125]]`
**gt:** `{'box_xyxy': [317.4800109863281, 47.27000045776367, 447.15000915527344, 218.26000595092773], 'category': 'umbrella'}`
**score:** `{'iou': 0.32014460995263616, 'hit_at_50': False, 'center_in_box': False, 'area_ratio': 2.5174335893435007, 'miss_kind': 'found_but_loose'}`  ❌
**latency:** 2772.3 ms

---
**item** `203034:1646741`  image `COCO_train2014_000000203034.jpg`  (640x480)
**prompt:** 'short white fridge'
**raw output:** '[]'
**parsed:** `[]`
**gt:** `{'box_xyxy': [244.4499969482422, 310.17999267578125, 325.9499969482422, 480.0], 'category': 'refrigerator'}`
**score:** `{'iou': 0.0, 'hit_at_50': False, 'center_in_box': False, 'area_ratio': None, 'miss_kind': 'not_found'}`  ❌
**latency:** 2813.5 ms

---
**item** `214001:558125`  image `COCO_train2014_000000214001.jpg`  (342x640)
**prompt:** 'girl on phone'
**raw output:** '[{"x_min": 0.0078125, "y_min": 0.17919921875, "x_max": 0.58203125, "y_max": 0.86767578125}]'
**parsed:** `[[2.671875, 114.6875, 199.0546875, 555.3125]]`
**gt:** `{'box_xyxy': [8.630000114440918, 28.760000228881836, 172.5799970626831, 629.9299831390381], 'category': 'person'}`
**score:** `{'iou': 0.6401315932180227, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 0.8779381256501746, 'miss_kind': None}`  ✅
**latency:** 2527.1 ms

---
**item** `432372:1067573`  image `COCO_train2014_000000432372.jpg`  (640x426)
**prompt:** 'front orange hotdog bottom row center'
**raw output:** '[]'
**parsed:** `[]`
**gt:** `{'box_xyxy': [156.3699951171875, 272.4700012207031, 445.0400085449219, 397.77000427246094], 'category': 'hot dog'}`
**score:** `{'iou': 0.0, 'hit_at_50': False, 'center_in_box': False, 'area_ratio': None, 'miss_kind': 'not_found'}`  ❌
**latency:** 2374.0 ms

---
**item** `565220:1068936`  image `COCO_train2014_000000565220.jpg`  (500x430)
**prompt:** 'hot dog on top'
**raw output:** '[{"x_min": 0.00048828125, "y_min": 0.03662109375, "x_max": 0.99951171875, "y_max": 0.92822265625}]'
**parsed:** `[[0.244140625, 15.7470703125, 499.755859375, 399.1357421875]]`
**gt:** `{'box_xyxy': [3.0399999618530273, 15.180000305175781, 484.64000606536865, 239.79000091552734], 'category': 'hot dog'}`
**score:** `{'iou': 0.5626183352276433, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 1.7703917825422475, 'miss_kind': None}`  ✅
**latency:** 1775.7 ms

---
**item** `312943:595522`  image `COCO_train2014_000000312943.jpg`  (510x640)
**prompt:** 'giraffe far right'
**raw output:** '[{"x_min": 0.7548828125, "y_min": 0.169921875, "x_max": 0.9951171875, "y_max": 0.7421875}]'
**parsed:** `[[384.990234375, 108.75, 507.509765625, 475.0]]`
**gt:** `{'box_xyxy': [386.4800109863281, 117.83000183105469, 508.1400146484375, 479.27000427246094], 'category': 'giraffe'}`
**score:** `{'iou': 0.9476257466545791, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 1.0204668900540923, 'miss_kind': None}`  ✅
**latency:** 3362.1 ms

---
**item** `472654:1257559`  image `COCO_train2014_000000472654.jpg`  (640x640)
**prompt:** 'woman right front'
**raw output:** '[{"x_min": 0.2587890625, "y_min": 0.15185546875, "x_max": 1.2529296875, "y_max": 0.84033203125}]'
**parsed:** `[[165.625, 97.1875, 640.0, 537.8125]]`
**gt:** `{'box_xyxy': [417.0799865722656, 100.43000030517578, 526.3799896240234, 537.6500015258789], 'category': 'person'}`
**score:** `{'iou': 0.22862792124266487, 'hit_at_50': False, 'center_in_box': False, 'area_ratio': 4.373918962149, 'miss_kind': 'found_but_loose'}`  ❌
**latency:** 3593.3 ms

---
**item** `244983:172644`  image `COCO_train2014_000000244983.jpg`  (640x480)
**prompt:** 'left train'
**raw output:** '[{"x_min": 0.05810546875, "y_min": 0.24951171875, "x_max": 0.59033203125, "y_max": 0.34033203125}]'
**parsed:** `[[37.1875, 119.765625, 377.8125, 163.359375]]`
**gt:** `{'box_xyxy': [76.58000183105469, 177.97999572753906, 343.00999450683594, 281.5299987792969], 'category': 'train'}`
**score:** `{'iou': 0.0, 'hit_at_50': False, 'center_in_box': False, 'area_ratio': 0.5382295279696659, 'miss_kind': 'found_but_loose'}`  ❌
**latency:** 3427.4 ms

---
**item** `152273:328489`  image `COCO_train2014_000000152273.jpg`  (640x480)
**prompt:** 'cell phone closest to top'
**raw output:** '[{"x_min": 0.12451171875, "y_min": 0.267578125, "x_max": 0.46337890625, "y_max": 0.50390625}]'
**parsed:** `[[79.6875, 128.4375, 296.5625, 241.875]]`
**gt:** `{'box_xyxy': [329.54998779296875, 101.9800033569336, 523.9099884033203, 189.64000701904297], 'category': 'cell phone'}`
**score:** `{'iou': 0.0, 'hit_at_50': False, 'center_in_box': False, 'area_ratio': 1.4439686442337782, 'miss_kind': 'found_but_loose'}`  ❌
**latency:** 3452.7 ms

---
**item** `253049:1083402`  image `COCO_train2014_000000253049.jpg`  (612x612)
**prompt:** 'picuture on the bottom left the cocunut white donut'
**raw output:** '[{"x_min": 0.1552734375, "y_min": 0.541015625, "x_max": 0.4345703125, "y_max": 0.951171875}]'
**parsed:** `[[95.02734375, 331.1015625, 265.95703125, 582.1171875]]`
**gt:** `{'box_xyxy': [108.6500015258789, 324.3399963378906, 302.5600051879883, 603.5199890136719], 'category': 'donut'}`
**score:** `{'iou': 0.6860623946009131, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 0.7925629171099289, 'miss_kind': None}`  ✅
**latency:** 3730.9 ms

---
**item** `31812:1086803`  image `COCO_train2014_000000031812.jpg`  (640x426)
**prompt:** 'cake with cherries on top'
**raw output:** '[{"x_min": 0.2119140625, "y_min": 0.08837890625, "x_max": 0.4306640625, "y_max": 0.24560546875}]'
**parsed:** `[[135.625, 37.6494140625, 275.625, 104.6279296875]]`
**gt:** `{'box_xyxy': [0.0, 65.0999984741211, 122.52999877929688, 240.27999114990234], 'category': 'cake'}`
**score:** `{'iou': 0.0, 'hit_at_50': False, 'center_in_box': False, 'area_ratio': 0.4368543063039902, 'miss_kind': 'found_but_loose'}`  ❌
**latency:** 2965.0 ms

---
**item** `223909:598325`  image `COCO_train2014_000000223909.jpg`  (640x428)
**prompt:** 'right gir'
**raw output:** '[{"x_min": 0.11279296875, "y_min": 0.2216796875, "x_max": 0.52978515625, "y_max": 0.9970703125}, {"x_min": 0.53759765625, "y_min": 0.0830078125, "x_max": 0.95458984375, "y_max": 0.9892578125}]'
**parsed:** `[[72.1875, 94.87890625, 339.0625, 426.74609375], [344.0625, 35.52734375, 610.9375, 423.40234375]]`
**gt:** `{'box_xyxy': [306.2300109863281, 23.56999969482422, 585.6400146484375, 427.99999237060547], 'category': 'giraffe'}`
**score:** `{'iou': 0.05714514548218016, 'hit_at_50': False, 'center_in_box': False, 'area_ratio': 0.7837668678269281, 'miss_kind': 'found_but_loose'}`  ❌
**latency:** 3895.1 ms

---
**item** `158701:486441`  image `COCO_train2014_000000158701.jpg`  (640x427)
**prompt:** 'left skiier'
**raw output:** '[{"x_min": -0.330078125, "y_min": 0.1865234375, "x_max": 0.6640625, "y_max": 0.4560546875}]'
**parsed:** `[[0.0, 79.6455078125, 425.0, 194.7353515625]]`
**gt:** `{'box_xyxy': [120.0999984741211, 131.94000244140625, 231.54999542236328, 361.6699981689453], 'category': 'person'}`
**score:** `{'iou': 0.10365437871895478, 'hit_at_50': False, 'center_in_box': True, 'area_ratio': 1.910416960412942, 'miss_kind': 'found_but_loose'}`  ❌
**latency:** 3280.5 ms

---
**item** `421488:238044`  image `COCO_train2014_000000421488.jpg`  (640x460)
**prompt:** 'man right'
**raw output:** '[{"x_min": -0.451171875, "y_min": 0.21484375, "x_max": 0.54296875, "y_max": 0.947265625}, {"x_min": -0.1728515625, "y_min": 0.05224609375, "x_max": 0.8212890625, "y_max": 0.85986328125}, {"x_min": -0.00048828125, "y_min": 0.14306640625, "x_max": 0.09228515625, "y_max": 0.53466796875}, {"x_min": -0.451171875, "y_min": 0.12841796875, "x_max": 0.54296875, "y_max": 0.52001953125}]'
**parsed:** `[[0.0, 98.828125, 347.5, 435.7421875], [0.0, 24.033203125, 525.625, 395.537109375], [0.0, 65.810546875, 59.0625, 245.947265625], [0.0, 59.072265625, 347.5, 239.208984375]]`
**gt:** `{'box_xyxy': [414.4100036621094, 25.3799991607666, 593.6500091552734, 392.1400089263916], 'category': 'person'}`
**score:** `{'iou': 0.0, 'hit_at_50': False, 'center_in_box': False, 'area_ratio': 1.7809717192589225, 'miss_kind': 'found_but_loose'}`  ❌
**latency:** 4746.2 ms

---
**item** `163828:2165500`  image `COCO_train2014_000000163828.jpg`  (640x512)
**prompt:** 'batter'
**raw output:** '[{"x_min": 0.6015625, "y_min": 0.15478515625, "x_max": 0.755859375, "y_max": 0.52294921875}]'
**parsed:** `[[385.0, 79.25, 483.75, 267.75]]`
**gt:** `{'box_xyxy': [380.32000732421875, 108.69000244140625, 525.9300079345703, 413.1000061035156], 'category': 'person'}`
**score:** `{'iou': 0.3325512647979113, 'hit_at_50': False, 'center_in_box': True, 'area_ratio': 0.4199507244882898, 'miss_kind': 'found_but_loose'}`  ❌
**latency:** 3128.8 ms
