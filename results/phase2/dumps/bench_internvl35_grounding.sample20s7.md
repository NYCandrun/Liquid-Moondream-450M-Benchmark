# Phase 2 dump — bench_internvl35_grounding.sample20s7

source: `results/raw/bench/bench_internvl35_grounding.sample20s7.jsonl`  items: 20

**cell summary: parse 20/20  correct 17/20**


---
**item** `297363:216168`  image `COCO_train2014_000000297363.jpg`  (640x480)
**prompt:** 'Please provide the bounding box coordinate of the region this sentence describes: <ref>yellow shirt guy</ref>'
**raw output:** 'yellow shirt guy[[662, 193, 999, 989]]'
**parsed:** `[[423.68, 92.64, 639.36, 474.71999999999997]]`
**gt:** `{'box_xyxy': [401.989990234375, 91.83000183105469, 639.2799835205078, 473.22999572753906], 'category': 'person'}`
**score:** `{'iou': 0.9031522936694446, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 0.9105505775213522, 'miss_kind': None}`  ✅
**latency:** 2292.6 ms

---
**item** `68812:1259562`  image `COCO_train2014_000000068812.jpg`  (640x427)
**prompt:** 'Please provide the bounding box coordinate of the region this sentence describes: <ref>man in very front</ref>'
**raw output:** 'man in very front[[128, 260, 512, 989]]'
**parsed:** `[[81.92, 111.02000000000001, 327.68, 422.303]]`
**gt:** `{'box_xyxy': [87.31999969482422, 105.55000305175781, 327.20999908447266, 420.28001403808594], 'category': 'person'}`
**score:** `{'iou': 0.9534175773792168, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 1.0132492744848718, 'miss_kind': None}`  ✅
**latency:** 1552.3 ms

---
**item** `495344:1394064`  image `COCO_train2014_000000495344.jpg`  (640x426)
**prompt:** 'Please provide the bounding box coordinate of the region this sentence describes: <ref>white bench on right all of the bench is shown</ref>'
**raw output:** 'white bench on right all of the bench is shown[[856, 762, 1000, 989]]'
**parsed:** `[[547.84, 324.612, 640.0, 421.314]]`
**gt:** `{'box_xyxy': [379.6000061035156, 305.9800109863281, 601.8899993896484, 426.00000762939453], 'category': 'bench'}`
**score:** `{'iou': 0.17213302174182096, 'hit_at_50': False, 'center_in_box': True, 'area_ratio': 0.33404455677831985, 'miss_kind': 'found_but_loose'}`  ❌
**latency:** 1977.0 ms

---
**item** `182347:459025`  image `COCO_train2014_000000182347.jpg`  (500x333)
**prompt:** 'Please provide the bounding box coordinate of the region this sentence describes: <ref>man wearing black shirt</ref>'
**raw output:** 'man wearing black shirt[[278, 245, 470, 984]]'
**parsed:** `[[139.0, 81.585, 235.0, 327.67199999999997]]`
**gt:** `{'box_xyxy': [141.42999267578125, 81.33999633789062, 236.4699935913086, 326.7899932861328], 'category': 'person'}`
**score:** `{'iou': 0.9556710243960154, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 1.0127224606591627, 'miss_kind': None}`  ✅
**latency:** 1646.7 ms

---
**item** `463505:458610`  image `COCO_train2014_000000463505.jpg`  (500x375)
**prompt:** 'Please provide the bounding box coordinate of the region this sentence describes: <ref>woman</ref>'
**raw output:** 'woman[[59, 120, 553, 987]]'
**parsed:** `[[29.5, 45.0, 276.5, 370.125]]`
**gt:** `{'box_xyxy': [32.09000015258789, 50.5099983215332, 283.78000259399414, 368.9200019836426], 'category': 'person'}`
**score:** `{'iou': 0.9418895541336161, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 1.002062130440879, 'miss_kind': None}`  ✅
**latency:** 1615.9 ms

---
**item** `562063:286973`  image `COCO_train2014_000000562063.jpg`  (468x640)
**prompt:** 'Please provide the bounding box coordinate of the region this sentence describes: <ref>right umbrella</ref>'
**raw output:** 'right umbrella[[685, 76, 955, 340]]'
**parsed:** `[[320.58000000000004, 48.64, 446.94, 217.60000000000002]]`
**gt:** `{'box_xyxy': [317.4800109863281, 47.27000045776367, 447.15000915527344, 218.26000595092773], 'category': 'umbrella'}`
**score:** `{'iou': 0.9629046569438672, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 0.9629046569438672, 'miss_kind': None}`  ✅
**latency:** 1603.9 ms

---
**item** `203034:1646741`  image `COCO_train2014_000000203034.jpg`  (640x480)
**prompt:** 'Please provide the bounding box coordinate of the region this sentence describes: <ref>short white fridge</ref>'
**raw output:** 'short white fridge[[523, 170, 707, 872]]'
**parsed:** `[[334.72, 81.60000000000001, 452.47999999999996, 418.56]]`
**gt:** `{'box_xyxy': [244.4499969482422, 310.17999267578125, 325.9499969482422, 480.0], 'category': 'refrigerator'}`
**score:** `{'iou': 0.0, 'hit_at_50': False, 'center_in_box': False, 'area_ratio': 2.867013133979574, 'miss_kind': 'found_but_loose'}`  ❌
**latency:** 1640.7 ms

---
**item** `214001:558125`  image `COCO_train2014_000000214001.jpg`  (342x640)
**prompt:** 'Please provide the bounding box coordinate of the region this sentence describes: <ref>girl on phone</ref>'
**raw output:** 'girl on phone[[1, 54, 437, 989]]'
**parsed:** `[[0.342, 34.56, 149.454, 632.96]]`
**gt:** `{'box_xyxy': [8.630000114440918, 28.760000228881836, 172.5799970626831, 629.9299831390381], 'category': 'person'}`
**score:** `{'iou': 0.8065796157042445, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 0.905306168651283, 'miss_kind': None}`  ✅
**latency:** 1249.1 ms

---
**item** `432372:1067573`  image `COCO_train2014_000000432372.jpg`  (640x426)
**prompt:** 'Please provide the bounding box coordinate of the region this sentence describes: <ref>front orange hotdog bottom row center</ref>'
**raw output:** 'front orange hotdog bottom row center[[216, 609, 675, 927]]'
**parsed:** `[[138.24, 259.43399999999997, 432.0, 394.90200000000004]]`
**gt:** `{'box_xyxy': [156.3699951171875, 272.4700012207031, 445.0400085449219, 397.77000427246094], 'category': 'hot dog'}`
**score:** `{'iou': 0.7992972859520125, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 1.1002126260722287, 'miss_kind': None}`  ✅
**latency:** 1662.2 ms

---
**item** `565220:1068936`  image `COCO_train2014_000000565220.jpg`  (500x430)
**prompt:** 'Please provide the bounding box coordinate of the region this sentence describes: <ref>hot dog on top</ref>'
**raw output:** 'hot dog on top[[0, 40, 974, 482]]'
**parsed:** `[[0.0, 17.2, 487.0, 207.26]]`
**gt:** `{'box_xyxy': [3.0399999618530273, 15.180000305175781, 484.64000606536865, 239.79000091552734], 'category': 'hot dog'}`
**score:** `{'iou': 0.8382248541589857, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 0.8556656803819855, 'miss_kind': None}`  ✅
**latency:** 1268.4 ms

---
**item** `312943:595522`  image `COCO_train2014_000000312943.jpg`  (510x640)
**prompt:** 'Please provide the bounding box coordinate of the region this sentence describes: <ref>giraffe far right</ref>'
**raw output:** 'giraffe far right[[770, 182, 1000, 745]]'
**parsed:** `[[392.7, 116.47999999999999, 510.0, 476.8]]`
**gt:** `{'box_xyxy': [386.4800109863281, 117.83000183105469, 508.1400146484375, 479.27000427246094], 'category': 'giraffe'}`
**score:** `{'iou': 0.9250131172257685, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 0.9611747186793964, 'miss_kind': None}`  ✅
**latency:** 1873.4 ms

---
**item** `472654:1257559`  image `COCO_train2014_000000472654.jpg`  (640x640)
**prompt:** 'Please provide the bounding box coordinate of the region this sentence describes: <ref>woman right front</ref>'
**raw output:** 'woman right front[[670, 156, 812, 830]]'
**parsed:** `[[428.8, 99.84, 519.6800000000001, 531.1999999999999]]`
**gt:** `{'box_xyxy': [417.0799865722656, 100.43000030517578, 526.3799896240234, 537.6500015258789], 'category': 'person'}`
**score:** `{'iou': 0.8182887103956278, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 0.8203288655725033, 'miss_kind': None}`  ✅
**latency:** 1312.2 ms

---
**item** `244983:172644`  image `COCO_train2014_000000244983.jpg`  (640x480)
**prompt:** 'Please provide the bounding box coordinate of the region this sentence describes: <ref>left train</ref>'
**raw output:** 'left train[[118, 357, 530, 587]]'
**parsed:** `[[75.52, 171.35999999999999, 339.20000000000005, 281.76]]`
**gt:** `{'box_xyxy': [76.58000183105469, 177.97999572753906, 343.00999450683594, 281.5299987792969], 'category': 'train'}`
**score:** `{'iou': 0.9216908580851575, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 1.0551471604755958, 'miss_kind': None}`  ✅
**latency:** 1480.2 ms

---
**item** `152273:328489`  image `COCO_train2014_000000152273.jpg`  (640x480)
**prompt:** 'Please provide the bounding box coordinate of the region this sentence describes: <ref>cell phone closest to top</ref>'
**raw output:** 'cell phone closest to top[[443, 367, 790, 625]]'
**parsed:** `[[283.52, 176.16, 505.6, 300.0]]`
**gt:** `{'box_xyxy': [329.54998779296875, 101.9800033569336, 523.9099884033203, 189.64000701904297], 'category': 'cell phone'}`
**score:** `{'iou': 0.05628014700099976, 'hit_at_50': False, 'center_in_box': False, 'area_ratio': 1.6142173685735053, 'miss_kind': 'found_but_loose'}`  ❌
**latency:** 1608.4 ms

---
**item** `253049:1083402`  image `COCO_train2014_000000253049.jpg`  (612x612)
**prompt:** 'Please provide the bounding box coordinate of the region this sentence describes: <ref>picuture on the bottom left the cocunut white donut</ref>'
**raw output:** 'picuture on the bottom left the cocunut white donut[[110, 530, 422, 989]]'
**parsed:** `[[67.32000000000001, 324.36, 258.264, 605.268]]`
**gt:** `{'box_xyxy': [108.6500015258789, 324.3399963378906, 302.5600051879883, 603.5199890136719], 'category': 'donut'}`
**score:** `{'iou': 0.6327523596721283, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 0.9907991327207807, 'miss_kind': None}`  ✅
**latency:** 1604.7 ms

---
**item** `31812:1086803`  image `COCO_train2014_000000031812.jpg`  (640x426)
**prompt:** 'Please provide the bounding box coordinate of the region this sentence describes: <ref>cake with cherries on top</ref>'
**raw output:** 'cake with cherries on top[[0, 161, 190, 540]]'
**parsed:** `[[0.0, 68.586, 121.6, 230.04000000000002]]`
**gt:** `{'box_xyxy': [0.0, 65.0999984741211, 122.52999877929688, 240.27999114990234], 'category': 'cake'}`
**score:** `{'iou': 0.9146510788508151, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 0.9146510788508151, 'miss_kind': None}`  ✅
**latency:** 1368.8 ms

---
**item** `223909:598325`  image `COCO_train2014_000000223909.jpg`  (640x428)
**prompt:** 'Please provide the bounding box coordinate of the region this sentence describes: <ref>right gir</ref>'
**raw output:** 'right gir[[480, 50, 920, 989]]'
**parsed:** `[[307.2, 21.400000000000002, 588.8000000000001, 423.292]]`
**gt:** `{'box_xyxy': [306.2300109863281, 23.56999969482422, 585.6400146484375, 427.99999237060547], 'category': 'giraffe'}`
**score:** `{'iou': 0.9688588171151636, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 1.0015132633903214, 'miss_kind': None}`  ✅
**latency:** 1241.8 ms

---
**item** `158701:486441`  image `COCO_train2014_000000158701.jpg`  (640x427)
**prompt:** 'Please provide the bounding box coordinate of the region this sentence describes: <ref>left skiier</ref>'
**raw output:** 'left skiier[[180, 300, 370, 850]]'
**parsed:** `[[115.19999999999999, 128.1, 236.8, 362.95]]`
**gt:** `{'box_xyxy': [120.0999984741211, 131.94000244140625, 231.54999542236328, 361.6699981689453], 'category': 'person'}`
**score:** `{'iou': 0.8965481649384934, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 1.1153890433411389, 'miss_kind': None}`  ✅
**latency:** 1332.7 ms

---
**item** `421488:238044`  image `COCO_train2014_000000421488.jpg`  (640x460)
**prompt:** 'Please provide the bounding box coordinate of the region this sentence describes: <ref>man right</ref>'
**raw output:** 'man right[[660, 52, 920, 830]]'
**parsed:** `[[422.40000000000003, 23.919999999999998, 588.8000000000001, 381.79999999999995]]`
**gt:** `{'box_xyxy': [414.4100036621094, 25.3799991607666, 593.6500091552734, 392.1400089263916], 'category': 'person'}`
**score:** `{'iou': 0.8988690551539542, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 0.9058865810027202, 'miss_kind': None}`  ✅
**latency:** 1424.2 ms

---
**item** `163828:2165500`  image `COCO_train2014_000000163828.jpg`  (640x512)
**prompt:** 'Please provide the bounding box coordinate of the region this sentence describes: <ref>batter</ref>'
**raw output:** 'batter[[561, 210, 780, 803]]'
**parsed:** `[[359.04, 107.52, 499.20000000000005, 411.136]]`
**gt:** `{'box_xyxy': [380.32000732421875, 108.69000244140625, 525.9300079345703, 413.1000061035156], 'category': 'person'}`
**score:** `{'iou': 0.7060314167372181, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 0.96006053841398, 'miss_kind': None}`  ✅
**latency:** 1476.4 ms
