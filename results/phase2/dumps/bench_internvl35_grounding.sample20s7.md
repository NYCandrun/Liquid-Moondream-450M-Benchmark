# Phase 2 dump — bench_internvl35_grounding.sample20s7

source: `/Users/jacobnikolau/Desktop/Claude/results/raw/bench/bench_internvl35_grounding.sample20s7.jsonl`  items: 20

**cell summary: parse 20/20  correct 15/20**


---
**item** `297363:216168`  image `COCO_train2014_000000297363.jpg`  (640x480)
**prompt:** 'Please provide the bounding box coordinate of the region this sentence describes: <ref>yellow shirt guy</ref>'
**raw output:** 'yellow shirt guy[[660, 190, 915, 989]]'
**parsed:** `[[422.40000000000003, 91.2, 585.6, 474.71999999999997]]`
**gt:** `{'box_xyxy': [401.989990234375, 91.83000183105469, 639.2799835205078, 473.22999572753906], 'category': 'person'}`
**score:** `{'iou': 0.6851467711004031, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 0.6915889813201314, 'miss_kind': None}`  ✅
**latency:** 1703.4 ms

---
**item** `68812:1259562`  image `COCO_train2014_000000068812.jpg`  (640x427)
**prompt:** 'Please provide the bounding box coordinate of the region this sentence describes: <ref>man in very front</ref>'
**raw output:** 'man in very front[[122, 260, 520, 989]]'
**parsed:** `[[78.08, 111.02000000000001, 332.8, 422.303]]`
**gt:** `{'box_xyxy': [87.31999969482422, 105.55000305175781, 327.20999908447266, 420.28001403808594], 'category': 'person'}`
**score:** `{'iou': 0.920426260310671, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 1.0501906542837995, 'miss_kind': None}`  ✅
**latency:** 1171.1 ms

---
**item** `495344:1394064`  image `COCO_train2014_000000495344.jpg`  (640x426)
**prompt:** 'Please provide the bounding box coordinate of the region this sentence describes: <ref>white bench on right all of the bench is shown</ref>'
**raw output:** 'white bench on right all of the bench is shown[[850, 660, 1000, 989]]'
**parsed:** `[[544.0, 281.16, 640.0, 421.314]]`
**gt:** `{'box_xyxy': [379.6000061035156, 305.9800109863281, 601.8899993896484, 426.00000762939453], 'category': 'bench'}`
**score:** `{'iou': 0.19955811837890639, 'hit_at_50': False, 'center_in_box': True, 'area_ratio': 0.5043165344166083, 'miss_kind': 'found_but_loose'}`  ❌
**latency:** 1516.5 ms

---
**item** `182347:459025`  image `COCO_train2014_000000182347.jpg`  (500x333)
**prompt:** 'Please provide the bounding box coordinate of the region this sentence describes: <ref>man wearing black shirt</ref>'
**raw output:** 'man wearing black shirt[[300, 247, 475, 984]]'
**parsed:** `[[150.0, 82.251, 237.5, 327.67199999999997]]`
**gt:** `{'box_xyxy': [141.42999267578125, 81.33999633789062, 236.4699935913086, 326.7899932861328], 'category': 'person'}`
**score:** `{'iou': 0.8938421869510966, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 0.9205562088655526, 'miss_kind': None}`  ✅
**latency:** 1167.3 ms

---
**item** `463505:458610`  image `COCO_train2014_000000463505.jpg`  (500x375)
**prompt:** 'Please provide the bounding box coordinate of the region this sentence describes: <ref>woman</ref>'
**raw output:** 'woman[[74, 106, 550, 989]]'
**parsed:** `[[37.0, 39.75, 275.0, 370.875]]`
**gt:** `{'box_xyxy': [32.09000015258789, 50.5099983215332, 283.78000259399414, 368.9200019836426], 'category': 'person'}`
**score:** `{'iou': 0.9112000906501545, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 0.9833684255386318, 'miss_kind': None}`  ✅
**latency:** 982.7 ms

---
**item** `562063:286973`  image `COCO_train2014_000000562063.jpg`  (468x640)
**prompt:** 'Please provide the bounding box coordinate of the region this sentence describes: <ref>right umbrella</ref>'
**raw output:** 'right umbrella[[687, 62, 942, 350]]'
**parsed:** `[[321.516, 39.68, 440.856, 224.0]]`
**gt:** `{'box_xyxy': [317.4800109863281, 47.27000045776367, 447.15000915527344, 218.26000595092773], 'category': 'umbrella'}`
**score:** `{'iou': 0.8587250196501859, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 0.9920835859421662, 'miss_kind': None}`  ✅
**latency:** 1046.1 ms

---
**item** `203034:1646741`  image `COCO_train2014_000000203034.jpg`  (640x480)
**prompt:** 'Please provide the bounding box coordinate of the region this sentence describes: <ref>short white fridge</ref>'
**raw output:** 'short white fridge[[550, 184, 720, 700]]'
**parsed:** `[[352.0, 88.32, 460.79999999999995, 336.0]]`
**gt:** `{'box_xyxy': [244.4499969482422, 310.17999267578125, 325.9499969482422, 480.0], 'category': 'refrigerator'}`
**score:** `{'iou': 0.0, 'hit_at_50': False, 'center_in_box': False, 'area_ratio': 1.9470332598839364, 'miss_kind': 'found_but_loose'}`  ❌
**latency:** 1095.1 ms

---
**item** `214001:558125`  image `COCO_train2014_000000214001.jpg`  (342x640)
**prompt:** 'Please provide the bounding box coordinate of the region this sentence describes: <ref>girl on phone</ref>'
**raw output:** 'girl on phone[[1, 30, 425, 989]]'
**parsed:** `[[0.342, 19.2, 145.35, 632.96]]`
**gt:** `{'box_xyxy': [8.630000114440918, 28.760000228881836, 172.5799970626831, 629.9299831390381], 'category': 'person'}`
**score:** `{'iou': 0.7800321322545603, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 0.9029877178833097, 'miss_kind': None}`  ✅
**latency:** 947.5 ms

---
**item** `432372:1067573`  image `COCO_train2014_000000432372.jpg`  (640x426)
**prompt:** 'Please provide the bounding box coordinate of the region this sentence describes: <ref>front orange hotdog bottom row center</ref>'
**raw output:** 'front orange hotdog bottom row center[[228, 522, 680, 929]]'
**parsed:** `[[145.92000000000002, 222.372, 435.20000000000005, 395.754]]`
**gt:** `{'box_xyxy': [156.3699951171875, 272.4700012207031, 445.0400085449219, 397.77000427246094], 'category': 'hot dog'}`
**score:** `{'iou': 0.6616862887980741, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 1.3866589628996002, 'miss_kind': None}`  ✅
**latency:** 1198.4 ms

---
**item** `565220:1068936`  image `COCO_train2014_000000565220.jpg`  (500x430)
**prompt:** 'Please provide the bounding box coordinate of the region this sentence describes: <ref>hot dog on top</ref>'
**raw output:** 'hot dog on top[[0, 40, 974, 482]]'
**parsed:** `[[0.0, 17.2, 487.0, 207.26]]`
**gt:** `{'box_xyxy': [3.0399999618530273, 15.180000305175781, 484.64000606536865, 239.79000091552734], 'category': 'hot dog'}`
**score:** `{'iou': 0.8382248541589857, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 0.8556656803819855, 'miss_kind': None}`  ✅
**latency:** 982.8 ms

---
**item** `312943:595522`  image `COCO_train2014_000000312943.jpg`  (510x640)
**prompt:** 'Please provide the bounding box coordinate of the region this sentence describes: <ref>giraffe far right</ref>'
**raw output:** 'giraffe far right[[737, 186, 1000, 752]]'
**parsed:** `[[375.87, 119.03999999999999, 510.0, 481.28]]`
**gt:** `{'box_xyxy': [386.4800109863281, 117.83000183105469, 508.1400146484375, 479.27000427246094], 'category': 'giraffe'}`
**score:** `{'iou': 0.8992729993353087, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 1.1049389626471353, 'miss_kind': None}`  ✅
**latency:** 1203.8 ms

---
**item** `472654:1257559`  image `COCO_train2014_000000472654.jpg`  (640x640)
**prompt:** 'Please provide the bounding box coordinate of the region this sentence describes: <ref>woman right front</ref>'
**raw output:** 'woman right front[[670, 152, 800, 840]]'
**parsed:** `[[428.8, 97.28, 512.0, 537.6]]`
**gt:** `{'box_xyxy': [417.0799865722656, 100.43000030517578, 526.3799896240234, 537.6500015258789], 'category': 'person'}`
**score:** `{'iou': 0.756969235573966, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 0.7666048160739402, 'miss_kind': None}`  ✅
**latency:** 1072.1 ms

---
**item** `244983:172644`  image `COCO_train2014_000000244983.jpg`  (640x480)
**prompt:** 'Please provide the bounding box coordinate of the region this sentence describes: <ref>left train</ref>'
**raw output:** 'left train[[120, 350, 620, 600]]'
**parsed:** `[[76.8, 168.0, 396.8, 288.0]]`
**gt:** `{'box_xyxy': [76.58000183105469, 177.97999572753906, 343.00999450683594, 281.5299987792969], 'category': 'train'}`
**score:** `{'iou': 0.7174401529951818, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 1.39186782460373, 'miss_kind': None}`  ✅
**latency:** 1051.9 ms

---
**item** `152273:328489`  image `COCO_train2014_000000152273.jpg`  (640x480)
**prompt:** 'Please provide the bounding box coordinate of the region this sentence describes: <ref>cell phone closest to top</ref>'
**raw output:** 'cell phone closest to top[[502, 216, 820, 611]]'
**parsed:** `[[321.28, 103.67999999999999, 524.8, 293.28]]`
**gt:** `{'box_xyxy': [329.54998779296875, 101.9800033569336, 523.9099884033203, 189.64000701904297], 'category': 'cell phone'}`
**score:** `{'iou': 0.42929419389480805, 'hit_at_50': False, 'center_in_box': False, 'area_ratio': 2.2648375183356557, 'miss_kind': 'found_but_loose'}`  ❌
**latency:** 1153.2 ms

---
**item** `253049:1083402`  image `COCO_train2014_000000253049.jpg`  (612x612)
**prompt:** 'Please provide the bounding box coordinate of the region this sentence describes: <ref>picuture on the bottom left the cocunut white donut</ref>'
**raw output:** 'picuture on the bottom left the cocunut white donut[[110, 530, 422, 989]]'
**parsed:** `[[67.32000000000001, 324.36, 258.264, 605.268]]`
**gt:** `{'box_xyxy': [108.6500015258789, 324.3399963378906, 302.5600051879883, 603.5199890136719], 'category': 'donut'}`
**score:** `{'iou': 0.6327523596721283, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 0.9907991327207807, 'miss_kind': None}`  ✅
**latency:** 1447.7 ms

---
**item** `31812:1086803`  image `COCO_train2014_000000031812.jpg`  (640x426)
**prompt:** 'Please provide the bounding box coordinate of the region this sentence describes: <ref>cake with cherries on top</ref>'
**raw output:** 'cake with cherries on top[[1, 170, 190, 547]]'
**parsed:** `[[0.64, 72.42, 121.6, 233.02200000000002]]`
**gt:** `{'box_xyxy': [0.0, 65.0999984741211, 122.52999877929688, 240.27999114990234], 'category': 'cake'}`
**score:** `{'iou': 0.9050358744807266, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 0.9050358744807268, 'miss_kind': None}`  ✅
**latency:** 1180.9 ms

---
**item** `223909:598325`  image `COCO_train2014_000000223909.jpg`  (640x428)
**prompt:** 'Please provide the bounding box coordinate of the region this sentence describes: <ref>right gir</ref>'
**raw output:** 'right gir[[487, 43, 940, 989]]'
**parsed:** `[[311.68, 18.404, 601.5999999999999, 423.292]]`
**gt:** `{'box_xyxy': [306.2300109863281, 23.56999969482422, 585.6400146484375, 427.99999237060547], 'category': 'giraffe'}`
**score:** `{'iou': 0.9059289204404058, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 1.0387900350149126, 'miss_kind': None}`  ✅
**latency:** 1026.4 ms

---
**item** `158701:486441`  image `COCO_train2014_000000158701.jpg`  (640x427)
**prompt:** 'Please provide the bounding box coordinate of the region this sentence describes: <ref>left skiier</ref>'
**raw output:** 'left skiier[[170, 306, 400, 852]]'
**parsed:** `[[108.80000000000001, 130.662, 256.0, 363.804]]`
**gt:** `{'box_xyxy': [120.0999984741211, 131.94000244140625, 231.54999542236328, 361.6699981689453], 'category': 'person'}`
**score:** `{'iou': 0.7460525819085254, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 1.340388096294547, 'miss_kind': None}`  ✅
**latency:** 1188.4 ms

---
**item** `421488:238044`  image `COCO_train2014_000000421488.jpg`  (640x460)
**prompt:** 'Please provide the bounding box coordinate of the region this sentence describes: <ref>man right</ref>'
**raw output:** 'man right[[700, 215, 870, 989]]'
**parsed:** `[[448.0, 98.89999999999999, 556.8, 454.94]]`
**gt:** `{'box_xyxy': [414.4100036621094, 25.3799991607666, 593.6500091552734, 392.1400089263916], 'category': 'person'}`
**score:** `{'iou': 0.4396334979239724, 'hit_at_50': False, 'center_in_box': True, 'area_ratio': 0.5892651588310158, 'miss_kind': 'found_but_loose'}`  ❌
**latency:** 1152.4 ms

---
**item** `163828:2165500`  image `COCO_train2014_000000163828.jpg`  (640x512)
**prompt:** 'Please provide the bounding box coordinate of the region this sentence describes: <ref>batter</ref>'
**raw output:** 'batter[[502, 209, 720, 812]]'
**parsed:** `[[321.28, 107.008, 460.79999999999995, 415.744]]`
**gt:** `{'box_xyxy': [380.32000732421875, 108.69000244140625, 525.9300079345703, 413.1000061035156], 'category': 'person'}`
**score:** `{'iou': 0.38948326035997854, 'hit_at_50': False, 'center_in_box': True, 'area_ratio': 0.9717926649315937, 'miss_kind': 'found_but_loose'}`  ❌
**latency:** 1151.3 ms
