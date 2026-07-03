# Phase 2 dump — deployed_lfm450m_gguf_grounding.sample20s7

source: `results/raw/deployed/deployed_lfm450m_gguf_grounding.sample20s7.jsonl`  items: 20

**cell summary: parse 20/20  correct 15/20**


---
**item** `297363:216168`  image `COCO_train2014_000000297363.jpg`  (640x480)
**prompt:** 'Detect all instances of: yellow shirt guy. Response must be a JSON array: [{"label": ..., "bbox": [x1, y1, x2, y2]}, ...]'
**raw output:** '[{"label": "yellow shirt guy", "bbox": [0.63, 0.24, 1.0, 0.99]}]'
**parsed:** `[[403.2, 115.19999999999999, 640.0, 475.2]]`
**gt:** `{'box_xyxy': [401.989990234375, 91.83000183105469, 639.2799835205078, 473.22999572753906], 'category': 'person'}`
**score:** `{'iou': 0.9265240110564462, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 0.9419418504323368, 'miss_kind': None}`  ✅
**latency:** 1427.8 ms

---
**item** `68812:1259562`  image `COCO_train2014_000000068812.jpg`  (640x427)
**prompt:** 'Detect all instances of: man in very front. Response must be a JSON array: [{"label": ..., "bbox": [x1, y1, x2, y2]}, ...]'
**raw output:** '[{"label": "man in very front", "bbox": [0.14, 0.28, 0.52, 0.99]}]'
**parsed:** `[[89.60000000000001, 119.56000000000002, 332.8, 422.73]]`
**gt:** `{'box_xyxy': [87.31999969482422, 105.55000305175781, 327.20999908447266, 420.28001403808594], 'category': 'person'}`
**score:** `{'iou': 0.9186992872590344, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 0.9765612649177612, 'miss_kind': None}`  ✅
**latency:** 1437.8 ms

---
**item** `495344:1394064`  image `COCO_train2014_000000495344.jpg`  (640x426)
**prompt:** 'Detect all instances of: white bench on right all of the bench is shown. Response must be a JSON array: [{"label": ..., "bbox": [x1, y1, x2, y2]}, ...]'
**raw output:** '[{"label": "white bench on right", "bbox": [0.0, 0.67, 0.99, 0.87]}]'
**parsed:** `[[0.0, 285.42, 633.6, 370.62]]`
**gt:** `{'box_xyxy': [379.6000061035156, 305.9800109863281, 601.8899993896484, 426.00000762939453], 'category': 'bench'}`
**score:** `{'iou': 0.21674674624728718, 'hit_at_50': False, 'center_in_box': False, 'area_ratio': 2.023397645683656, 'miss_kind': 'found_but_loose'}`  ❌
**latency:** 1550.7 ms

---
**item** `182347:459025`  image `COCO_train2014_000000182347.jpg`  (500x333)
**prompt:** 'Detect all instances of: man wearing black shirt. Response must be a JSON array: [{"label": ..., "bbox": [x1, y1, x2, y2]}, ...]'
**raw output:** '[{"label": "man wearing black shirt", "bbox": [0.29, 0.3, 0.5, 0.93]}]'
**parsed:** `[[145.0, 99.89999999999999, 250.0, 309.69]]`
**gt:** `{'box_xyxy': [141.42999267578125, 81.33999633789062, 236.4699935913086, 326.7899932861328], 'category': 'person'}`
**score:** `{'iou': 0.7333742045528167, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 0.9442883227983145, 'miss_kind': None}`  ✅
**latency:** 1110.4 ms

---
**item** `463505:458610`  image `COCO_train2014_000000463505.jpg`  (500x375)
**prompt:** 'Detect all instances of: woman. Response must be a JSON array: [{"label": ..., "bbox": [x1, y1, x2, y2]}, ...]'
**raw output:** '[{"label": "woman", "bbox": [0.08, 0.14, 0.55, 0.99]}]'
**parsed:** `[[40.0, 52.50000000000001, 275.0, 371.25]]`
**gt:** `{'box_xyxy': [32.09000015258789, 50.5099983215332, 283.78000259399414, 368.9200019836426], 'category': 'person'}`
**score:** `{'iou': 0.9215564793822165, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 0.9346852451123543, 'miss_kind': None}`  ✅
**latency:** 1235.9 ms

---
**item** `562063:286973`  image `COCO_train2014_000000562063.jpg`  (468x640)
**prompt:** 'Detect all instances of: right umbrella. Response must be a JSON array: [{"label": ..., "bbox": [x1, y1, x2, y2]}, ...]'
**raw output:** '[{"label": "right umbrella", "bbox": [0.63, 0.06, 0.91, 0.38]}]'
**parsed:** `[[294.84, 38.4, 425.88, 243.2]]`
**gt:** `{'box_xyxy': [317.4800109863281, 47.27000045776367, 447.15000915527344, 218.26000595092773], 'category': 'umbrella'}`
**score:** `{'iou': 0.6082347092988539, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 1.2103852028924036, 'miss_kind': None}`  ✅
**latency:** 1386.3 ms

---
**item** `203034:1646741`  image `COCO_train2014_000000203034.jpg`  (640x480)
**prompt:** 'Detect all instances of: short white fridge. Response must be a JSON array: [{"label": ..., "bbox": [x1, y1, x2, y2]}, ...]'
**raw output:** '[{"label": "short white fridge", "bbox": [0.21, 0.5, 0.46, 0.98]}]'
**parsed:** `[[134.4, 240.0, 294.40000000000003, 470.4]]`
**gt:** `{'box_xyxy': [244.4499969482422, 310.17999267578125, 325.9499969482422, 480.0], 'category': 'refrigerator'}`
**score:** `{'iou': 0.18741776524366005, 'hit_at_50': False, 'center_in_box': False, 'area_ratio': 2.6635201913596953, 'miss_kind': 'found_but_loose'}`  ❌
**latency:** 1384.1 ms

---
**item** `214001:558125`  image `COCO_train2014_000000214001.jpg`  (342x640)
**prompt:** 'Detect all instances of: girl on phone. Response must be a JSON array: [{"label": ..., "bbox": [x1, y1, x2, y2]}, ...]'
**raw output:** '[{"label": "girl on phone", "bbox": [0.15, 0.02, 0.52, 0.99]}]'
**parsed:** `[[51.3, 12.8, 177.84, 633.6]]`
**gt:** `{'box_xyxy': [8.630000114440918, 28.760000228881836, 172.5799970626831, 629.9299831390381], 'category': 'person'}`
**score:** `{'iou': 0.699657657519211, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 0.7970229699478621, 'miss_kind': None}`  ✅
**latency:** 1365.9 ms

---
**item** `432372:1067573`  image `COCO_train2014_000000432372.jpg`  (640x426)
**prompt:** 'Detect all instances of: front orange hotdog bottom row center. Response must be a JSON array: [{"label": ..., "bbox": [x1, y1, x2, y2]}, ...]'
**raw output:** '[{"label": "front orange hotdog bottom row center", "bbox": [0.16, 0.53, 0.6, 0.82]}]'
**parsed:** `[[102.4, 225.78, 384.0, 349.32]]`
**gt:** `{'box_xyxy': [156.3699951171875, 272.4700012207031, 445.0400085449219, 397.77000427246094], 'category': 'hot dog'}`
**score:** `{'iou': 0.327187633865383, 'hit_at_50': False, 'center_in_box': True, 'area_ratio': 0.9618060254505717, 'miss_kind': 'found_but_loose'}`  ❌
**latency:** 1518.6 ms

---
**item** `565220:1068936`  image `COCO_train2014_000000565220.jpg`  (500x430)
**prompt:** 'Detect all instances of: hot dog on top. Response must be a JSON array: [{"label": ..., "bbox": [x1, y1, x2, y2]}, ...]'
**raw output:** '[{"label": "hot dog on top", "bbox": [0.0, 0.12, 1.0, 0.87]}]'
**parsed:** `[[0.0, 51.6, 500.0, 374.1]]`
**gt:** `{'box_xyxy': [3.0399999618530273, 15.180000305175781, 484.64000606536865, 239.79000091552734], 'category': 'hot dog'}`
**score:** `{'iou': 0.5069208033850968, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 1.4906790588943508, 'miss_kind': None}`  ✅
**latency:** 1169.5 ms

---
**item** `312943:595522`  image `COCO_train2014_000000312943.jpg`  (510x640)
**prompt:** 'Detect all instances of: giraffe far right. Response must be a JSON array: [{"label": ..., "bbox": [x1, y1, x2, y2]}, ...]'
**raw output:** '[{"label": "giraffe far right", "bbox": [0.79, 0.17, 1.0, 0.8]}]'
**parsed:** `[[402.90000000000003, 108.80000000000001, 510.0, 512.0]]`
**gt:** `{'box_xyxy': [386.4800109863281, 117.83000183105469, 508.1400146484375, 479.27000427246094], 'category': 'giraffe'}`
**score:** `{'iou': 0.7744267117165706, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 0.9820327074004485, 'miss_kind': None}`  ✅
**latency:** 1424.1 ms

---
**item** `472654:1257559`  image `COCO_train2014_000000472654.jpg`  (640x640)
**prompt:** 'Detect all instances of: woman right front. Response must be a JSON array: [{"label": ..., "bbox": [x1, y1, x2, y2]}, ...]'
**raw output:** '[{"label": "woman right front", "bbox": [0.56, 0.15, 0.79, 0.86]}]'
**parsed:** `[[358.40000000000003, 96.0, 505.6, 550.4]]`
**gt:** `{'box_xyxy': [417.0799865722656, 100.43000030517578, 526.3799896240234, 537.6500015258789], 'category': 'person'}`
**score:** `{'iou': 0.5094265834580508, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 1.3996709130688105, 'miss_kind': None}`  ✅
**latency:** 1344.6 ms

---
**item** `244983:172644`  image `COCO_train2014_000000244983.jpg`  (640x480)
**prompt:** 'Detect all instances of: left train. Response must be a JSON array: [{"label": ..., "bbox": [x1, y1, x2, y2]}, ...]'
**raw output:** '[{"label": "left train", "bbox": [0.03, 0.37, 0.62, 0.59]}]'
**parsed:** `[[19.2, 177.6, 396.8, 283.2]]`
**gt:** `{'box_xyxy': [76.58000183105469, 177.97999572753906, 343.00999450683594, 281.5299987792969], 'category': 'train'}`
**score:** `{'iou': 0.6918904322619976, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 1.445315549068513, 'miss_kind': None}`  ✅
**latency:** 1230.1 ms

---
**item** `152273:328489`  image `COCO_train2014_000000152273.jpg`  (640x480)
**prompt:** 'Detect all instances of: cell phone closest to top. Response must be a JSON array: [{"label": ..., "bbox": [x1, y1, x2, y2]}, ...]'
**raw output:** '[{"label": "cell phone closest to top", "bbox": [0.49, 0.27, 0.82, 0.53]}]'
**parsed:** `[[313.6, 129.60000000000002, 524.8, 254.4]]`
**gt:** `{'box_xyxy': [329.54998779296875, 101.9800033569336, 523.9099884033203, 189.64000701904297], 'category': 'cell phone'}`
**score:** `{'iou': 0.36781763266677925, 'hit_at_50': False, 'center_in_box': False, 'area_ratio': 1.5470349420683003, 'miss_kind': 'found_but_loose'}`  ❌
**latency:** 1220.2 ms

---
**item** `253049:1083402`  image `COCO_train2014_000000253049.jpg`  (612x612)
**prompt:** 'Detect all instances of: picuture on the bottom left the cocunut white donut. Response must be a JSON array: [{"label": ..., "bbox": [x1, y1, x2, y2]}, ...]'
**raw output:** '[{"label": "picuture on the bottom left the cocunut white donut", "bbox": [0.0, 0.5, 0.4, 0.99]}]'
**parsed:** `[[0.0, 306.0, 244.8, 605.88]]`
**gt:** `{'box_xyxy': [108.6500015258789, 324.3399963378906, 302.5600051879883, 603.5199890136719], 'category': 'donut'}`
**score:** `{'iou': 0.4245256581751636, 'hit_at_50': False, 'center_in_box': True, 'area_ratio': 1.3560459612121738, 'miss_kind': 'found_but_loose'}`  ❌
**latency:** 1377.8 ms

---
**item** `31812:1086803`  image `COCO_train2014_000000031812.jpg`  (640x426)
**prompt:** 'Detect all instances of: cake with cherries on top. Response must be a JSON array: [{"label": ..., "bbox": [x1, y1, x2, y2]}, ...]'
**raw output:** '[{"label": "cake with cherries on top", "bbox": [0.0, 0.21, 0.23, 0.56]}]'
**parsed:** `[[0.0, 89.46, 147.20000000000002, 238.56000000000003]]`
**gt:** `{'box_xyxy': [0.0, 65.0999984741211, 122.52999877929688, 240.27999114990234], 'category': 'cake'}`
**score:** `{'iou': 0.7266097557321166, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 1.022488707783511, 'miss_kind': None}`  ✅
**latency:** 1296.9 ms

---
**item** `223909:598325`  image `COCO_train2014_000000223909.jpg`  (640x428)
**prompt:** 'Detect all instances of: right gir. Response must be a JSON array: [{"label": ..., "bbox": [x1, y1, x2, y2]}, ...]'
**raw output:** '[{"label": "right gir", "bbox": [0.47, 0.13, 0.94, 0.93]}]'
**parsed:** `[[300.79999999999995, 55.64, 601.5999999999999, 398.04]]`
**gt:** `{'box_xyxy': [306.2300109863281, 23.56999969482422, 585.6400146484375, 427.99999237060547], 'category': 'giraffe'}`
**score:** `{'iou': 0.7950917299452283, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 0.9114362160779377, 'miss_kind': None}`  ✅
**latency:** 1309.6 ms

---
**item** `158701:486441`  image `COCO_train2014_000000158701.jpg`  (640x427)
**prompt:** 'Detect all instances of: left skiier. Response must be a JSON array: [{"label": ..., "bbox": [x1, y1, x2, y2]}, ...]'
**raw output:** '[{"label": "left skiier", "bbox": [0.2, 0.3, 0.41, 0.82]}]'
**parsed:** `[[128.0, 128.1, 262.4, 350.14]]`
**gt:** `{'box_xyxy': [120.0999984741211, 131.94000244140625, 231.54999542236328, 361.6699981689453], 'category': 'person'}`
**score:** `{'iou': 0.687791124649259, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 1.1655548663430844, 'miss_kind': None}`  ✅
**latency:** 1347.3 ms

---
**item** `421488:238044`  image `COCO_train2014_000000421488.jpg`  (640x460)
**prompt:** 'Detect all instances of: man right. Response must be a JSON array: [{"label": ..., "bbox": [x1, y1, x2, y2]}, ...]'
**raw output:** '[{"label": "man right", "bbox": [0.6, 0.13, 0.89, 0.81]}]'
**parsed:** `[[384.0, 59.800000000000004, 569.6, 372.6]]`
**gt:** `{'box_xyxy': [414.4100036621094, 25.3799991607666, 593.6500091552734, 392.1400089263916], 'category': 'person'}`
**score:** `{'iou': 0.645092723587202, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 0.8831364137519104, 'miss_kind': None}`  ✅
**latency:** 1365.3 ms

---
**item** `163828:2165500`  image `COCO_train2014_000000163828.jpg`  (640x512)
**prompt:** 'Detect all instances of: batter. Response must be a JSON array: [{"label": ..., "bbox": [x1, y1, x2, y2]}, ...]'
**raw output:** '[{"label": "batter", "bbox": [0.51, 0.23, 0.76, 0.8]}]'
**parsed:** `[[326.4, 117.76, 486.4, 409.6]]`
**gt:** `{'box_xyxy': [380.32000732421875, 108.69000244140625, 525.9300079345703, 413.1000061035156], 'category': 'person'}`
**score:** `{'iou': 0.5154477077995719, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 1.0534518139634559, 'miss_kind': None}`  ✅
**latency:** 1243.5 ms
