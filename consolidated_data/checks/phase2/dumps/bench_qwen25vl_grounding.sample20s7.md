# Phase 2 dump — bench_qwen25vl_grounding.sample20s7

source: `/Users/jacobnikolau/Desktop/Claude/results/raw/bench/bench_qwen25vl_grounding.sample20s7.jsonl`  items: 20

**cell summary: parse 20/20  correct 18/20**


---
**item** `297363:216168`  image `COCO_train2014_000000297363.jpg`  (640x480)
**prompt:** 'Detect yellow shirt guy in the image and return its location in the form of coordinates. The format of output should be like {"bbox_2d": [x1, y1, x2, y2], "label": "..."}'
**raw output:** '```json\n[\n\t{"bbox_2d": [403, 95, 644, 476], "label": "yellow shirt guy"}\n]\n```'
**parsed:** `[[403.0, 95.0, 640.0, 476.0]]`
**gt:** `{'box_xyxy': [401.989990234375, 91.83000183105469, 639.2799835205078, 473.22999572753906], 'category': 'person'}`
**score:** `{'iou': 0.9774360708843833, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 0.9977304249775798, 'miss_kind': None}`  ✅
**latency:** 2576.2 ms

---
**item** `68812:1259562`  image `COCO_train2014_000000068812.jpg`  (640x427)
**prompt:** 'Detect man in very front in the image and return its location in the form of coordinates. The format of output should be like {"bbox_2d": [x1, y1, x2, y2], "label": "..."}'
**raw output:** '```json\n[\n\t{"bbox_2d": [87, 109, 356, 415], "label": "man in very front"}\n]\n```'
**parsed:** `[[87.0, 109.0, 356.0, 415.0]]`
**gt:** `{'box_xyxy': [87.31999969482422, 105.55000305175781, 327.20999908447266, 420.28001403808594], 'category': 'person'}`
**score:** `{'iou': 0.869658433664416, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 1.0902432492989726, 'miss_kind': None}`  ✅
**latency:** 1229.8 ms

---
**item** `495344:1394064`  image `COCO_train2014_000000495344.jpg`  (640x426)
**prompt:** 'Detect white bench on right all of the bench is shown in the image and return its location in the form of coordinates. The format of output should be like {"bbox_2d": [x1, y1, x2, y2], "label": "..."}'
**raw output:** '```json\n[\n\t{"bbox_2d": [378, 304, 569, 415], "label": "white bench on right all of the bench is shown in the image"}\n]\n```'
**parsed:** `[[378.0, 304.0, 569.0, 415.0]]`
**gt:** `{'box_xyxy': [379.6000061035156, 305.9800109863281, 601.8899993896484, 426.00000762939453], 'category': 'bench'}`
**score:** `{'iou': 0.758243690163735, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 0.7946626899522513, 'miss_kind': None}`  ✅
**latency:** 1508.6 ms

---
**item** `182347:459025`  image `COCO_train2014_000000182347.jpg`  (500x333)
**prompt:** 'Detect man wearing black shirt in the image and return its location in the form of coordinates. The format of output should be like {"bbox_2d": [x1, y1, x2, y2], "label": "..."}'
**raw output:** '```json\n[\n\t{"bbox_2d": [460, 109, 503, 217], "label": "man wearing black shirt"}\n]\n```'
**parsed:** `[[460.0, 109.0, 500.0, 217.0]]`
**gt:** `{'box_xyxy': [141.42999267578125, 81.33999633789062, 236.4699935913086, 326.7899932861328], 'category': 'person'}`
**score:** `{'iou': 0.0, 'hit_at_50': False, 'center_in_box': False, 'area_ratio': 0.18518861512254742, 'miss_kind': 'found_but_loose'}`  ❌
**latency:** 1236.9 ms

---
**item** `463505:458610`  image `COCO_train2014_000000463505.jpg`  (500x375)
**prompt:** 'Detect woman in the image and return its location in the form of coordinates. The format of output should be like {"bbox_2d": [x1, y1, x2, y2], "label": "..."}'
**raw output:** '```json\n[\n\t{"bbox_2d": [34, 45, 270, 360], "label": "woman"}\n]\n```'
**parsed:** `[[34.0, 45.0, 270.0, 360.0]]`
**gt:** `{'box_xyxy': [32.09000015258789, 50.5099983215332, 283.78000259399414, 368.9200019836426], 'category': 'person'}`
**score:** `{'iou': 0.8968414377520775, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 0.9276195393795901, 'miss_kind': None}`  ✅
**latency:** 1100.7 ms

---
**item** `562063:286973`  image `COCO_train2014_000000562063.jpg`  (468x640)
**prompt:** 'Detect right umbrella in the image and return its location in the form of coordinates. The format of output should be like {"bbox_2d": [x1, y1, x2, y2], "label": "..."}'
**raw output:** '```json\n[\n\t{"bbox_2d": [320, 46, 457, 220], "label": "right umbrella"}\n]\n```'
**parsed:** `[[320.0, 46.0, 457.0, 220.0]]`
**gt:** `{'box_xyxy': [317.4800109863281, 47.27000045776367, 447.15000915527344, 218.26000595092773], 'category': 'umbrella'}`
**score:** `{'iou': 0.8958538332130085, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 1.0751265442322713, 'miss_kind': None}`  ✅
**latency:** 1178.9 ms

---
**item** `203034:1646741`  image `COCO_train2014_000000203034.jpg`  (640x480)
**prompt:** 'Detect short white fridge in the image and return its location in the form of coordinates. The format of output should be like {"bbox_2d": [x1, y1, x2, y2], "label": "..."}'
**raw output:** '```json\n[\n\t{"bbox_2d": [154, 189, 327, 476], "label": "short white fridge"}\n]\n```'
**parsed:** `[[154.0, 189.0, 327.0, 476.0]]`
**gt:** `{'box_xyxy': [244.4499969482422, 310.17999267578125, 325.9499969482422, 480.0], 'category': 'refrigerator'}`
**score:** `{'iou': 0.27041100099893606, 'hit_at_50': False, 'center_in_box': False, 'area_ratio': 3.587414307215718, 'miss_kind': 'found_but_loose'}`  ❌
**latency:** 1233.6 ms

---
**item** `214001:558125`  image `COCO_train2014_000000214001.jpg`  (342x640)
**prompt:** 'Detect girl on phone in the image and return its location in the form of coordinates. The format of output should be like {"bbox_2d": [x1, y1, x2, y2], "label": "..."}'
**raw output:** '```json\n[\n\t{"bbox_2d": [10, 34, 165, 637], "label": "girl on phone"}\n]\n```'
**parsed:** `[[10.0, 34.0, 165.0, 637.0]]`
**gt:** `{'box_xyxy': [8.630000114440918, 28.760000228881836, 172.5799970626831, 629.9299831390381], 'category': 'person'}`
**score:** `{'iou': 0.9268644136608138, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 0.9482881198248013, 'miss_kind': None}`  ✅
**latency:** 1161.7 ms

---
**item** `432372:1067573`  image `COCO_train2014_000000432372.jpg`  (640x426)
**prompt:** 'Detect front orange hotdog bottom row center in the image and return its location in the form of coordinates. The format of output should be like {"bbox_2d": [x1, y1, x2, y2], "label": "..."}'
**raw output:** '```json\n[\n\t{"bbox_2d": [135, 247, 446, 389], "label": "front orange hotdog bottom row center"}\n]\n```'
**parsed:** `[[135.0, 247.0, 446.0, 389.0]]`
**gt:** `{'box_xyxy': [156.3699951171875, 272.4700012207031, 445.0400085449219, 397.77000427246094], 'category': 'hot dog'}`
**score:** `{'iou': 0.7204132788256152, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 1.2209446590710218, 'miss_kind': None}`  ✅
**latency:** 1356.9 ms

---
**item** `565220:1068936`  image `COCO_train2014_000000565220.jpg`  (500x430)
**prompt:** 'Detect hot dog on top in the image and return its location in the form of coordinates. The format of output should be like {"bbox_2d": [x1, y1, x2, y2], "label": "..."}'
**raw output:** '```json\n[\n\t{"bbox_2d": [0, 18, 493, 215], "label": "hot dog on top"}\n]\n```'
**parsed:** `[[0.0, 18.0, 493.0, 215.0]]`
**gt:** `{'box_xyxy': [3.0399999618530273, 15.180000305175781, 484.64000606536865, 239.79000091552734], 'category': 'hot dog'}`
**score:** `{'iou': 0.85923691270185, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 0.8978371527372294, 'miss_kind': None}`  ✅
**latency:** 1104.6 ms

---
**item** `312943:595522`  image `COCO_train2014_000000312943.jpg`  (510x640)
**prompt:** 'Detect giraffe far right in the image and return its location in the form of coordinates. The format of output should be like {"bbox_2d": [x1, y1, x2, y2], "label": "..."}'
**raw output:** '```json\n[\n\t{"bbox_2d": [380, 56, 504, 487], "label": "giraffe far right"}\n]\n```'
**parsed:** `[[380.0, 56.0, 504.0, 487.0]]`
**gt:** `{'box_xyxy': [386.4800109863281, 117.83000183105469, 508.1400146484375, 479.27000427246094], 'category': 'giraffe'}`
**score:** `{'iou': 0.7731369029577638, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 1.2153879147564022, 'miss_kind': None}`  ✅
**latency:** 1271.5 ms

---
**item** `472654:1257559`  image `COCO_train2014_000000472654.jpg`  (640x640)
**prompt:** 'Detect woman right front in the image and return its location in the form of coordinates. The format of output should be like {"bbox_2d": [x1, y1, x2, y2], "label": "..."}'
**raw output:** '```json\n[\n\t{"bbox_2d": [417, 103, 536, 538], "label": "woman right front"}\n]\n```'
**parsed:** `[[417.0, 103.0, 536.0, 538.0]]`
**gt:** `{'box_xyxy': [417.0799865722656, 100.43000030517578, 526.3799896240234, 537.6500015258789], 'category': 'person'}`
**score:** `{'iou': 0.912795159728582, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 1.0832183866297498, 'miss_kind': None}`  ✅
**latency:** 1208.7 ms

---
**item** `244983:172644`  image `COCO_train2014_000000244983.jpg`  (640x480)
**prompt:** 'Detect left train in the image and return its location in the form of coordinates. The format of output should be like {"bbox_2d": [x1, y1, x2, y2], "label": "..."}'
**raw output:** '```json\n[\n\t{"bbox_2d": [76, 180, 345, 280], "label": "left train"}\n]\n```'
**parsed:** `[[76.0, 180.0, 345.0, 280.0]]`
**gt:** `{'box_xyxy': [76.58000183105469, 177.97999572753906, 343.00999450683594, 281.5299987792969], 'category': 'train'}`
**score:** `{'iou': 0.9568040120661506, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 0.9750324083812587, 'miss_kind': None}`  ✅
**latency:** 1183.0 ms

---
**item** `152273:328489`  image `COCO_train2014_000000152273.jpg`  (640x480)
**prompt:** 'Detect cell phone closest to top in the image and return its location in the form of coordinates. The format of output should be like {"bbox_2d": [x1, y1, x2, y2], "label": "..."}'
**raw output:** '```json\n[\n\t{"bbox_2d": [334, 98, 527, 186], "label": "cell phone closest to top"}\n]\n```'
**parsed:** `[[334.0, 98.0, 527.0, 186.0]]`
**gt:** `{'box_xyxy': [329.54998779296875, 101.9800033569336, 523.9099884033203, 189.64000701904297], 'category': 'cell phone'}`
**score:** `{'iou': 0.8832503209719011, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 0.996854112644171, 'miss_kind': None}`  ✅
**latency:** 1291.7 ms

---
**item** `253049:1083402`  image `COCO_train2014_000000253049.jpg`  (612x612)
**prompt:** 'Detect picuture on the bottom left the cocunut white donut in the image and return its location in the form of coordinates. The format of output should be like {"bbox_2d": [x1, y1, x2, y2], "label": "..."}'
**raw output:** '```json\n[\n\t{"bbox_2d": [10, 314, 305, 607], "label": "picuture on the bottom left the cocunut white donut"}\n]\n```'
**parsed:** `[[10.0, 314.0, 305.0, 607.0]]`
**gt:** `{'box_xyxy': [108.6500015258789, 324.3399963378906, 302.5600051879883, 603.5199890136719], 'category': 'donut'}`
**score:** `{'iou': 0.6263179661265507, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 1.5966331066382742, 'miss_kind': None}`  ✅
**latency:** 1439.6 ms

---
**item** `31812:1086803`  image `COCO_train2014_000000031812.jpg`  (640x426)
**prompt:** 'Detect cake with cherries on top in the image and return its location in the form of coordinates. The format of output should be like {"bbox_2d": [x1, y1, x2, y2], "label": "..."}'
**raw output:** '```json\n[\n\t{"bbox_2d": [0, 67, 134, 258], "label": "cake with cherries on top"}\n]\n```'
**parsed:** `[[0.0, 67.0, 134.0, 258.0]]`
**gt:** `{'box_xyxy': [0.0, 65.0999984741211, 122.52999877929688, 240.27999114990234], 'category': 'cake'}`
**score:** `{'iou': 0.8220914398081697, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 1.1923705269210905, 'miss_kind': None}`  ✅
**latency:** 1251.7 ms

---
**item** `223909:598325`  image `COCO_train2014_000000223909.jpg`  (640x428)
**prompt:** 'Detect right gir in the image and return its location in the form of coordinates. The format of output should be like {"bbox_2d": [x1, y1, x2, y2], "label": "..."}'
**raw output:** '```json\n[\n\t{"bbox_2d": [308, 24, 591, 415], "label": "right gir"}\n]\n```'
**parsed:** `[[308.0, 24.0, 591.0, 415.0]]`
**gt:** `{'box_xyxy': [306.2300109863281, 23.56999969482422, 585.6400146484375, 427.99999237060547], 'category': 'giraffe'}`
**score:** `{'iou': 0.9431760724066982, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 0.9792146140050989, 'miss_kind': None}`  ✅
**latency:** 1125.1 ms

---
**item** `158701:486441`  image `COCO_train2014_000000158701.jpg`  (640x427)
**prompt:** 'Detect left skiier in the image and return its location in the form of coordinates. The format of output should be like {"bbox_2d": [x1, y1, x2, y2], "label": "..."}'
**raw output:** '```json\n[\n\t{"bbox_2d": [117, 128, 234, 356], "label": "left skiier"}\n]\n```'
**parsed:** `[[117.0, 128.0, 234.0, 356.0]]`
**gt:** `{'box_xyxy': [120.0999984741211, 131.94000244140625, 231.54999542236328, 361.6699981689453], 'category': 'person'}`
**score:** `{'iou': 0.9144411172619775, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 1.0418925756140613, 'miss_kind': None}`  ✅
**latency:** 1198.5 ms

---
**item** `421488:238044`  image `COCO_train2014_000000421488.jpg`  (640x460)
**prompt:** 'Detect man right in the image and return its location in the form of coordinates. The format of output should be like {"bbox_2d": [x1, y1, x2, y2], "label": "..."}'
**raw output:** '```json\n[\n\t{"bbox_2d": [418, 23, 597, 364], "label": "man right"}\n]\n```'
**parsed:** `[[418.0, 23.0, 597.0, 364.0]]`
**gt:** `{'box_xyxy': [414.4100036621094, 25.3799991607666, 593.6500091552734, 392.1400089263916], 'category': 'person'}`
**score:** `{'iou': 0.8838033785689834, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 0.9285183389291599, 'miss_kind': None}`  ✅
**latency:** 1141.1 ms

---
**item** `163828:2165500`  image `COCO_train2014_000000163828.jpg`  (640x512)
**prompt:** 'Detect batter in the image and return its location in the form of coordinates. The format of output should be like {"bbox_2d": [x1, y1, x2, y2], "label": "..."}'
**raw output:** '```json\n[\n\t{"bbox_2d": [379, 106, 528, 406], "label": "batter"}\n]\n```'
**parsed:** `[[379.0, 106.0, 528.0, 406.0]]`
**gt:** `{'box_xyxy': [380.32000732421875, 108.69000244140625, 525.9300079345703, 413.1000061035156], 'category': 'person'}`
**score:** `{'iou': 0.9465926456193964, 'hit_at_50': True, 'center_in_box': True, 'area_ratio': 1.008457033052496, 'miss_kind': None}`  ✅
**latency:** 1163.3 ms
