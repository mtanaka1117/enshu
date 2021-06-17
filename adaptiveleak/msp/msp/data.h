#include <stdint.h>
#include <msp430.h>
#include "utils/fixed_point.h"
#ifndef DATA_H_
#define DATA_H_
#define MAX_NUM_SEQ 5u
static const uint32_t DATASET_LENGTH = 1500ul;
#pragma PERSISTENT(DATASET)
static FixedPoint DATASET[1500] = { 8613,-1741,489,-9435,-686,-4954,7817,-1741,887,-7701,-1043,-4231,6292,-1832,1320,-4088,-3355,-2482,6087,-1729,1138,-2174,-4399,-1171,5973,-1388,558,-2983,-4374,-733,5848,-1138,-46,-6145,-3543,-1083,5439,-1092,-455,-7867,-2787,-125,4767,-1263,-614,-7689,-2530,1304,4449,-1798,-444,-6828,-1942,1281,4460,-2571,-46,-4254,-2019,646,4927,-2742,80,-505,-3438,2019,6007,-2549,-159,2462,-4043,3213,7509,-2276,-603,1614,-3666,1804,8875,-1456,-831,-2575,-2935,-1003,9512,-182,-626,-3936,-1994,-778,9318,523,-364,-2847,-1526,1521,10741,-273,-193,-2450,-1454,2400,11662,-671,-205,-3043,-60,1409,12550,-1422,-262,-2490,1331,1404,12334,-1855,-353,1033,2727,4194,12334,-2537,-466,2832,3300,4676,12698,-4130,-831,693,4334,2677,12197,-5575,205,-4979,1942,-518,9591,-5495,819,-2282,1359,-3923,7373,-3823,512,1802,1011,-2257,7862,-2526,273,1269,1183,-15,9262,-2367,455,-563,3080,55,9648,-2856,899,-598,3628,-858,9023,-3550,1092,283,2322,-1872,8363,-3459,1377,2374,1877,-2144,7600,-2890,899,5167,1822,-2057,6656,-2014,375,7184,1847,-1694,5643,-1183,-148,8289,1802,-958,5200,-956,-353,7366,1829,-398,4574,-705,-558,5224,1862,75,4198,-148,-68,4058,2107,1281,4415,34,-57,3508,2772,2915,5302,-250,80,1822,3871,3786,6269,-1399,387,636,4151,2720,7191,-2355,922,878,4011,2007,7839,-2856,1593,2487,3365,1041,8090,-2935,2105,4809,3338,218,8112,-2549,2423,6220,2482,-558,8317,-1377,2469,7721,1784,-460,10445,-102,2162,7619,876,668,11275,-125,2276,5642,1271,280,11071,-205,2947,2817,-683,-1591,11230,-774,2662,4524,-2887,1884,12390,-2992,1764,4952,-3483,3358,9774,-3766,2173,1887,-4479,-1854,10286,1399,262,-3098,1756,-1076,9682,614,193,-3003,-838,1061,9944,-2298,-375,2565,-2670,3801,10706,-4710,-2367,8742,2745,4286,9887,-5143,-2241,8765,7732,2412,7760,-3527,-1536,7506,7842,-540,6133,-1741,-1877,3370,1634,-1256,6713,-569,-2617,-1439,-5517,-355,8306,-626,-3538,-3768,-8084,613,9398,-1604,-3652,-2515,-988,278,9057,-2094,-2435,-1864,3873,-993,8192,-2094,-626,-580,5194,-3053,7134,-1411,125,-25,243,-4872,6656,-558,-11,568,-3831,-4834,6519,23,-740,1061,-5630,-3931,6474,262,-1161,1967,-5234,-2405,6326,250,-1422,2570,-4449,-1071,6372,-125,-1536,2930,-3358,145,6724,-284,-1582,3413,-2374,1073,7020,-296,-1308,3963,-1476,1644,7384,307,-1047,4174,-2487,1654,7384,307,-1047,4043,-3018,1819,7885,1001,-1126,3083,-2570,1736,8055,398,-1149,1764,-1877,1504,8055,398,-1149,1491,-683,1133,8863,637,-1286,2279,1041,1101,9136,592,-1320,1381,2830,1379,9136,592,-1320,303,3876,55,9455,1252,-865,-1696,2007,-1834,10377,796,-57,-3203,-1942,-4281,10377,796,-57,-1326,-1596,-4844,11594,-762,-1263,215,2775,-4842,10126,341,-2037,-791,2139,-4161,10126,341,-2037,-2009,-2695,-3160,9887,-751,-3015,-1439,-4406,-2322,9671,-1343,-3584,-3490,-2767,-1997,8932,-2765,-4289,-4579,2214,-2012,8704,-3072,-3447,-3358,6540,-2347,8397,-2355,-2082,-1073,3918,-1359,8351,-1229,-1035,776,-711,548,8727,-694,-1126,2217,-3160,2257,9091,-375,-1695,2465,-1201,2657,8943,-444,-2287,1311,2775,2122,7999,-159,-2230,-843,4559,1741,6679,125,-1570,-4151,3038,1869,5825,137,-774,-5184,1103,2685,5734,-91,-239,-4887,390,3733,6076,-478,34,-2985,1151,4264,6110,-614,102,-1817,373,4276,6076,-501,125,-2450,-1814,4033,10524,2253,-23,-1591,3688,-1331,11048,387,990,-1396,3776,-1726,10889,-1001,1365,-433,353,-1651,9239,-2651,910,-70,-1321,-1842,7031,-2890,1058,-2349,-2279,-2902,6224,-2230,933,-2164,-2510,-1566,8442,-1377,865,2014,-2985,796,8442,-1377,865,3718,-2802,1584,12026,-717,330,1954,-1076,155,10650,182,262,290,23,-996,10650,182,262,-1899,1229,-1098,8488,421,307,-4751,1997,-1188,6793,592,455,-6998,2187,-1161,6269,899,193,-6175,2127,-555,6804,808,148,-6163,3138,270,7225,364,341,-7214,3633,360,6679,125,319,-7584,3563,666,6440,102,694,-7133,3571,1504,6565,102,1343,-6220,3238,1344,7009,-239,1695,-4999,3218,585,7009,-398,1536,-4046,2107,490,6542,-125,1297,-1439,698,733,7077,785,1240,-916,1376,618,9023,1229,1855,-2542,2537,1299,9774,262,2219,-2752,1504,1514,8943,-580,2128,923,-325,443,8852,-125,2241,3318,-1334,-1108,10240,1229,1502,5702,-2097,293,12311,1308,1229,4214,-761,465,10604,-1047,1365,360,-2415,-2324,8567,-1616,478,2174,-3678,-1234,7782,-2344,-102,-313,-1714,-1314,8158,-2378,23,-1741,145,108,9353,-2276,614,-1434,1261,-358,8897,-2230,1126,-590,253,-781,8203,-1536,1149,125,-773,-588,8021,-1217,944,1214,-1506,-578,7885,-1343,910,2740,-1852,-205,7327,-1422,751,4133,-1837,-140,6679,-1365,637,5745,-1939,-563,6269,-853,671,7391,-2104,-238,6349,-34,319,8284,-2162,696,6918,751,-46,6983,-2447,1359,7612,1263,-159,4979,-2062,1021,7782,1547,23,3333,-1474,801,7373,1547,296,2680,-743,1173,7214,1388,330,2585,-140,1639,7680,1229,466,1681,688,1534,7919,990,853,2407,908,1686,8067,933,808,4992,110,2369,7111,-523,-523,373,-1284,-3103,7111,-182,-558,-50,-1989,-3088,7077,137,-489,-613,-1491,-2362,6690,273,-398,-1096,-911,-1671,6451,102,-216,-1311,-205,-796,6440,23,-216,-978,535,-425,6793,-102,364,-313,-1299,-723,7236,-205,375,180,-3440,-1379,7225,-216,-296,533,-3218,-1686,6975,182,-899,-78,-786,-1669,6963,-159,-1399,-515,-1411,-883,7919,-671,-1968,1441,-1519,1411,9023,-2276,-2970,4601,-2357,4824,10536,-3846,-4847,4439,3145,5017,11401,-4062,-4096,5064,5990,3906,13358,-2367,-3868,6941,240,2550,14290,-1547,-4426,2657,6080,-363,12607,-228,-2708,-5397,6598,-4344,11685,-216,-68,-5625,4101,-4389,11560,-375,762,1869,-268,-1236,10934,148,-80,4904,-2635,-500,10706,535,-1092,3390,-3588,-503,9978,1047,-2719,-398,-2497,-876,9000,967,-2719,-4286,-1426,-1234,7919,205,-2469,-5677,-1694,-1153,6588,-853,-1718,-5550,-1864,-1146,6167,-910,-1638,-4474,-158,-543,5382,-569,-1559,-2064,1304,1286,5302,68,-1627,-691,-838,2712,6007,307,-1832,-828,-1754,3360,6667,46,-1957,-1759,-1221,2705,7214,-296,-1684,-1899,-1021,2092,7646,-239,-1650,-1599,120,2299,7589,-353,-1468,-1399,2089,2139,7145,-319,-1070,-1254,1726,1556,6599,-330,-887,-736,1178,946,6269,-478,-569,868,1296,508,6190,-375,-546,2032,1521,408,6303,-148,-910,2022,1141,473,6417,319,-1354,943,596,673,6406,478,-1650,-353,590,1036,6178,216,-1263,-1509,766,1441,6190,-34,-876,-1081,63,1116,6747,-353,-1024,533,-23,796,7657,-751,-1798,1779,1326,560,8658,-1070,-2970,688,3923,721,9614,-1502,-3698,-2752,4521,120,10149,-1718,-2560,-5117,3736,-738,10422,-2298,-1582,-2967,-473,-788,10422,-2298,-1582,-4186,-1356,145,8158,-1217,-330,5385,-2782,3053,8886,-1695,-68,5417,-2117,2917,8670,-1183,330,5665,-1626,2767,8670,-1183,330,6618,-2104,3998,8556,102,-307,6390,-2877,4802,9944,-523,-512,5590,-2877,5027,9944,-523,-512,3110,-738,4504,11275,-1343,-319,2122,881,3565,12129,-1934,512,4326,2049,3555,12129,-1934,512,6303,2187,3841,12823,-2367,284,4827,1379,4116,13619,-3994,-159,3898,3611,1812,12732,-5643,284,5317,3623,-848,9933,-5302,694,8537,1761,-405,8408,-4142,660,10266,1609,941,8829,-3664,580,8875,2202,518,8943,-3607,853,7179,1244,-893,8533,-3402,887,7199,903,-1376,8238,-3197,637,7779,1364,-1068,8021,-2981,558,9470,1589,-523,7566,-2799,364,10626,1038,-488,6895,-2640,455,10942,330,-748,5894,-2241,444,10982,-736,-1181,4847,-1456,11,12058,-2054,-545,4710,-1013,-614,12718,-2014,1231,4983,-1286,-717,11575,-510,2405,5006,-1422,-535,10574,-88,2209,4733,-1502,375,10459,-1276,2219,4927,-1411,671,10584,-1466,3198,5996,-1206,808,9385,-953,3075,7293,-1172,1138,7969,-123,1746,8101,-1229,1388,6851,-188,185,8169,-1525,1695,6403,-731,-163,7919,-1752,1764,6976,-353,913,7839,-1729,1741,6736,-190,1279,7839,-1422,1843,6065,-495,1261,7817,-1297,1650,5430,-178,1526,7566,-1479,1536,4754,515,1591,7441,-1195,1877,3983,495,1446,7464,-671,2014,3368,-295,1827,7532,-751,1911,2154,-731,2087,7726,-1331,1911,2302,-555,2217,8203,-1832,1718,3195,-100,1719,8522,-1843,1798,2207,500,300,8112,-1263,2185,881,-776,98,8067,-967,2071,1474,-1254,2212,8590,-1559,2002,1459,195,3025,8613,-1957,2423,163,683,2602,8238,-1775,2844,-1686,-911,1689,8192,-1217,2560,-803,-2477,3183 };
#endif
