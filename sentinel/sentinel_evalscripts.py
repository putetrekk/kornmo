'''
All 12 bands available from the Sentinel2 satellites
'''
all_bands = '''
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"],
      units: "REFLECTANCE"
    }],
    output: {
      id: "default",
      bands: 12,
      sampleType: SampleType.FLOAT32
    }
  }
}
function evaluatePixel(sample) {
    return [ sample.B01, sample.B02, sample.B03, sample.B04, sample.B05, sample.B06, sample.B07, sample.B08, sample.B8A, sample.B09, sample.B11, sample.B12]
}
'''

'''
All 12 bands available from the Sentinel2 satellites with a datamask
'''
all_bands_with_mask = '''
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12", "dataMask"],
      units: "REFLECTANCE"
    }],
    output: {
      id: "default",
      bands: 13,
      sampleType: SampleType.FLOAT32
    }
  }
}
function evaluatePixel(sample) {
    return [ sample.B01, sample.B02, sample.B03, sample.B04, sample.B05, sample.B06, sample.B07, sample.B08, sample.B8A, sample.B09, sample.B11, sample.B12]
}
'''


'''
True color

https://custom-scripts.sentinel-hub.com/sentinel-2/true_color/
'''
true_color = """
//VERSION=3
function setup(){
  return{
    input: ["B02", "B03", "B04", "dataMask"],
    output: {bands: 4}
  }
}

function evaluatePixel(sample){
  // Set gain for visualisation
  let gain = 2.5;
  // Return RGB
  return [sample.B04 * gain, sample.B03 * gain, sample.B02 * gain, sample.dataMask];
}
"""

'''
Natural color

The natural color product tries to represent spectral responses of the 
satellite bands so as to match the color perceived by the human eye.

https://custom-scripts.sentinel-hub.com/sentinel-2/natural_color/
'''
natural_color = """
function sum(a, b) {
    return a + b;
}

function zip(a, b, f) {
  return a.map(function(ai,i){return f(ai, b[i]);});
}

function mapConst(arr, c, f) {
  return arr.map(function(ai,i){return f(ai, c, i);});
}

function dotSS(a, b) {
  return a * b;
}

//vector * scalar
function dotVS(v, s) {
  return mapConst(v, s, dotSS);
}

//vector . vector
function dotVV(a, b) {
  return zip(a, b, dotSS).reduce(sum);
}

//matrix . vector
function dotMV(A, v) {
  return mapConst(A, v, dotVV);
}

function adj(C) {
  return C < 0.0031308 ? (12.92 * C) : (1.055 * Math.pow(C, 0.41666) - 0.055);
}

function labF(t) {
    return t > 0.00885645 ? Math.pow(t,1.0/3.0) : (0.137931 + 7.787 * t);
}

function invLabF(t) {
    return t > 0.2069 ? (t*t*t) : (0.12842 * (t - 0.137931));
}

function XYZ_to_Lab(XYZ) {
  var lfY = labF(XYZ[1]);
  return [(116.0 * lfY - 16)/100,
          5 * (labF(XYZ[0]) - lfY),
          2 * (lfY - labF(XYZ[2]))];
}

function Lab_to_XYZ(Lab) {
  var YL = (100*Lab[0] + 16)/116;
  return [invLabF(YL + Lab[1]/5.0),
          invLabF(YL),
          invLabF(YL - Lab[2]/2.0)];
}

function XYZ_to_sRGBlin(xyz) {
    return dotMV([[3.240, -1.537, -0.499], [-0.969, 1.876, 0.042], [0.056, -0.204, 1.057]], xyz);
}

function XYZ_to_sRGB(xyz) {
    return XYZ_to_sRGBlin(xyz).map(adj);
}

function Lab_to_sRGB(Lab) {
  return XYZ_to_sRGB(Lab_to_XYZ(Lab));
}

function getSolarIrr() {
  return [B02, 0.939*B03, 0.779*B04];
}

function S2_to_XYZ(rad, T, gain) {
  return dotVS(dotMV(T, rad), gain);
}

function ProperGamma_S2_to_sRGB(rad, T, gg, gamma, gL) {
  var XYZ = S2_to_XYZ(rad, T, gg);
  var Lab = XYZ_to_Lab(XYZ);
  var L = Math.pow(gL * Lab[0], gamma);
  return Lab_to_sRGB([L, Lab[1], Lab[2]]);
}

var T = [
  [0.268,0.361,0.371],
  [0.240,0.587,0.174],
  [1.463,-0.427,-0.043]
];

// Gamma and gain parameters
var gain = 2.5;
var gammaAdj = 2.2;
var gainL = 1;

return ProperGamma_S2_to_sRGB(getSolarIrr(), T, gain, gammaAdj, gainL);
"""


'''
NDVI (Normalized difference vegetation index)

The well known and widely used NDVI is a simple, but effective index for
quantifying green vegetation. It normalizes green leaf scattering in 
Near Infra-red wavelengths with chlorophyll absorption in red wavelengths.

https://custom-scripts.sentinel-hub.com/sentinel-2/ndvi/
'''
ndvi = """
let ndvi = (B08 - B04) / (B08 + B04);

return colorBlend(ndvi,
   [-0.2, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 ],
   [[0, 0, 0],							   //  < -.2 = #000000 (black)
    [165/255,0,38/255],        //  -> 0 = #a50026
    [215/255,48/255,39/255],   //  -> .1 = #d73027
    [244/255,109/255,67/255],  //  -> .2 = #f46d43
    [253/255,174/255,97/255],  //  -> .3 = #fdae61
    [254/255,224/255,139/255], //  -> .4 = #fee08b
    [255/255,255/255,191/255], //  -> .5 = #ffffbf
    [217/255,239/255,139/255], //  -> .6 = #d9ef8b
    [166/255,217/255,106/255], //  -> .7 = #a6d96a
    [102/255,189/255,99/255],  //  -> .8 = #66bd63
    [26/255,152/255,80/255],   //  -> .9 = #1a9850
    [0,104/255,55/255]         //  -> 1.0 = #006837
   ]);
"""


'''
SAVI (Soil Adjusted Vegetation Index)

https://custom-scripts.sentinel-hub.com/sentinel-2/savi/
'''
savi = """
// Soil Adjusted Vegetation Index  (abbrv. SAVI)
// General formula: (800nm - 670nm) / (800nm + 670nm + L) * (1 + L)
// URL https://www.indexdatabase.de/db/si-single.php?sensor_id=96&rsindex_id=87
// Initialize parameters

let L = 0.428; // L = soil brightness correction factor could range from (0 -1)
let index = (B08 - B04) / (B08 + B04 + L) * (1.0 + L); // calculate savi index

// using colorblend visualization  in EO browser
return colorBlend   // Start of colorBlend function
(index,	            //  specify the index
     [ 0,0.1, 0.2,0.4, 0.5,0.7,1], //  specify the borders
     [ [0.69,0.88,0.90],   // specify RGB colors
       [0.74,0.72,0.42],
       [0.60,0.80,0.20],
       [0.13, 0.54, 0.13],
       [0, 0.50, 0],
       [0, 0.39, 0],
       [0, 0.29, 0],
     ]
);
"""


'''
False color infrared

The False color infrared composite maps near-infrared spectral band B8 with red 
and green bands, B4 and B3, to sRGB components directly. It is most commonly used
to assess plant density and healht, as plants reflect near infrared and green light,
while absorbing red. Since they reflect more near infrared than green, plant-covered
land appears deep red. Denser plant growth is darker red. Cities and exposed ground
are gray or tan, and water appears blue or black.

https://custom-scripts.sentinel-hub.com/sentinel-2/false_color_infrared/
'''
false_color_infrared = """
let gain = 2.5;
return [B08, B04, B03].map(a => gain * a);
"""
