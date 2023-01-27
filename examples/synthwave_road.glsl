// Copied from https://www.shadertoy.com/view/7ltcRn
// Created by https://www.shadertoy.com/user/alexdav

// First attempt at a Synthwave Road
// ... using 2D techniques

const vec3 sunsetUp = vec3(1., 0.59, 0.32);
const vec3 sunsetDown = vec3(0.58, 0.11, 0.44);
const vec3 sunUp = vec3(1, 0.9, 0);
const vec3 sunDown = vec3(0.75, 0.21, 0.44);
const float sunSize = 0.7;
const float sunStripe = 0.05;
const float sunStripeOffset = 0.04;
const vec2 sunPos = vec2(0, 0.1);
const float sunDistortFactor = 0.002;
const float sunStripeSpeed = 0.02;

const float horizonY = -.24;
const float roadScale = -0.2;
const float speed = 20.0;
const float bendingRate = 0.1;

const vec3 gridColor = vec3(0.8, 0.5, 0.75);
const vec3 gridColor2 = vec3(0.59, 0.05, 0.45);
const vec3 groundColor = vec3(0.24, 0.11, 0.26);
const vec3 roadColor = vec3(0.05, 0.05, 0.05);
const vec3 roadColor2 = vec3(0.1, 0.1, 0.1);
const vec3 paintColor = vec3(0.05, 0.05, 0.05);
const vec3 paintColor2 = vec3(1, 1, 0.33);
const vec3 centerLineColor = vec3(1, 1, 0.33);
const vec3 buildingsColor = vec3(0, 0, 0);

const float roadWidth = 2.8;
const float roadDetail = 4.0;
const float zoom = 0.05;
const float buildingsScrollSpeed = 20.;
const float buildingsHeight = 0.15;
const float buildingsWidthFactor = 25.;
const float sideLineWidth = 0.2;
const float centerWidth = 0.15;
const float gridLineWidth = 0.1;

const float glowIntensity = 1.2;
const float glowRadius = 0.1;
const float glowColorFactor = 0.4;

#define hash21(p) fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453)

vec4 sky(vec2 p)
{
    vec2 sunDistort = vec2(sin(60. * p.y + iTime * 2.) * sunDistortFactor);
    float sun = length(p + sunDistort - sunPos);
    vec3 skyColor = mix(sunsetDown, sunsetUp, p.y);

    float sunStripe = mod(iTime * sunStripeSpeed + sunStripeOffset + (p.y - horizonY) * (p.y - horizonY), sunStripe);
    vec3 ret = (sun < sunSize) && sunStripe > 0.01 ? mix(sunDown, sunUp, (p.y + sunPos.y) / sunSize) : skyColor;

    float scroll = buildingsScrollSpeed * cos(bendingRate * iTime);
    float height = hash21(floor(scroll + p.xx * buildingsWidthFactor)) * buildingsHeight;
    ret = p.y - horizonY < height ? buildingsColor : ret;

    return vec4(ret, 1.);
}

vec4 road(vec2 p)
{
    vec3 q = vec3(p, 1) / (roadScale - p.y);

    float refl = 0.8 * (1. - abs(p.y - horizonY));
    refl = refl * refl * refl;

    float k = zoom * sin(bendingRate * iTime);
    float bendFactor = k * q.z * q.z;

    float w = abs(q.x + bendFactor);
    float road = sin(roadDetail * q.z + speed * iTime);

    vec3 c;

    if (w > roadWidth) {
        bool vGrid = mod(w - roadWidth, 1.) > (1. - gridLineWidth);
        bool hGrid = (road < 0. && road > -gridLineWidth * 3.);

        vec3 grid = (vGrid || hGrid) ? mix(gridColor, gridColor2, refl) : groundColor;
        float blend = 1. - abs(p.y - horizonY);
        c = mix(grid, groundColor, blend * blend);
    }
    else {
        if (road > 0.) {
            vec3 fragColor = w > (roadWidth - sideLineWidth) ? paintColor : roadColor;
            c = fragColor;
        }
        else {
            vec3 fragColor = w > (roadWidth - sideLineWidth) ? paintColor2 : (w > (centerWidth * 0.5) ? roadColor2 : centerLineColor);
            c = fragColor;
        }

        vec4 invSky = sky(-p + vec2(sin(30. * p.y + iTime * 2.) * 0.01, horizonY));
        c.rgb = (invSky.rgb * refl) + c.rgb * (1. - refl);
    }

    float d = abs(w - roadWidth);
    d = min(d, w);
    d = max(d, road);

    return vec4(c, d);
}

vec3 scene(vec2 fragCoord)
{
    vec2 normFragCoord = ((fragCoord - 0.5 * iResolution.xy) / iResolution.yy) * 2.;

    vec4 c;
    vec3 glowColor;

    if (normFragCoord.y > horizonY) {
        c = sky(normFragCoord);
    }
    else {
        vec4 skyColor = sky(normFragCoord);
        vec4 roadColor = road(normFragCoord);

        c = roadColor;
        glowColor = skyColor.a < roadColor.a ? sunDown : paintColor2;
        c.a = min(skyColor.a, roadColor.a);

        c.rgb = roadColor.rgb;

        float glow = pow(glowRadius / c.a, glowIntensity);
        c.rgb += glow * glowColor * glowColorFactor;
    }

    return c.rgb;
    //return 1. - c.aaa;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec3 c = scene(fragCoord);
    fragColor.xyz = c.xyz;
}
