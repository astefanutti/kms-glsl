// Copied from https://www.shadertoy.com/view/ss2BDc
// Created by https://www.shadertoy.com/user/Shane

/*

	Hexagon Triangle Wedge Pattern
	------------------------------

    I haven't posted anything in a while due to Shadertoy having problems
    dealing with excess traffic -- If you're going to have a website problem,
    too much popularity would be the one you'd want to have. :) Having said
    that, I doubt it's been a fun couple of weeks for the creators.

    Anyway, this is a simple rendering of one of the many cliche hexagon grid
    based patterns that exist. I'm sure most have seen this one around.

    There's not really much to it: Create a hexagon grid, render some hexagons
    in the centers, then construct triangles around the hexagon edges in a
    pinwheel fashion. How you do that is down to common sense. If you know
    how to move, rotate and render 2D objects, then it shouldn't give you too
    much trouble.

    The default pattern is kind of interesting, given the fact that the triangles
    look like they infinitly overlap one another. There's also an option below
    to display a semi-regular hexagon triangle pinwheel arrangement.

    There would be faster ways to render the pattern. The brute force 7-tap
    method I've employed works well enough, but you could get that number down.
    I also have an extruded version ready to go. It was surprisingly easy to
    put together, and I intend to post that a little later.



    Other Tiled Pattern Examples:

    // Here's a much more interesting irregular tiling.
    Moebius Lizard - iapafoto
    https://www.shadertoy.com/view/wtjyz1

    // Heaps of different tiled configurations.
    Wythoff Uniform Tilings + Duals - fizzer
    https://www.shadertoy.com/view/3tyXWw

    // A decent irregular tiling example.
    escherized tiling 2 (WIP) - Fabrice
    https://www.shadertoy.com/view/lsdBR7

*/


// Color scheme - White: 0, Random color: 1, Ordered color: 2.
#define COLOR 1

// A pinwheel arrangement with wedge looking triangles. Commenting this
// out will result in a regular hexagon triangle pinwheel arrangement.
#define WEDGE

// Show the hexagon grid that the pattern is based on. Knowing where
// the cell boundary lies can be helpful.
//#define SHOW_GRID


// Standard 2D rotation formula.
mat2 rot2(in float a){ float c = cos(a), s = sin(a); return mat2(c, -s, s, c); }

float hash21(vec2 p){

    //return texture(iChannel0, p).x;
    // IQ's vec2 to float hash.
    //return fract(sin(dot(p, vec2(57.609, 27.781)))*43758.5453);

    // Using a very slight variation on Dave Hoskin's hash formula,
    // which can be found here: https://www.shadertoy.com/view/4djSRW
    vec3 p3 = fract(vec3(p.xyx)*.1031);
    p3 += dot(p3, p3.yzx + 43.123);
    return fract((p3.x + p3.y)*p3.z);
}

// Flat top hexagon scaling.
const vec2 s = vec2(1.7320508, 1);

// Hexagon edge and vertex IDs. They're useful for neighboring edge comparisons,
// etc. Multiplying them by "s" gives the actual vertex postion.
//
// Vertices and edges: Clockwise from the left.
//
// Note that these are all six times larger than usual. We're doing this to
// get rid of decimal places, especially those that involve division by three.
// I't a common accuracy hack. Unfortunately, "1. - 1./3." is not always the
// same as "2./3." on a GPU.

// Multiplied by 12 to give integer entries only.
const vec2[6] vID = vec2[6](vec2(-4, 0), vec2(-2, 6), vec2(2, 6),
vec2(4, 0), vec2(2, -6), vec2(-2, -6));

const vec2[6] eID = vec2[6](vec2(-3, 3), vec2(0, 6), vec2(3),
vec2(3, -3), vec2(0, -6), vec2(-3));

// Hexagonal bound: Not technically a distance function, but it's
// good enough for this example.
float getHex(vec2 p){

    // Flat top hexagon.
    return max(dot(abs(p.xy), s/2.), abs(p.y*s.y));
}

// Triangle bound.
float getTri(vec2 p){

    p.x = abs(p.x);
    return max(dot(p, s/2.), -p.y*s.y);
}

// Hexagonal grid coordinates. This returns the local coordinates and the cell's center.
// The process is explained in more detail here:
//
// Minimal Hexagon Grid - Shane
// https://www.shadertoy.com/view/Xljczw
//
vec4 getGrid(vec2 p){
    vec4 ip = floor(vec4(p/s, p/s - .5)) + .5;
    vec4 q = p.xyxy - vec4(ip.xy, ip.zw + .5)*s.xyxy;
    return dot(q.xy, q.xy)<dot(q.zw, q.zw)? vec4(q.xy, ip.xy) : vec4(q.zw, ip.zw + .5);

}

void mainImage(out vec4 fragColor, in vec2 fragCoord){
    // Aspect correct screen coordinates.
    float res = min(iResolution.y, 800.);
    vec2 uv = (fragCoord.xy - iResolution.xy*.5)/res;

    // Global scale factor.
    #ifdef WEDGE
    const float sc = 3.;
    #else
    const float sc = 4.5;
    #endif
    // Smoothing factor.
    float sf = sc/res;

    // Scene rotation, scaling and translation.
    mat2 sRot = mat2(1, 0, 0, 1);//rot2(-3.14159/24.); // Scene rotation.
    vec2 camDir = sRot*normalize(s); // Camera movement direction.
    vec2 ld = sRot*vec2(-cos(3.14159/5.), -sin(3.14159/5.)); // Light direction.//-.5, -1.732
    vec2 p = sRot*uv*sc + camDir*s.xy*iTime/6.;

    // Hexagonal grid coordinates.
    vec4 p4 = getGrid(p);


    // Rendering the grid boundaries, or just some black hexagons in the center.
    float gHx = getHex(p4.xy);

    float df = 1e5, dfHi = 1e5;

    // Cell object ID and cell arrangement ID.
    vec2 id;
    float tID;

    // A cell ratio factor for the two arrangements.
    #ifdef WEDGE
    const float divF = 1./6.;
    #else
    const float divF = 1./4.;
    #endif

    // Set the initial ID and minimum distance to the central hexagon.
    df = gHx - divF;
    #ifdef WEDGE
    id = p4.zw*18.;
    #else
    id = p4.zw*12.;
    #endif

    // Pinwheel ID for the hexagon: There are seven objects per cell. The
    // central hexagon ID is the highest and the surrounding pinwheel objects
    // are numbered zero through to five.
    tID = 6.;

    // Offset hexagon for highlighting purposes.
    dfHi = getHex(p4.xy - ld*.001) - divF;

    // Iterate through all six sides of the hexagon cell.
    for(int i = min(0, iFrame); i<6; i++){

        #ifdef WEDGE

        // Triangle 1 central offset index. These numbers have been multiplied
        // by 18 to produce integers for more index accuracy... It's related to
        // GPUs giving different results for "1. - 1./3." and "2./3.".
        vec2 indx1 = vID[i];

        // Local coordinates.
        vec2 q = p4.xy - indx1*s/18.;
        // The sign matters for bump mapping, etc.
        mat2 mR = rot2(6.2831/6.*float(i));
        // Triangle one (and highlight field).
        float tri1 = getTri(mR*q) - divF;
        float tri1B = getTri(mR*(q - ld*.001)) - divF;

        // Go to the neighboring cell and retrieve the opposite overlapping
        // triangle by rotating forward 4 vertices.
        //
        // Triangle 2 central offset index.
        vec2 indx2 = eID[i]*3. + vID[(i + 4)%6];
        // Local coordinates.
        q = p4.xy - indx2*s/18.;

        // The neighboring triangle (and highlight field). See the image
        // for a clearer picture.
        float tri2 = getTri(mR*q) - divF;
        float tri2B = getTri(mR*(q - ld*.001)) - divF;
        // Using the neighboring triangle to chop a little wedge out of
        // the main triangle. Obviously, that's what gives it a V-shape.
        tri1 = max(tri1, -tri2);
        tri1B = max(tri1B, -tri2B);

        // Set the minimum distance and IDs for the inner object.
        if(tri1<df) {

            df = tri1;
            tID = float(i);
            id = p4.zw*18. + indx1;
        }

        // Set the minimum distance and IDs for the outer object. If you
        // don't include this, the neighboring triangle won't fill in the
        // V-shape's wedge gap.
        if(tri2<df) {

            df = tri2;
            tID =  float((i + 4)%6);
            id = p4.zw*18. + indx2;
        }

        // Offset distance field value for highlighting.
        dfHi = min(dfHi, min(tri1B, tri2B));

        #else

        vec2 indx1 = vID[i]; // Vertex postion index.
        vec2 q = p4.xy - indx1*s/12.; // Local coordinates.
        mat2 mR = rot2(6.2831/6.*float(i));
        // Triangle one (and highlight field).
        float tri1 = getTri(mR*q) - divF; //1./.8660254+1.;//
        float tri1B = getTri(mR*(q - ld*.001)) - divF;

        // Set the minimum distance and IDs for the inner object.
        if(tri1<df) {

            df = tri1;
            tID = float(i);
            id = p4.zw*12. + indx1;
        }

        // Offset distance field value for highlighting.
        dfHi = min(dfHi, tri1B);

        #endif
    }

    // Giving the object some edging and rescaling the ID
    #ifdef WEDGE
    float ew = .018;
    id /= 18.;
    #else
    float ew = .025;
    id /= 12.;
    #endif
    //
    df += ew;
    dfHi += ew;

    // Using the IDs for coloring.
    #if COLOR == 1
    float rnd2 = hash21(id);
    vec3 tCol = .5 + .45*cos(6.2831*rnd2 + vec3(0, 1, 2)*1.5);
    // if(tID>5.5) tCol = vec3(.8); // White hexagons.
    #elif COLOR == 2
    #ifndef WEDGE
    // This is a bit of a cop-out, but I didn't feel like arranging for all
    // colors to line up in a manner similar to the ordered wedge arrangement...
    // It could be done by a less lazy person though. :)
    if(tID<5.5) tID = mod(tID, 2.) + 2.;
    #endif
    vec3 tCol = .5 + .45*cos(6.2831*tID/6. + vec3(0, 1, 2)*1.5 + .5);
    if(tID>5.5) tCol = vec3(.8); // White hexagons.
    #else
    vec3 tCol = vec3(.8);
    #endif

    // Evening the color intensity a bit.
    //tCol /= (.75 + dot(tCol, vec3(.299, .587, .114))*.5);

    /*
    // Textures work too, but they're not used here.
    vec3 tx = texture(iChannel0, p/8.).xyz; tx *= tx;
    tx = smoothstep(-.1, .5, tx);
    vec3 tx2 = texture(iChannel1, id/8.).xyz; tx2 *= tx2;
    tx2 = smoothstep(-.1, .5, tx2);
    tCol *= tx*tx2*1.5;
    */

    // Applying some directional derivative based highlighting.
    float b = max(dfHi - df, 0.)/.001;
    tCol = tCol*.65 + mix(tCol, vec3(1), .05)*b*b*.65;


    // Subtle line pattern overlay.
    vec2 ruv = rot2(-3.14159/3.)*p;
    float lSc = (120./sc);
    float pat = (abs(fract(ruv.x*lSc - .5) - .5) - .125)/lSc/2.;
    tCol *= smoothstep(0., sf, pat)*.35 + 1.;

    // Applying the object color to the background.
    vec3 col = mix(vec3(0), tCol, (1. - smoothstep(0., sf, df)));

    #ifdef SHOW_GRID
    col = mix(col, vec3(0), (1. - smoothstep(0., sf*8., abs(gHx - .5) - .012))*.35);
    col = mix(col, vec3(0), (1. - smoothstep(0., sf, abs(gHx - .5) - .012)));
    col = mix(col, vec3(1), (1. - smoothstep(0., sf, abs(gHx - .5) - .003)));
    #endif

    // Rough gamma correction.
    fragColor = vec4(sqrt(max(col, 0.)), 1);
}
